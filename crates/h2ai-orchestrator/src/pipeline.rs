use crate::engine::EngineInput;
use crate::error::OrchestratorError;
use crate::mape_k::{PipelineOutcome, PipelineParams, PipelineWaveResult, WaveEvents};
use crate::phases;
use crate::phases::bootstrap::Output as BootstrapOutput;
use crate::phases::complexity::Output as ComplexityOutput;
use crate::phases::domain_coverage::Output as DomainCovOutput;
use crate::verification::EvalCache;
use async_nats::Client as NatsClient;
use chrono::Utc;
use futures::StreamExt;
use h2ai_memory::provider::MemoryProvider;
use h2ai_nats::subjects::{agent_telemetry_subject, ephemeral_task_subject, task_result_subject};
use h2ai_provisioner::provider::AgentProvider;
use h2ai_telemetry::provider::AuditProvider;
use h2ai_telemetry::redaction::redact_event;
use h2ai_types::agent::{
    AgentDescriptor, AgentTelemetryEvent, ContextPayload, TaskPayload, TaskResult,
};
use h2ai_types::identity::{AgentId, TaskId};
use h2ai_types::sizing::TauValue;
use std::time::Duration;

pub struct OrchestratorPipeline<M, P, A> {
    memory: M,
    provisioner: P,
    auditor: A,
    nats: NatsClient,
}

impl<M, P, A> OrchestratorPipeline<M, P, A>
where
    M: MemoryProvider,
    P: AgentProvider,
    A: AuditProvider,
{
    pub fn new(memory: M, provisioner: P, auditor: A, nats: NatsClient) -> Self {
        Self {
            memory,
            provisioner,
            auditor,
            nats,
        }
    }

    async fn assemble_context(&self, session_id: &str) -> Result<String, OrchestratorError> {
        let history = self
            .memory
            .get_recent_history(session_id, 10)
            .await
            .map_err(|e| OrchestratorError::Memory(e.to_string()))?;
        let context = history
            .iter()
            .filter_map(|v| v.get("content").and_then(|c| c.as_str()).map(str::to_owned))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(context)
    }

    pub async fn execute(
        &self,
        session_id: &str,
        instructions: &str,
        agent: AgentDescriptor,
        tau: TauValue,
        max_tokens: u64,
    ) -> Result<TaskId, OrchestratorError> {
        let context = self.assemble_context(session_id).await?;

        let task_id = TaskId::new();
        let agent_id = AgentId::from(task_id.to_string());
        let payload = TaskPayload {
            task_id: task_id.clone(),
            agent_id: agent_id.clone(),
            agent: agent.clone(),
            instructions: instructions.to_string(),
            context: ContextPayload::Inline(context),
            tau,
            max_tokens,
            wave_mode: h2ai_types::agent::WaveMode::Normal,
        };

        self.provisioner
            .ensure_agent_capacity(&agent, 1)
            .await
            .map_err(|e| OrchestratorError::Provision(e.to_string()))?;

        let subject = ephemeral_task_subject(&task_id);
        let payload_json = serde_json::to_string(&payload)
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;
        self.nats
            .publish(subject, payload_json.into())
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        Ok(task_id)
    }

    /// Commit a completed TaskResult to memory and flush audit log.
    pub async fn finalize(
        &self,
        session_id: &str,
        result: &TaskResult,
    ) -> Result<(), OrchestratorError> {
        let memory_entry = serde_json::json!({
            "role": "assistant",
            "content": result.output,
            "task_id": result.task_id.to_string(),
            "token_cost": result.token_cost,
        });
        self.memory
            .commit_new_memories(session_id, vec![memory_entry])
            .await
            .map_err(|e| OrchestratorError::Memory(e.to_string()))?;

        self.auditor
            .flush()
            .await
            .map_err(|e| OrchestratorError::Telemetry(e.to_string()))?;

        Ok(())
    }

    /// Record a telemetry event, applying redaction first.
    pub async fn record_telemetry(
        &self,
        event: AgentTelemetryEvent,
    ) -> Result<(), OrchestratorError> {
        let redacted = redact_event(event);
        self.auditor
            .record_event(redacted)
            .await
            .map_err(|e| OrchestratorError::Telemetry(e.to_string()))?;
        Ok(())
    }

    /// Full dispatch-and-await pipeline.
    ///
    /// Publishes TaskPayload, subscribes to telemetry and result subjects, drives a
    /// `tokio::select!` loop routing telemetry to the AuditProvider and returning the
    /// TaskResult once received. Finalizes (commits memory + flushes audit) on success.
    /// Returns `Err(Timeout)` if no result arrives within `timeout`.
    pub async fn execute_and_await(
        &self,
        session_id: &str,
        instructions: &str,
        agent: AgentDescriptor,
        tau: TauValue,
        max_tokens: u64,
        timeout: Duration,
    ) -> Result<TaskResult, OrchestratorError> {
        let context = self.assemble_context(session_id).await?;

        let task_id = TaskId::new();
        let agent_id = AgentId::from(task_id.to_string());
        let payload = TaskPayload {
            task_id: task_id.clone(),
            agent_id: agent_id.clone(),
            agent: agent.clone(),
            instructions: instructions.to_string(),
            context: ContextPayload::Inline(context),
            tau,
            max_tokens,
            wave_mode: h2ai_types::agent::WaveMode::Normal,
        };

        self.provisioner
            .ensure_agent_capacity(&agent, 1)
            .await
            .map_err(|e| OrchestratorError::Provision(e.to_string()))?;

        let payload_json = serde_json::to_string(&payload)
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;
        self.nats
            .publish(ephemeral_task_subject(&task_id), payload_json.into())
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        let mut telemetry_sub = self
            .nats
            .subscribe(agent_telemetry_subject(&agent_id))
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        let mut result_sub = self
            .nats
            .subscribe(task_result_subject(&task_id))
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            tokio::select! {
                msg = result_sub.next() => {
                    match msg {
                        Some(msg) => {
                            let result = serde_json::from_slice::<TaskResult>(&msg.payload)
                                .map_err(|e| OrchestratorError::Deserialize(e.to_string()))?;
                            self.finalize(session_id, &result).await?;
                            return Ok(result);
                        }
                        None => return Err(OrchestratorError::Transport(
                            "result subject closed unexpectedly".into(),
                        )),
                    }
                }
                msg = telemetry_sub.next() => {
                    if let Some(msg) = msg {
                        if let Ok(event) = serde_json::from_slice::<AgentTelemetryEvent>(&msg.payload) {
                            if let Err(e) = self.record_telemetry(event).await {
                                tracing::warn!("telemetry record failed: {e}");
                            }
                        }
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(OrchestratorError::Timeout {
                        task_id: task_id.to_string(),
                    });
                }
            }
        }
    }
}

// ── ExecutionPipeline ──────────────────────────────────────────────────────────

/// Macro to unwrap a `StepResult`, returning `PipelineWaveResult` on early-exit or fatal.
macro_rules! phase {
    ($expr:expr, $events:expr) => {
        match $expr {
            crate::phases::StepResult::Done(out) => out,
            crate::phases::StepResult::EarlyExit(r) => {
                return PipelineWaveResult {
                    outcome: PipelineOutcome::EarlyExit(r),
                    events: $events,
                }
            }
            crate::phases::StepResult::Fatal(e) => {
                return PipelineWaveResult {
                    outcome: PipelineOutcome::Fatal(e),
                    events: $events,
                }
            }
        }
    };
}

/// Stateless pipeline: sequences the per-wave phase modules for one execution wave.
///
/// Constructed once before the retry loop; `run()` is called once per wave.
/// Holds references to the pre-loop phase outputs (bootstrap, complexity, domain coverage)
/// that are constant across all waves.
pub struct ExecutionPipeline<'a> {
    input: &'a EngineInput<'a>,
    bootstrap: &'a BootstrapOutput,
    complexity: &'a ComplexityOutput,
    _domain_cov: &'a DomainCovOutput,
    /// Cross-wave eval cache for verification; cheap to clone (Arc-backed).
    task_eval_cache: EvalCache,
}

impl<'a> ExecutionPipeline<'a> {
    /// Create an `ExecutionPipeline` from pre-loop phase outputs.
    pub fn new(
        input: &'a EngineInput<'a>,
        bootstrap: &'a BootstrapOutput,
        complexity: &'a ComplexityOutput,
        domain_cov: &'a DomainCovOutput,
        task_eval_cache: EvalCache,
    ) -> Self {
        Self {
            input,
            bootstrap,
            complexity,
            _domain_cov: domain_cov,
            task_eval_cache,
        }
    }

    /// Run all per-wave phases for one MAPE-K iteration.
    ///
    /// Returns a `PipelineWaveResult` whose `outcome` is one of:
    /// - `Resolved(MergeOutput)` — success; the controller assembles `EngineOutput`.
    /// - `EarlyExit(ExitReason)` — retryable failure; the controller decides next action.
    /// - `Fatal(EngineError)` — non-retryable; the controller propagates the error.
    pub async fn run(&self, params: PipelineParams, retry_count: usize) -> PipelineWaveResult {
        // Initialise WaveEvents with carry-forward SRANI state so early-exit paths
        // don't accidentally reset the EMA/count/tier to zero.
        let mut events = WaveEvents {
            srani_last_wave_fired: params.srani_last_wave_fired,
            srani_tier_updated: params.srani_tier,
            srani_ema_cfi_updated: params.srani_ema_cfi,
            srani_count_updated: params.srani_count as usize,
            ..WaveEvents::default()
        };

        let task_id = &self.input.task_id;
        let retry_count_u32 = retry_count as u32;

        // Derive the active context for Phase 3 from retry_context or base system_context.
        let active_ctx: String = params
            .retry_context
            .as_deref()
            .unwrap_or(&self.bootstrap.system_context)
            .to_owned();

        // ── Phase 2: Topology Provisioning ─────────────────────────────────────
        let topology_out = phase!(
            phases::topology::run(phases::topology::Input {
                engine_input: self.input,
                force_topology: params.force_topology.clone(),
                tau_reduction_factor: params.tau_reduction_factor,
                tau_spread_factor: params.tau_spread_factor,
                retry_count: retry_count_u32,
                assessed_quadrant: self.complexity.assessed_quadrant,
                cg_mean: self.complexity.cg_mean,
                n_max_ceiling: self.complexity.n_max_ceiling,
                explorer_adapter_kind: &self.bootstrap.explorer_adapter_kind,
                pending_tombstone: params.pending_tombstone.clone(),
                n_agents: params.optimizer.n_agents,
            }),
            events
        );

        if retry_count > 0 {
            events.topology_retry_event = Some(topology_out.provisioned.clone());
        }

        let provisioned = &topology_out.provisioned;
        let explorer_count = topology_out.explorer_count;
        let p_mean = topology_out.p_mean;
        let rho_mean = topology_out.rho_mean;
        let attribution_basis = topology_out.attribution_basis;

        // ── Phase 2.5: Multiplication Condition Gate ────────────────────────────
        self.input.store.set_phase(
            task_id,
            crate::task_store::TaskPhase::MultiplicationCheck,
            explorer_count,
            retry_count_u32,
        );

        phase!(
            phases::multiply::run(phases::multiply::Input {
                engine_input: self.input,
                provisioned,
                baseline_competence: p_mean,
                error_correlation: rho_mean,
                retry_count: retry_count_u32,
            }),
            events
        );

        // ── Phase 2.6: Pool Diversity Guard ─────────────────────────────────────
        phase!(
            phases::diversity::run(phases::diversity::Input {
                engine_input: self.input,
                provisioned,
            }),
            events
        );

        // ── Phase 3: Parallel Generation ────────────────────────────────────────
        let gen_out = phase!(
            phases::generation::run(phases::generation::Input {
                engine_input: self.input,
                task_id,
                retry_count: retry_count_u32,
                active_ctx,
                system_context: self.bootstrap.system_context.clone(),
                system_context_with_rubric: self.bootstrap.system_context_with_rubric.clone(),
                explorer_count,
                provisioned,
                pending_tombstone: params.pending_tombstone.clone(),
                adapter_rotation_offset: params.adapter_rotation_offset,
                leader_context: params.leader_context.clone(),
                prev_assembled_contexts: params.prev_assembled_contexts.clone(),
                compression_adapter: self.input.compression_adapter.clone(),
                stable_cache: self.input.stable_cache.clone(),
            })
            .await,
            events
        );

        events
            .failed_proposals
            .extend(gen_out.failed_proposals.iter().cloned());
        events
            .researcher_grounding_events
            .extend(gen_out.researcher_grounding_events.iter().cloned());
        events.assembled_contexts = gen_out.assembled_contexts.clone();
        let tau_values = gen_out.tau_values.clone();
        let tao_turns_mean = gen_out.tao_turns_mean;
        let all_raw_texts_this_wave = gen_out.all_raw_texts.clone();
        let turn1_map = gen_out.turn1_map.clone();

        // ── GAP-C1: Correlated Hallucination Detection ──────────────────────────
        let gen_out = phase!(
            phases::hallucination::run(
                gen_out,
                phases::hallucination::Input {
                    engine_input: self.input,
                    task_id,
                    retry_count: retry_count_u32,
                    system_context: &self.bootstrap.system_context,
                    system_context_with_rubric: &self.bootstrap.system_context_with_rubric,
                },
            )
            .await,
            events
        )
        .generation;

        // ── SRANI: Specification-Relative Architectural Noun Intersection ────────
        let srani_out = match phases::srani::run(
            gen_out,
            phases::srani::Input {
                engine_input: self.input,
                task_id,
                srani_tier: params.srani_tier,
                srani_last_wave_fired: params.srani_last_wave_fired,
                retry_context: params.retry_context.clone(),
            },
        )
        .await
        {
            phases::StepResult::Done(out) => out,
            _ => unreachable!("srani phase never early-exits or fatals"),
        };

        // Update SRANI carry-forward state in events so observe() sees the updated values.
        events.srani_last_wave_fired = srani_out.srani_last_wave_fired_updated;
        events.srani_tier_updated = srani_out.srani_tier_updated;
        events.srani_ema_cfi_updated = srani_out.srani_ema_cfi_updated;
        events.srani_count_updated = srani_out.srani_count_updated;
        events.srani_retry_context = srani_out.retry_context.clone();
        events
            .srani_events
            .extend(srani_out.srani_events.iter().cloned());
        events
            .researcher_grounding_events
            .extend(srani_out.researcher_grounding_events.iter().cloned());
        let gen_out = srani_out.generation;

        // Save proposal texts so the MAPE-K loop can carry them forward for the
        // next wave's leader context injection.
        {
            let texts: std::collections::HashMap<h2ai_types::identity::ExplorerId, String> =
                gen_out
                    .proposals
                    .iter()
                    .map(|p| (p.explorer_id.clone(), p.raw_output.clone()))
                    .collect();
            events.wave_proposal_texts = texts;
        }

        // ── Phase 3.5: Verification (LLM-as-Judge) ──────────────────────────────
        let verify_out = phase!(
            phases::verify::run(
                gen_out.proposals,
                phases::verify::Input {
                    engine_input: self.input,
                    task_id,
                    verification_config: params.verification_config.clone(),
                    provisioned,
                    task_eval_cache: std::sync::Arc::clone(&self.task_eval_cache),
                    turn1_map: turn1_map.clone(),
                    tau_values: tau_values.clone(),
                },
            )
            .await,
            events
        );

        let turn1_proposals_for_scoring = verify_out.turn1_proposals_for_scoring.clone();
        let conflict_rate_this_wave = verify_out.conflict_rate;

        // ── Phase 4: Auditor Gate ────────────────────────────────────────────────
        let audit_out = phase!(
            phases::audit::run(
                verify_out,
                phases::audit::Input {
                    engine_input: self.input,
                    task_id,
                    retry_count: retry_count_u32,
                    explorer_count,
                    provisioned,
                },
            )
            .await,
            events
        );

        events
            .shadow_audit_events
            .extend(audit_out.shadow_audit_events.iter().cloned());

        let proposal_set = audit_out.proposal_set;
        let synthesis_candidates = audit_out.synthesis_candidates;
        let pruned = audit_out.pruned;
        let iteration_verification_events = audit_out.iteration_verification_events;

        // ── Phase 4.5: Constraint Frontier (enrichment) ─────────────────────────
        let frontier_event = phases::frontier::run(phases::frontier::Input {
            engine_input: self.input,
            task_id,
            synthesis_candidates: &synthesis_candidates,
        });
        events.frontier_event = frontier_event.clone();

        // Per-explorer correctness for H1 ρ_actual.
        let adapter_correctness: Vec<(h2ai_types::identity::ExplorerId, bool)> =
            iteration_verification_events
                .iter()
                .map(|e| (e.explorer_id.clone(), e.passed))
                .collect();

        // ── Oracle Gate ──────────────────────────────────────────────────────────
        let oracle_gate_passed_flag: Option<bool> = phase!(
            phases::oracle::run(phases::oracle::Input {
                engine_input: self.input,
            })
            .await,
            events
        );

        // ── Coherence State ──────────────────────────────────────────────────────
        // Build from all pruned proposals accumulated so far; enrich with frontier.
        // Note: `all_pruned` across waves is held by MapeKController — here we use
        // only this wave's pruned proposals for the coherence snapshot passed to synthesis.
        let wave_coherence = {
            let base = crate::coherence::CoherenceState::from_pruned(
                &self.input.constraint_corpus,
                &pruned,
            );
            if let Some(ref fe) = frontier_event {
                base.with_contradictions(
                    &self.input.constraint_corpus,
                    &fe.explorer_ids,
                    &fe.satisfaction_matrix,
                    &fe.constraint_ids,
                )
            } else {
                base
            }
        };

        // ── TaoMultiplierEstimator Option B feed ─────────────────────────────────
        if !turn1_proposals_for_scoring.is_empty() {
            use crate::verification::VerificationPhase;
            let turn1_scores = VerificationPhase::score_proposals(
                turn1_proposals_for_scoring,
                self.input.verification_adapter,
                &params.verification_config,
                &self.input.constraint_corpus,
            )
            .await;

            let mut est = self.input.tao_estimator.write().await;
            for (t1_prop, t1_score) in &turn1_scores {
                if let Some(final_ev) = iteration_verification_events
                    .iter()
                    .find(|e| e.explorer_id == t1_prop.explorer_id && e.passed)
                {
                    est.update(*t1_score, final_ev.score);
                }
            }
        }

        // Accumulate pruned into wave events for coherence tracking by the controller.
        // (MapeKController.observe() appends these to all_pruned.)
        events.pruned_events.extend(pruned.iter().cloned());
        events.conflict_rate = conflict_rate_this_wave;

        // ── Phase 5a: Synthesis (optional) ──────────────────────────────────────
        let synthesis_out = phase!(
            phases::synthesis::run(phases::synthesis::Input {
                engine_input: self.input,
                task_id,
                assessed_quadrant: self.complexity.assessed_quadrant,
                wave_coherence: &wave_coherence,
                synthesis_candidates: &synthesis_candidates,
            })
            .await,
            events
        );

        // If synthesis produced a resolved text, take the synthesis fast-path.
        if let Some(synthesis_text) = synthesis_out.resolved_text {
            let synthesis_gain = synthesis_out.synthesis_gain;
            let synthesis_comparison_events = synthesis_out.comparison_events;

            // Build attribution for the synthesis path.
            let (mut attribution, attribution_interval) = {
                use crate::attribution::{
                    bootstrap_interval, AttributionInput, HarnessAttribution,
                };
                use crate::diagnostics::TalagrandDiagnostic;
                let iter_talagrand_state = {
                    let scores: Vec<f64> = iteration_verification_events
                        .iter()
                        .map(|e| e.score)
                        .collect();
                    TalagrandDiagnostic::from_verification_scores(&[scores])
                        .map(|d| d.calibration_state)
                };
                let attr_input = AttributionInput {
                    p_mean,
                    rho_mean,
                    n_agents: explorer_count,
                    verification_filter_ratio: {
                        let total = synthesis_candidates.len() + pruned.len();
                        if total > 0 {
                            synthesis_candidates.len() as f64 / total as f64
                        } else {
                            1.0
                        }
                    },
                    tao_turns_mean,
                    tao_per_turn_factor: self.input.tao_multiplier,
                    prediction_basis: attribution_basis,
                    talagrand_state: iter_talagrand_state,
                    eigen_calibration: self.input.calibration.eigen.clone(),
                };
                let attr = HarnessAttribution::compute(&attr_input);
                let interval = {
                    let cg_samples = &self.input.calibration.coefficients.cg_samples;
                    if cg_samples.len() >= 2 {
                        Some(bootstrap_interval(&attr_input, cg_samples, 1000))
                    } else {
                        None
                    }
                };
                (attr, interval)
            };
            attribution.synthesis_gain = synthesis_gain;

            let filter_ratio = {
                let total = synthesis_candidates.len() + pruned.len();
                if total > 0 {
                    synthesis_candidates.len() as f64 / total as f64
                } else {
                    1.0
                }
            };

            events.filter_ratio = filter_ratio;
            let q_confidence = attribution.q_confidence + synthesis_gain;
            events.quality_measurement = Some(crate::self_optimizer::QualityMeasurement {
                params: params.optimizer.clone(),
                q_confidence,
            });

            let selection_resolved = h2ai_types::events::SelectionResolvedEvent {
                task_id: task_id.clone(),
                valid_proposals: synthesis_candidates
                    .iter()
                    .map(|p| p.explorer_id.clone())
                    .collect(),
                pruned_proposals: pruned
                    .iter()
                    .map(|p| (p.explorer_id.clone(), p.reason.clone()))
                    .collect(),
                merge_strategy: provisioned.merge_strategy.clone(),
                timestamp: Utc::now(),
                merge_elapsed_secs: None,
                n_input_proposals: synthesis_candidates.len(),
            };

            // Accumulate iteration verification events (mirrors engine.rs line 1058).
            events
                .verification_events
                .extend(iteration_verification_events.iter().cloned());

            // Update bandit state.
            if let Some(ref bandit_arc) = self.input.bandit_state {
                let n_used = params.optimizer.n_agents;
                let tier3_score = Some(attribution.q_confidence.clamp(0.0, 1.0));
                let mut bandit = bandit_arc.write().await;
                bandit.update(n_used, None, tier3_score);
            }

            self.input.store.mark_resolved(task_id);

            let merge_out = crate::mape_k::MergeOutput {
                task_id: task_id.clone(),
                resolved_output: synthesis_text,
                selection_resolved: true,
                selection_resolved_event: selection_resolved,
                attribution,
                attribution_interval,
                talagrand: None,
                suggested_next_params: None,
                waste_ratio: filter_ratio,
                applied_optimizations: vec![],
                epistemic_yield: None,
                frontier_event: frontier_event.clone(),
                adapter_correctness,
                coherence_state: wave_coherence,
                comparison_events: synthesis_comparison_events,
                oracle_gate_passed: oracle_gate_passed_flag,
                tau_values,
                iteration_verification_events,
            };

            return PipelineWaveResult {
                outcome: PipelineOutcome::Resolved(Box::new(merge_out)),
                events,
            };
        }

        // Synthesis did not resolve — proceed to Phase 5: Merge.
        let synthesis_gain = synthesis_out.synthesis_gain;
        let synthesis_comparison_events = synthesis_out.comparison_events;

        let surviving_texts_for_yield: Vec<String> = synthesis_candidates
            .iter()
            .map(|p| p.raw_output.clone())
            .collect();

        let total_evaluated = synthesis_candidates.len() + pruned.len();
        let filter_ratio = if total_evaluated > 0 {
            synthesis_candidates.len() as f64 / total_evaluated as f64
        } else {
            1.0
        };
        events.filter_ratio = filter_ratio;

        // ── Phase 5: Merge ───────────────────────────────────────────────────────
        let (merge_step, tau_expansion_hint) = phases::merge::run(
            proposal_set,
            pruned,
            synthesis_gain,
            synthesis_comparison_events,
            phases::merge::Input {
                engine_input: self.input,
                task_id,
                retry_count: retry_count_u32,
                explorer_count,
                filter_ratio,
                p_mean,
                rho_mean,
                tao_turns_mean,
                attribution_basis,
                tau_values,
                all_raw_texts_this_wave,
                surviving_texts: surviving_texts_for_yield,
                iteration_verification_events: &iteration_verification_events,
                frontier_event: &frontier_event,
                adapter_correctness,
                oracle_gate_passed: oracle_gate_passed_flag,
                wave_coherence: &wave_coherence,
                quality_history: &[],
                n_max_ceiling: self.complexity.n_max_ceiling,
                cg_mean: self.complexity.cg_mean,
                current_params: &params.optimizer,
                verification_config: params.verification_config.clone(),
                assessed_quadrant: self.complexity.assessed_quadrant,
                all_pruned: &[],
                synthesis_candidates_len: synthesis_candidates.len(),
                provisioned_merge_strategy: provisioned.merge_strategy.clone(),
            },
        )
        .await;

        if let Some(tau) = tau_expansion_hint {
            events.talagrand_feedback = Some(crate::mape_k::TalagrandFeedback {
                tau_spread_next: tau,
            });
        }

        // Always extend verification events after merge phase.
        events
            .verification_events
            .extend(iteration_verification_events.iter().cloned());

        match merge_step {
            phases::StepResult::Done(merge_out) => {
                // Quality measurement for the controller's quality_history.
                events.quality_measurement = Some(crate::self_optimizer::QualityMeasurement {
                    params: params.optimizer.clone(),
                    q_confidence: merge_out.attribution.q_confidence,
                });
                self.input.store.mark_resolved(task_id);
                if let Some(ref bandit_arc) = self.input.bandit_state {
                    let n_used = params.optimizer.n_agents;
                    let tier3_score = Some(merge_out.attribution.q_confidence.clamp(0.0, 1.0));
                    let mut bandit = bandit_arc.write().await;
                    bandit.update(n_used, None, tier3_score);
                    if let Some(ref suggested) = merge_out.suggested_next_params {
                        bandit.apply_optimizer_hint(n_used, suggested.n_agents);
                    }
                }
                PipelineWaveResult {
                    outcome: PipelineOutcome::Resolved(Box::new(merge_out)),
                    events,
                }
            }
            phases::StepResult::EarlyExit(r) => PipelineWaveResult {
                outcome: PipelineOutcome::EarlyExit(r),
                events,
            },
            phases::StepResult::Fatal(e) => PipelineWaveResult {
                outcome: PipelineOutcome::Fatal(e),
                events,
            },
        }
    }
}
