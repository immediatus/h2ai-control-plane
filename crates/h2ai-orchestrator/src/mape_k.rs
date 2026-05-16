use crate::diagnostics::TalagrandDiagnostic;
use crate::engine::{EngineError, EngineOutput};
use crate::phases::ExitReason;
use crate::self_optimizer::{OptimizerParams, QualityMeasurement, SelfOptimizer, SuggestInput};
use h2ai_autonomic::retry::{RetryAction, RetryPolicy};
use h2ai_types::config::{TaoConfig, TopologyKind, VerificationConfig};
use h2ai_types::events::{
    ConstraintFrontierEvent, CorrelatedEnsembleWarning, CorrelatedFabricationEvent,
    ProposalFailedEvent, ResearcherGroundingEvent, ShadowAuditorResultEvent,
    TopologyProvisionedEvent, VerificationScoredEvent, ZeroSurvivalEvent,
};
use h2ai_types::identity::{ExplorerId, TaskId};

/// Immutable snapshot of MAPE-K params for one wave.
#[derive(Clone, Debug)]
pub struct PipelineParams {
    pub optimizer: OptimizerParams,
    pub force_topology: Option<TopologyKind>,
    pub tau_reduction_factor: f64,
    pub tau_spread_factor: f64,
    pub adapter_rotation_offset: usize,
    pub retry_context: Option<String>,
    pub tao_config: TaoConfig,
    pub verification_config: VerificationConfig,
    pub srani_ema_cfi: f64,
    pub srani_count: u64,
    pub srani_tier: usize,
    pub srani_last_wave_fired: bool,
    pub pending_tombstone: Option<String>,
}

/// Talagrand feedback stored in WaveEvents.
#[derive(Clone, Debug)]
pub struct TalagrandFeedback {
    pub tau_spread_next: f64,
}

/// TaoEstimator update stored in WaveEvents.
#[derive(Clone, Debug)]
pub struct TaoEstimatorUpdate {
    pub t1_score: f64,
    pub final_score: f64,
}

/// All events produced in one pipeline wave, collected for observe().
#[derive(Clone, Debug)]
pub struct WaveEvents {
    pub verification_events: Vec<VerificationScoredEvent>,
    pub failed_proposals: Vec<ProposalFailedEvent>,
    pub shadow_audit_events: Vec<ShadowAuditorResultEvent>,
    pub correlated_warnings: Vec<CorrelatedEnsembleWarning>,
    pub srani_events: Vec<CorrelatedFabricationEvent>,
    pub researcher_grounding_events: Vec<ResearcherGroundingEvent>,
    pub quality_measurement: Option<crate::self_optimizer::QualityMeasurement>,
    pub talagrand_feedback: Option<TalagrandFeedback>,
    pub tao_estimator_update: Option<TaoEstimatorUpdate>,
    pub topology_retry_event: Option<TopologyProvisionedEvent>,
    pub frontier_event: Option<ConstraintFrontierEvent>,
    /// Updated `srani_last_wave_fired` flag from the SRANI phase output.
    /// Initialized to the pre-wave value; overwritten when SRANI phase runs.
    pub srani_last_wave_fired: bool,
    /// Updated srani_tier from the SRANI phase output.
    /// Initialized to the pre-wave value; overwritten when SRANI phase runs.
    pub srani_tier_updated: usize,
    /// Updated srani_ema_cfi from the SRANI phase output.
    /// Initialized to the pre-wave value; overwritten when SRANI phase runs.
    pub srani_ema_cfi_updated: f64,
    /// Updated srani_count from the SRANI phase output (as usize, matching engine.rs).
    /// Initialized to the pre-wave value; overwritten when SRANI phase runs.
    pub srani_count_updated: usize,
    /// Updated retry_context from the SRANI phase (may have been extended with SRANI hint).
    pub srani_retry_context: Option<String>,
    /// Verification filter ratio from this wave's merge phase (surviving / total evaluated).
    /// 1.0 when no merge ran (early-exit before merge). Used by the coordinator to call decide().
    pub filter_ratio: f64,
    /// Pruned branch events from this wave's audit phase.
    /// Accumulated by `observe()` into `all_pruned` so `RetryPolicy::decide` can
    /// collect `remediation_hint` strings for `RetryWithHints`.
    pub pruned_events: Vec<h2ai_types::events::BranchPrunedEvent>,
    /// Mean pairwise constraint-conflict rate across all proposals in this wave.
    /// `None` when fewer than 2 proposals were generated or corpus is empty.
    pub conflict_rate: Option<f64>,
}

impl Default for WaveEvents {
    fn default() -> Self {
        Self {
            verification_events: Vec::new(),
            failed_proposals: Vec::new(),
            shadow_audit_events: Vec::new(),
            correlated_warnings: Vec::new(),
            srani_events: Vec::new(),
            researcher_grounding_events: Vec::new(),
            quality_measurement: None,
            talagrand_feedback: None,
            tao_estimator_update: None,
            topology_retry_event: None,
            frontier_event: None,
            srani_last_wave_fired: false,
            srani_tier_updated: 0,
            srani_ema_cfi_updated: 0.0,
            srani_count_updated: 0,
            srani_retry_context: None,
            filter_ratio: 1.0,
            pruned_events: Vec::new(),
            conflict_rate: None,
        }
    }
}

/// Successful merge result — passed from pipeline to controller via PipelineOutcome::Resolved.
pub struct MergeOutput {
    pub task_id: TaskId,
    pub resolved_output: String,
    /// `true` = merge resolved successfully (always true when `Done` is returned).
    pub selection_resolved: bool,
    /// Full `SelectionResolvedEvent` produced by `MergeEngine::resolve`.
    /// Carried here so engine.rs can publish it without reconstructing timing data.
    pub selection_resolved_event: h2ai_types::events::SelectionResolvedEvent,
    pub attribution: crate::attribution::HarnessAttribution,
    pub attribution_interval: Option<crate::attribution::AttributionInterval>,
    pub talagrand: Option<crate::diagnostics::TalagrandDiagnostic>,
    pub suggested_next_params: Option<OptimizerParams>,
    pub waste_ratio: f64,
    pub applied_optimizations: Vec<h2ai_types::events::AppliedOptimization>,
    pub epistemic_yield: Option<f64>,
    pub frontier_event: Option<ConstraintFrontierEvent>,
    pub adapter_correctness: Vec<(ExplorerId, bool)>,
    pub coherence_state: crate::coherence::CoherenceState,
    pub comparison_events: Vec<h2ai_types::events::VerifierComparisonEvent>,
    pub oracle_gate_passed: Option<bool>,
    /// τ values from this wave's generation phase; pushed into `tau_values_tried` by engine.rs.
    pub tau_values: Vec<f64>,
    /// Per-iteration verification events; appended to `all_verification_events` by engine.rs.
    pub iteration_verification_events: Vec<h2ai_types::events::VerificationScoredEvent>,
}

/// Pipeline outcome after one wave.
pub enum PipelineOutcome {
    Resolved(Box<MergeOutput>),
    EarlyExit(ExitReason),
    Fatal(EngineError),
}

/// What the pipeline returns after one wave.
pub struct PipelineWaveResult {
    pub outcome: PipelineOutcome,
    pub events: WaveEvents,
}

/// Controller decision after observing a wave.
pub enum MapeKDecision {
    Return(Box<EngineOutput>),
    Retry,
    Fail(EngineError),
}

/// MAPE-K controller — owns all retry state. Full impl added in Task 9.
#[allow(dead_code)] // bandit_state / tao_estimator / tao_multiplier reserved for future pipeline use
pub struct MapeKController {
    // Optimizer
    pub(crate) current_params: OptimizerParams,
    pub(crate) quality_history: Vec<QualityMeasurement>,
    pub(crate) n_max_ceiling: u32,
    pub(crate) cg_mean: f64,

    // Topology
    pub(crate) force_topology: Option<TopologyKind>,
    pub(crate) tried_topologies: Vec<TopologyKind>,

    // τ feedback
    pub(crate) tau_reduction_factor: f64,
    pub(crate) tau_spread_factor: f64,
    pub(crate) tau_values_tried: Vec<Vec<f64>>,

    // Retry routing
    pub(crate) retry_context: Option<String>,
    pub(crate) adapter_rotation_offset: usize,
    pub(crate) mode_collapse_count: usize,
    pub(crate) last_multiplication_failure:
        Option<h2ai_types::sizing::MultiplicationConditionFailure>,
    pub(crate) pending_tombstone: Option<String>,
    pub(crate) system_context_with_rubric: String,
    pub(crate) max_retries: usize,

    // SRANI EMA
    pub(crate) srani_ema: f64,
    pub(crate) srani_count: u64,
    pub(crate) srani_tier: usize,
    pub(crate) srani_last_wave_fired: bool,

    // Per-wave config overrides
    pub(crate) tao_config: TaoConfig,
    pub(crate) verification_config: VerificationConfig,

    // Cross-wave aggregation
    pub(crate) all_verification_events: Vec<VerificationScoredEvent>,
    pub(crate) all_failed_proposals: Vec<ProposalFailedEvent>,
    pub(crate) all_shadow_audit_events: Vec<ShadowAuditorResultEvent>,
    pub(crate) all_correlated_warnings: Vec<CorrelatedEnsembleWarning>,
    pub(crate) all_srani_events: Vec<CorrelatedFabricationEvent>,
    pub(crate) all_researcher_grounding_events: Vec<ResearcherGroundingEvent>,
    pub(crate) all_pruned: Vec<h2ai_types::events::BranchPrunedEvent>,
    /// Pruned events from the most recent wave only — used for tombstone synthesis
    /// so the LLM receives violations from the immediately preceding wave rather
    /// than the full historical accumulator (which causes attention dilution).
    pub(crate) last_wave_pruned: Vec<h2ai_types::events::BranchPrunedEvent>,
    pub(crate) topology_retry_events: Vec<TopologyProvisionedEvent>,

    // Immutable fields
    pub(crate) task_id: TaskId,
    pub(crate) assessed_quadrant: h2ai_types::sizing::TaskQuadrant,
    pub(crate) complexity_event: h2ai_types::events::TaskComplexityAssessedEvent,
    pub(crate) diversity_degraded_event: Option<h2ai_types::events::DiversityGuardDegradedEvent>,
    pub(crate) bandit_state:
        Option<std::sync::Arc<tokio::sync::RwLock<crate::bandit::BanditState>>>,
    pub(crate) tao_estimator:
        std::sync::Arc<tokio::sync::RwLock<crate::tao_loop::TaoMultiplierEstimator>>,
    pub(crate) tao_multiplier: f64,
    pub(crate) calibration_ensemble: Option<h2ai_types::sizing::EnsembleCalibration>,
    pub(crate) cfg_ref: std::sync::Arc<h2ai_config::H2AIConfig>,
    pub(crate) task_deadline: Option<std::time::Instant>,
}

impl MapeKController {
    // ── Constructor ────────────────────────────────────────────────────────────

    /// Build the controller from the engine input and pre-loop phase outputs.
    ///
    /// Reads the bandit asynchronously so `new` is `async`.
    pub async fn new(
        input: &crate::engine::EngineInput<'_>,
        bootstrap_out: &crate::phases::bootstrap::Output,
        complexity_out: &crate::phases::complexity::Output,
    ) -> Self {
        let task_id = input.task_id.clone();
        let assessed_quadrant = complexity_out.assessed_quadrant;
        let complexity_event = complexity_out.complexity_event.clone();
        let cg_mean = complexity_out.cg_mean;
        let n_max_ceiling = complexity_out.n_max_ceiling;

        let manifest_count = input.manifest.explorers.count as u32;
        let n_optimal_hint = input
            .calibration
            .ensemble
            .as_ref()
            .map(|ec| (ec.n_optimal as u32).min(manifest_count))
            .unwrap_or(manifest_count);

        let bandit_n = if let Some(ref bandit_arc) = input.bandit_state {
            let bandit = bandit_arc.read().await;
            Some(bandit.select(input.cfg))
        } else {
            None
        };
        let initial_n_agents = bandit_n
            .unwrap_or(n_optimal_hint)
            .max(1)
            .min(n_max_ceiling.max(1));

        let current_params = OptimizerParams {
            n_agents: initial_n_agents,
            max_turns: input.tao_config.max_turns as u32,
            verify_threshold: input.verification_config.threshold,
        };

        let srani_ema = input.srani_ema_cfi;
        let srani_count = input.srani_count as u64;

        let task_deadline = input
            .cfg
            .task_deadline_secs
            .map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));

        Self {
            current_params,
            quality_history: Vec::new(),
            n_max_ceiling,
            cg_mean,
            force_topology: None,
            tried_topologies: Vec::new(),
            tau_reduction_factor: 1.0,
            tau_spread_factor: 1.0,
            tau_values_tried: Vec::new(),
            retry_context: None,
            adapter_rotation_offset: 0,
            mode_collapse_count: 0,
            last_multiplication_failure: None,
            pending_tombstone: None,
            system_context_with_rubric: bootstrap_out.system_context_with_rubric.clone(),
            max_retries: input.cfg.max_autonomic_retries as usize,
            srani_ema,
            srani_count,
            srani_tier: 0,
            srani_last_wave_fired: false,
            tao_config: input.tao_config.clone(),
            verification_config: input.verification_config.clone(),
            all_verification_events: Vec::new(),
            all_failed_proposals: Vec::new(),
            all_shadow_audit_events: Vec::new(),
            all_correlated_warnings: Vec::new(),
            all_srani_events: Vec::new(),
            all_researcher_grounding_events: Vec::new(),
            all_pruned: Vec::new(),
            last_wave_pruned: Vec::new(),
            topology_retry_events: Vec::new(),
            task_id,
            assessed_quadrant,
            complexity_event,
            diversity_degraded_event: None,
            bandit_state: input.bandit_state.clone(),
            tao_estimator: std::sync::Arc::clone(&input.tao_estimator),
            tao_multiplier: input.tao_multiplier,
            calibration_ensemble: input.calibration.ensemble.clone(),
            cfg_ref: std::sync::Arc::new(input.cfg.clone()),
            task_deadline,
        }
    }

    // ── Snapshot ───────────────────────────────────────────────────────────────

    /// Return an immutable snapshot of the current MAPE-K parameters for one wave.
    pub fn params(&self) -> PipelineParams {
        PipelineParams {
            optimizer: self.current_params.clone(),
            force_topology: self.force_topology.clone(),
            tau_reduction_factor: self.tau_reduction_factor,
            tau_spread_factor: self.tau_spread_factor,
            adapter_rotation_offset: self.adapter_rotation_offset,
            retry_context: self.retry_context.clone(),
            tao_config: self.tao_config.clone(),
            verification_config: self.verification_config.clone(),
            srani_ema_cfi: self.srani_ema,
            srani_count: self.srani_count,
            srani_tier: self.srani_tier,
            srani_last_wave_fired: self.srani_last_wave_fired,
            pending_tombstone: self.pending_tombstone.clone(),
        }
    }

    // ── Observe ────────────────────────────────────────────────────────────────

    /// Aggregate events from a completed wave into the cross-wave accumulators.
    pub fn observe(&mut self, wave: &PipelineWaveResult) {
        let e = &wave.events;
        self.all_verification_events
            .extend(e.verification_events.iter().cloned());
        self.all_failed_proposals
            .extend(e.failed_proposals.iter().cloned());
        self.all_shadow_audit_events
            .extend(e.shadow_audit_events.iter().cloned());
        self.all_correlated_warnings
            .extend(e.correlated_warnings.iter().cloned());
        self.all_srani_events.extend(e.srani_events.iter().cloned());
        self.all_researcher_grounding_events
            .extend(e.researcher_grounding_events.iter().cloned());
        if let Some(ref qm) = e.quality_measurement {
            self.quality_history.push(qm.clone());
        }
        if let Some(ref tf) = e.talagrand_feedback {
            self.tau_spread_factor = tf.tau_spread_next;
        }
        if let Some(ref retry_ev) = e.topology_retry_event {
            self.topology_retry_events.push(retry_ev.clone());
        }
        // SRANI state updates from the wave.
        self.srani_last_wave_fired = e.srani_last_wave_fired;
        self.srani_tier = e.srani_tier_updated;
        self.srani_ema = e.srani_ema_cfi_updated;
        self.srani_count = e.srani_count_updated as u64;
        if let Some(ref rc) = e.srani_retry_context {
            self.retry_context = Some(rc.clone());
        }
        // Snapshot current wave's pruned events before extending the cross-wave accumulator.
        self.last_wave_pruned = e.pruned_events.clone();
        // Accumulate pruned events so RetryPolicy::decide can extract remediation hints.
        self.all_pruned.extend(e.pruned_events.iter().cloned());
    }

    // ── Decide ─────────────────────────────────────────────────────────────────

    /// Evaluate the wave outcome and return the MAPE-K decision.
    ///
    /// `retry_count` is the current loop iteration (0-based) used when building
    /// the constraint-feedback hint string.  `filter_ratio` is the wave's
    /// verification pass rate; it is forwarded to `run_apply_optimizer`.
    pub fn decide(
        &mut self,
        outcome: PipelineOutcome,
        retry_count: u32,
        filter_ratio: f64,
    ) -> MapeKDecision {
        match outcome {
            PipelineOutcome::Resolved(merge_out) => {
                let merge_out = *merge_out;
                // Push tau_values from the successful wave.
                self.tau_values_tried.push(merge_out.tau_values.clone());
                // Push quality measurement from the merge result.
                self.quality_history.push(QualityMeasurement {
                    params: self.current_params.clone(),
                    q_confidence: merge_out.attribution.q_confidence,
                });
                // Extend cross-wave comparison events.
                MapeKDecision::Return(Box::new(self.finalize(merge_out)))
            }

            PipelineOutcome::Fatal(e) => MapeKDecision::Fail(e),

            PipelineOutcome::EarlyExit(reason) => {
                self.handle_exit_reason(reason, retry_count, filter_ratio)
            }
        }
    }

    /// Internal: map an `ExitReason` to a `MapeKDecision`.
    fn handle_exit_reason(
        &mut self,
        reason: ExitReason,
        retry_count: u32,
        filter_ratio: f64,
    ) -> MapeKDecision {
        match reason {
            ExitReason::MultiplicationFailed {
                msg: _,
                tau_values,
                failure,
            } => {
                tracing::warn!(
                    target: "h2ai.mape_k",
                    failure = ?failure,
                    "multiplication condition failed"
                );
                self.last_multiplication_failure = Some(failure);
                self.tau_values_tried.push(tau_values);

                let zero_event = ZeroSurvivalEvent {
                    task_id: self.task_id.clone(),
                    retry_count,
                    timestamp: chrono::Utc::now(),
                    n_eff_cosine_actual: None,
                    failure_mode: None,
                };
                self.apply_retry_action(
                    RetryPolicy::decide(
                        &zero_event,
                        &self.tried_topologies.clone(),
                        self.all_pruned.clone(),
                        self.tau_values_tried.clone(),
                        self.last_multiplication_failure.clone(),
                    ),
                    retry_count,
                    1.0, // filter_ratio not applicable here
                )
            }

            ExitReason::DiversityFailed { n_eff, tau_values } => {
                self.last_multiplication_failure = Some(
                    h2ai_types::sizing::MultiplicationConditionFailure::InsufficientPoolDiversity {
                        n_eff,
                        threshold: self.cfg_ref.safety.diversity_threshold,
                    },
                );
                self.tau_values_tried.push(tau_values);
                let zero_event = ZeroSurvivalEvent {
                    task_id: self.task_id.clone(),
                    retry_count,
                    timestamp: chrono::Utc::now(),
                    n_eff_cosine_actual: Some(n_eff),
                    failure_mode: Some(h2ai_types::events::FailureMode::ModeCollapse),
                };
                self.apply_retry_action(
                    RetryPolicy::decide(
                        &zero_event,
                        &self.tried_topologies.clone(),
                        self.all_pruned.clone(),
                        self.tau_values_tried.clone(),
                        self.last_multiplication_failure.clone(),
                    ),
                    retry_count,
                    1.0,
                )
            }

            ExitReason::ZeroSurvival {
                failure_mode: detected_failure_mode,
                coherence: _,
                n_eff_cosine: zs_n_eff_cosine,
                filter_ratio: _wave_filter_ratio,
                tau_values: zs_tau_values,
            } => {
                self.tau_values_tried.push(zs_tau_values);

                // Apply FailureMode-specific state mutations before RetryPolicy selection.
                match &detected_failure_mode {
                    Some(h2ai_types::events::FailureMode::ModeCollapse) => {
                        let pool_len = 1usize; // conservative fallback; engine knows the pool length
                        let _ = pool_len;
                        self.mode_collapse_count += 1;
                        self.pending_tombstone = None;
                    }
                    Some(h2ai_types::events::FailureMode::ConstrainedExploration) => {
                        // Use only the immediately preceding wave's violations — not the
                        // full cross-wave accumulator — to avoid feeding the LLM constraint
                        // errors from multiple waves ago that it has already corrected.
                        let wave_violations: Vec<h2ai_types::events::ConstraintViolation> = self
                            .last_wave_pruned
                            .iter()
                            .flat_map(|p| p.violated_constraints.iter().cloned())
                            .collect();
                        self.pending_tombstone =
                            h2ai_autonomic::epistemic::synthesize_tombstone(&wave_violations);
                    }
                    Some(h2ai_types::events::FailureMode::CorrelatedHallucination { .. }) => {
                        // Handled by HallucinationDetected before Phase 3.5.
                    }
                    None => {}
                }

                let zero_event = ZeroSurvivalEvent {
                    task_id: self.task_id.clone(),
                    retry_count,
                    timestamp: chrono::Utc::now(),
                    n_eff_cosine_actual: zs_n_eff_cosine,
                    failure_mode: detected_failure_mode,
                };

                self.apply_retry_action(
                    RetryPolicy::decide(
                        &zero_event,
                        &self.tried_topologies.clone(),
                        self.all_pruned.clone(),
                        self.tau_values_tried.clone(),
                        self.last_multiplication_failure.clone(),
                    ),
                    retry_count,
                    filter_ratio,
                )
            }

            ExitReason::HallucinationDetected {
                retry_context_hint,
                tau_values,
                warning,
                researcher_grounding_events,
            } => {
                self.all_correlated_warnings.push(warning);
                self.all_researcher_grounding_events
                    .extend(researcher_grounding_events);
                self.retry_context = Some(retry_context_hint);
                self.tau_values_tried.push(tau_values);
                self.run_apply_optimizer(1.0);
                MapeKDecision::Retry
            }

            ExitReason::OracleBlocked => MapeKDecision::Fail(EngineError::MaxRetriesExhausted {
                partial_verification_events: self.all_verification_events.clone(),
            }),
        }
    }

    /// Map a `RetryAction` to a `MapeKDecision`, mutating controller state.
    fn apply_retry_action(
        &mut self,
        action: RetryAction,
        retry_count: u32,
        filter_ratio: f64,
    ) -> MapeKDecision {
        match action {
            RetryAction::Retry(next_topology) => {
                self.force_topology = Some(next_topology);
                self.run_apply_optimizer(filter_ratio);
                MapeKDecision::Retry
            }
            RetryAction::RetryWithTauReduction {
                topology,
                tau_factor,
            } => {
                self.force_topology = Some(topology);
                self.tau_reduction_factor *= tau_factor;
                self.run_apply_optimizer(filter_ratio);
                MapeKDecision::Retry
            }
            RetryAction::RetryWithHints { topology, hints } => {
                self.force_topology = Some(topology);
                if !hints.is_empty() {
                    let attempts_remaining = (self.max_retries as u32).saturating_sub(retry_count);
                    let hint_lines = hints
                        .iter()
                        .map(|h| format!("• {h}"))
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    self.retry_context = Some(format!(
                        "{ctx}\n\n--- CONSTRAINT FEEDBACK (iteration {retry_count}) ---\n\
                        The following constraints were violated. Fix ALL of these in your next response:\n\n\
                        {hint_lines}\n\n\
                        {attempts_remaining} retry attempt(s) remaining.\n\
                        ---",
                        ctx = self.system_context_with_rubric
                    ));
                }
                self.run_apply_optimizer(filter_ratio);
                MapeKDecision::Retry
            }
            RetryAction::Fail(reason) => {
                tracing::warn!(
                    target: "h2ai.mape_k",
                    task_id = %self.task_id,
                    retry_count,
                    reason = ?reason,
                    "retry policy decided Fail — giving up"
                );
                MapeKDecision::Fail(EngineError::MaxRetriesExhausted {
                    partial_verification_events: self.all_verification_events.clone(),
                })
            }
        }
    }

    // ── Finalize ───────────────────────────────────────────────────────────────

    /// Assemble the final `EngineOutput` from a successful merge result and the
    /// cross-wave accumulators held in the controller.
    pub fn finalize(&self, merge_out: MergeOutput) -> EngineOutput {
        let run_scores: Vec<f64> = self
            .all_verification_events
            .iter()
            .map(|e| e.score)
            .collect();
        let talagrand = TalagrandDiagnostic::from_verification_scores(&[run_scores]);
        EngineOutput {
            task_id: merge_out.task_id,
            resolved_output: merge_out.resolved_output,
            selection_resolved: merge_out.selection_resolved_event,
            attribution: merge_out.attribution,
            attribution_interval: merge_out.attribution_interval,
            verification_events: self.all_verification_events.clone(),
            failed_proposals: self.all_failed_proposals.clone(),
            talagrand,
            suggested_next_params: merge_out.suggested_next_params,
            waste_ratio: merge_out.waste_ratio,
            applied_optimizations: merge_out.applied_optimizations,
            topology_retry_events: self.topology_retry_events.clone(),
            mode_collapse_count: self.mode_collapse_count,
            epistemic_yield: merge_out.epistemic_yield,
            task_quadrant: Some(self.assessed_quadrant),
            complexity_event: self.complexity_event.clone(),
            frontier_event: merge_out.frontier_event,
            adapter_correctness: merge_out.adapter_correctness,
            coherence_state: merge_out.coherence_state,
            comparison_events: merge_out.comparison_events,
            shadow_audit_events: self.all_shadow_audit_events.clone(),
            correlated_warnings: self.all_correlated_warnings.clone(),
            researcher_grounding_events: self.all_researcher_grounding_events.clone(),
            diversity_degraded_event: self.diversity_degraded_event.clone(),
            srani_events: self.all_srani_events.clone(),
            srani_ema_cfi_updated: self.srani_ema,
            srani_count_updated: self.srani_count as usize,
            oracle_gate_passed: merge_out.oracle_gate_passed,
        }
    }

    // ── Self-Optimizer ─────────────────────────────────────────────────────────

    /// Update `current_params`, `tao_config`, and `verification_config` via `SelfOptimizer`.
    pub fn run_apply_optimizer(&mut self, filter_ratio: f64) {
        let (p_mean, rho_mean) = match &self.calibration_ensemble {
            Some(ec) => (ec.p_mean, ec.rho_mean),
            None => (
                0.5 + self.cg_mean / 2.0,
                (1.0 - self.cg_mean).clamp(0.0, 1.0),
            ),
        };
        let n_optimal = self
            .calibration_ensemble
            .as_ref()
            .map(|ec| ec.n_optimal as u32);
        let suggested = SelfOptimizer::suggest(SuggestInput {
            current: &self.current_params,
            history: &self.quality_history,
            n_max_ceiling: self.n_max_ceiling,
            n_optimal,
            p_mean,
            rho_mean,
            filter_ratio,
            cfg: &self.cfg_ref,
        });
        if suggested.max_turns != self.current_params.max_turns {
            self.tao_config.max_turns = suggested.max_turns as u8;
        }
        if (suggested.verify_threshold - self.current_params.verify_threshold).abs() > 1e-9 {
            self.verification_config.threshold = suggested.verify_threshold;
        }
        self.current_params = suggested;
    }

    // ── Coordinator helpers ────────────────────────────────────────────────────

    /// Returns the task deadline for the coordinator's deadline check.
    pub fn deadline(&self) -> Option<std::time::Instant> {
        self.task_deadline
    }

    /// Returns all verification events collected — used for MaxRetriesExhausted error.
    pub fn take_verification_events(&self) -> Vec<h2ai_types::events::VerificationScoredEvent> {
        self.all_verification_events.clone()
    }

    // ── Test helpers ───────────────────────────────────────────────────────────

    #[cfg(test)]
    pub fn new_for_test(cfg: h2ai_config::H2AIConfig) -> Self {
        use crate::tao_loop::TaoMultiplierEstimator;
        use h2ai_types::events::TaskComplexityAssessedEvent;
        use h2ai_types::identity::TaskId;
        use h2ai_types::sizing::{ProbeSkipReason, TaskQuadrant};

        let task_id = TaskId::new();
        let complexity_event = TaskComplexityAssessedEvent {
            task_id: task_id.clone(),
            tcc_structural: 0.0,
            tcc_empirical: None,
            tcc_effective: 0.0,
            n_eff_pool: None,
            task_quadrant: TaskQuadrant::Precision,
            probe_skipped: true,
            probe_skip_reason: ProbeSkipReason::None,
            heavy_fraction: 0.0,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: 0,
            timestamp: chrono::Utc::now(),
        };
        let tao_config = TaoConfig::default();
        let verification_config = VerificationConfig::default();
        let max_retries = cfg.max_autonomic_retries as usize;

        Self {
            current_params: OptimizerParams {
                n_agents: 3,
                max_turns: 4,
                verify_threshold: verification_config.threshold,
            },
            quality_history: Vec::new(),
            n_max_ceiling: 9,
            cg_mean: 0.5,
            force_topology: None,
            tried_topologies: Vec::new(),
            tau_reduction_factor: 1.0,
            tau_spread_factor: 1.0,
            tau_values_tried: Vec::new(),
            retry_context: None,
            adapter_rotation_offset: 0,
            mode_collapse_count: 0,
            last_multiplication_failure: None,
            pending_tombstone: None,
            system_context_with_rubric: String::new(),
            max_retries,
            srani_ema: 0.0,
            srani_count: 0,
            srani_tier: 0,
            srani_last_wave_fired: false,
            tao_config,
            verification_config,
            all_verification_events: Vec::new(),
            all_failed_proposals: Vec::new(),
            all_shadow_audit_events: Vec::new(),
            all_correlated_warnings: Vec::new(),
            all_srani_events: Vec::new(),
            all_researcher_grounding_events: Vec::new(),
            all_pruned: Vec::new(),
            last_wave_pruned: Vec::new(),
            topology_retry_events: Vec::new(),
            task_id,
            assessed_quadrant: TaskQuadrant::Precision,
            complexity_event,
            diversity_degraded_event: None,
            bandit_state: None,
            tao_estimator: std::sync::Arc::new(tokio::sync::RwLock::new(
                TaoMultiplierEstimator::new_with_alpha(0.1),
            )),
            tao_multiplier: 1.0,
            calibration_ensemble: None,
            cfg_ref: std::sync::Arc::new(cfg),
            task_deadline: None,
        }
    }

    #[cfg(test)]
    pub fn verification_event_count(&self) -> usize {
        self.all_verification_events.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phases::ExitReason;
    use h2ai_config::H2AIConfig;

    fn default_controller() -> MapeKController {
        MapeKController::new_for_test(H2AIConfig::default())
    }

    fn empty_wave(outcome: PipelineOutcome) -> PipelineWaveResult {
        PipelineWaveResult {
            outcome,
            events: WaveEvents::default(),
        }
    }

    #[test]
    fn decide_retry_or_fail_on_multiplication_failed() {
        use h2ai_types::sizing::MultiplicationConditionFailure;

        let mut ctrl = default_controller();
        let wave = empty_wave(PipelineOutcome::EarlyExit(
            ExitReason::MultiplicationFailed {
                msg: "test".into(),
                tau_values: vec![0.3, 0.5, 0.7],
                failure: MultiplicationConditionFailure::InsufficientCompetence {
                    actual: 0.1,
                    required: 0.6,
                },
            },
        ));
        ctrl.observe(&wave);
        let decision = ctrl.decide(
            PipelineOutcome::EarlyExit(ExitReason::MultiplicationFailed {
                msg: "test".into(),
                tau_values: vec![0.3, 0.5, 0.7],
                failure: MultiplicationConditionFailure::InsufficientCompetence {
                    actual: 0.1,
                    required: 0.6,
                },
            }),
            0,
            1.0,
        );
        // Either Retry (policy chose a new topology) or Fail (retries exhausted)
        assert!(matches!(
            decision,
            MapeKDecision::Retry | MapeKDecision::Fail(_)
        ));
    }

    #[test]
    fn observe_aggregates_verification_events_across_waves() {
        use chrono::Utc;
        use h2ai_types::events::VerificationScoredEvent;
        use h2ai_types::identity::{ExplorerId, TaskId};

        let mut ctrl = default_controller();

        let make_event = || VerificationScoredEvent {
            task_id: TaskId::new(),
            explorer_id: ExplorerId::new(),
            score: 0.8,
            reason: "ok".into(),
            passed: true,
            cache_hit: false,
            timestamp: Utc::now(),
        };

        let mut wave1_events = WaveEvents::default();
        wave1_events.verification_events.push(make_event());
        ctrl.observe(&PipelineWaveResult {
            outcome: PipelineOutcome::EarlyExit(ExitReason::OracleBlocked),
            events: wave1_events,
        });

        let mut wave2_events = WaveEvents::default();
        wave2_events.verification_events.push(make_event());
        ctrl.observe(&PipelineWaveResult {
            outcome: PipelineOutcome::EarlyExit(ExitReason::OracleBlocked),
            events: wave2_events,
        });

        assert_eq!(ctrl.verification_event_count(), 2);
    }

    #[test]
    fn wave_events_default_has_none_conflict_rate() {
        let events = WaveEvents::default();
        assert!(events.conflict_rate.is_none());
    }

    #[test]
    fn decide_fail_on_oracle_blocked() {
        let mut ctrl = default_controller();
        let wave = PipelineWaveResult {
            outcome: PipelineOutcome::EarlyExit(ExitReason::OracleBlocked),
            events: WaveEvents::default(),
        };
        ctrl.observe(&wave);
        let decision = ctrl.decide(
            PipelineOutcome::EarlyExit(ExitReason::OracleBlocked),
            0,
            1.0,
        );
        assert!(matches!(decision, MapeKDecision::Fail(_)));
    }
}
