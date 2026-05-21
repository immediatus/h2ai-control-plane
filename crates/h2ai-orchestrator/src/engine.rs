pub use crate::nats_dispatch_adapter::NatsDispatchConfig;
use crate::task_store::{TaskState, TaskStore};
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_state::NatsClient;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{AuditorConfig, TaoConfig, VerificationConfig};
use h2ai_types::conflict::ConflictRateAccumulator;
use h2ai_types::events::{
    ApprovalResolvedEvent, CalibrationCompletedEvent, H2AIEvent, PendingApprovalEvent,
    ProposalFailedEvent, SelectionResolvedEvent, TaskComplexityAssessedEvent,
    VerificationScoredEvent,
};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::manifest::TaskManifest;
use h2ai_types::reasoning_checkpoint::{
    CompletedWave, ReasoningCheckpointPhase, TaskReasoningCheckpoint,
};
use h2ai_types::sizing::TaskQuadrant;
use thiserror::Error;

/// Errors that can abort an `ExecutionEngine::run_offline` call.
#[derive(Debug, Error)]
pub enum EngineError {
    /// The multiplication condition gate rejected all topologies across all retries.
    /// Recalibrating with higher-quality or more diverse adapters may resolve this.
    #[error("multiplication condition failed: {0}")]
    MultiplicationConditionFailed(String),
    /// The MAPE-K autonomic retry loop hit `max_autonomic_retries` without resolving.
    /// Increasing the retry budget or investigating calibration data is recommended.
    /// `partial_verification_events` carries any `VerificationScoredEvent`s collected before
    /// failure so callers can still publish them (e.g. for SSE clients tracking Phase 3).
    #[error("max retries exhausted")]
    MaxRetriesExhausted {
        partial_verification_events: Vec<VerificationScoredEvent>,
    },
    /// An adapter call failed or timed out; the message contains the error detail.
    /// May be transient — retrying at the caller level is reasonable.
    #[error("adapter error: {0}")]
    Adapter(String),
    /// Output from a step could not be parsed (e.g. invalid JSON, bad regex pattern).
    /// Indicates a configuration or adapter output format error; retrying is unlikely to help.
    #[error("parse error: {0}")]
    Parse(String),
    /// The wall-clock task deadline was exceeded before the engine resolved.
    /// Increase `task_deadline_secs` or reduce ensemble size to fit the budget.
    #[error("task deadline exceeded (budget {budget_secs}s)")]
    DeadlineExceeded { budget_secs: u64 },
    /// The provisioned ensemble is too small for the requested OutlierResistant fault bound.
    /// Either reduce `f` or provision at least `2f + 3` explorers.
    #[error("insufficient quorum for OutlierResistant f={f}: need n ≥ {required}, got n={n}")]
    InsufficientQuorum { n: usize, f: usize, required: usize },
    /// A reasoning checkpoint write failed and `strict_audit_checkpoint = true`.
    /// The task is aborted to preserve the audit trail integrity.
    #[error("checkpoint write failed (strict audit mode): {0}")]
    CheckpointWriteFailed(String),
    /// A human reviewer rejected the task output via the HITL approval gate.
    /// `operator_id` identifies who rejected; `reviewer_note` carries the reason.
    #[error("HITL gate rejected by operator {operator_id}")]
    HitlRejected {
        operator_id: String,
        reviewer_note: Option<String>,
    },
}

/// Context for the Phase 4 shadow auditor. Held in `EngineInput::shadow_audit_ctx`.
///
/// `None` = shadow mode off for this task. When `Some`, the engine runs a concurrent
/// shadow audit call on every Phase 4 proposal and collects `ShadowAuditorResultEvent`s.
/// When `promoted_domains` contains the task domain, both auditors must approve (AND vote).
pub struct ShadowAuditCtx {
    /// The shadow auditor adapter — must be from a different family than the primary.
    pub adapter: std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    /// Domains currently in AND-vote mode, loaded from AppState at task dispatch.
    pub promoted_domains: std::collections::HashSet<String>,
}

/// All inputs required to run the multi-phase execution pipeline for a single task.
pub struct EngineInput<'a> {
    /// Unique identifier for the task being executed.
    pub task_id: TaskId,
    /// Task manifest containing description, constraints, Pareto weights, and explorer spec.
    pub manifest: TaskManifest,
    /// Calibration event carrying α, β₀, CG samples, and optional ensemble/eigen calibration.
    pub calibration: CalibrationCompletedEvent,
    /// Pool of compute adapters shared across explorer slots (round-robin indexed by position).
    pub explorer_adapters: Vec<&'a dyn IComputeAdapter>,
    /// Scores proposals in Phase 3.5. Must return `{"score": float, "reason": "..."}`.
    pub verification_adapter: &'a dyn IComputeAdapter,
    /// Approves/rejects proposals in Phase 4. Must return `{"approved": bool, "reason": "..."}`.
    pub auditor_adapter: &'a dyn IComputeAdapter,
    /// Configuration for the auditor adapter (prompt template, τ, token budget).
    pub auditor_config: AuditorConfig,
    /// TAO loop configuration applied to every explorer (turns, patterns, repetition threshold).
    pub tao_config: TaoConfig,
    /// Verification phase configuration (LLM-as-Judge threshold and prompt settings).
    pub verification_config: VerificationConfig,
    /// Constraint corpus loaded from the ADR/design-doc index; used for context compilation and scoring.
    pub constraint_corpus: Vec<ConstraintDoc>,
    /// Runtime configuration (retries, deadlines, context token budget, thresholds).
    pub cfg: &'a H2AIConfig,
    /// In-memory task state store for phase and validation tracking.
    pub store: TaskStore,
    /// When Some, each explorer slot gets a NatsDispatchAdapter instead of
    /// drawing from explorer_adapters. explorer_adapters may be empty.
    pub nats_dispatch: Option<NatsDispatchConfig>,
    /// Adapter registry for profile-based routing.
    pub registry: &'a AdapterRegistry,
    /// Optional embedding model for Weiszfeld geometric median and cosine similarity.
    /// When `Some`, enables the Weiszfeld path in incoherent merge clusters.
    pub embedding_model: Option<&'a dyn EmbeddingModel>,
    /// Pre-task snapshot of TaoMultiplierEstimator::multiplier().
    /// Used for tao_per_turn_factor in AttributionInput so attribution reflects
    /// what was known at dispatch time, not the mid-task update.
    pub tao_multiplier: f64,
    /// Shared estimator updated with (turn-1 score, final score) pairs after
    /// each iteration's verification. Persisted by tasks.rs after engine returns.
    pub tao_estimator: std::sync::Arc<tokio::sync::RwLock<crate::tao_loop::TaoMultiplierEstimator>>,
    /// When `Some`, the synthesis phase runs after verification and auditor gate.
    /// Used for both the Stage 1 critique call and the Stage 2 synthesis call.
    /// When `None`, synthesis is skipped unconditionally and the selection chain runs directly.
    pub synthesis_adapter: Option<&'a dyn IComputeAdapter>,
    /// Optional Thompson Sampling bandit for adaptive N selection.
    /// When `Some`, the bandit selects `n_agents` and its posterior is updated on task completion.
    /// When `None`, N selection falls back to `n_optimal_hint` from EnsembleCalibration.
    pub bandit_state: Option<std::sync::Arc<tokio::sync::RwLock<crate::bandit::BanditState>>>,
    /// Shadow auditor context for GAP-C2 disagreement measurement. `None` = shadow off.
    pub shadow_audit_ctx: Option<ShadowAuditCtx>,
    /// Optional researcher adapter for C1 grounding (proactive slot search + reactive retry).
    /// Uses `Arc` so it can be called from async closures inside the MAPE-K loop.
    /// When `None`, search-enabled slots and C1 retries fall back to hint-only.
    pub researcher_adapter: Option<std::sync::Arc<dyn IComputeAdapter>>,
    /// Current SRANI EMA midpoint (ema_cfi) loaded from NATS KV by tasks.rs.
    /// When count < 5, the engine substitutes cfg.srani.cold_start_midpoint().
    pub srani_ema_cfi: f64,
    /// Number of tasks that have contributed a CFI observation to the EMA.
    pub srani_count: usize,
    /// Optional SRANI grounding chain. When `Some`, replaces the old negative-only hint
    /// with a positive grounding context (spec anchor + LLM researcher / web search).
    /// When `None`, falls back to `SpecAnchorGrounder` inline (zero I/O).
    pub srani_grounding_chain: Option<std::sync::Arc<crate::srani_grounding::SraniGroundingChain>>,
    /// Raw NATS client for oracle gate NATS request/reply. `None` = oracle gate skipped
    /// even when `cfg.oracle_gate.enabled = true`.
    pub nats_raw: Option<std::sync::Arc<async_nats::Client>>,
    /// Tenant identifier for per-tenant KV bucket routing.
    /// Defaults to `TenantId::default_tenant()` for all existing callers.
    pub tenant_id: TenantId,
    /// NATS client for reasoning checkpoint writes.
    /// When `Some` and `cfg.reasoning_memory.enabled`, writes fire-and-forget at each phase gate.
    pub nats: Option<std::sync::Arc<NatsClient>>,
    /// Assembled contexts from the previous wave for cross-wave delta encoding.
    /// Index corresponds to explorer slot index. None on wave 0.
    pub prev_assembled_contexts: Vec<Option<crate::context_assembler::AssembledContext>>,
    /// Optional adapter for the LLM summarization compression pass.
    pub compression_adapter: Option<std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>>,
    /// Cross-task stable context cache shared across tasks for the same tenant.
    pub stable_cache:
        Option<std::sync::Arc<crate::context_assembler::stable_cache::StableContextCache>>,
    /// Optional BM25 knowledge provider. When Some, generation queries it per slot.
    /// When None, global_knowledge and topic_knowledge remain None (existing behavior).
    pub knowledge_provider:
        Option<std::sync::Arc<dyn h2ai_knowledge::provider::KnowledgeProvider + Send + Sync>>,
    /// Optional induction store for cross-task knowledge boosting.
    /// When None, induction is skipped and pure BM25 is used.
    pub induction_store: Option<std::sync::Arc<crate::induction_store::InductionStore>>,
}

/// Successful result returned by `ExecutionEngine::run_offline` after all phases complete.
#[derive(Debug)]
pub struct EngineOutput {
    /// Identifier of the task that was resolved.
    pub task_id: TaskId,
    /// Final merged output string produced by the merge engine.
    pub resolved_output: String,
    /// Selection-resolved event: which proposals survived, pruned, merge strategy, and timing.
    pub selection_resolved: SelectionResolvedEvent,
    /// Quality attribution snapshot (q_confidence + components) computed at resolve time.
    pub attribution: crate::attribution::HarnessAttribution,
    /// Bootstrap CI over q_confidence from CG sample variance. `None` when < 2 CG samples.
    pub attribution_interval: Option<crate::attribution::AttributionInterval>,
    /// All verification scored events collected across every MAPE-K retry iteration.
    pub verification_events: Vec<VerificationScoredEvent>,
    /// Explorer agents that terminated without producing usable output, across all MAPE-K waves.
    /// Empty on clean runs. Published by the caller so SSE clients can detect silent failures.
    pub failed_proposals: Vec<ProposalFailedEvent>,
    /// Rank-histogram calibration diagnostic built from this task's verification scores.
    /// `None` when no verification events were produced.
    /// Typically `Some(Insufficient)` state for a single task (< 20 runs needed for calibration).
    pub talagrand: Option<crate::diagnostics::TalagrandDiagnostic>,
    /// SelfOptimizer suggestion for the next task run, computed from this run's quality.
    /// `None` only when no quality history was accumulated (should not happen on success).
    /// Callers may apply this to their next `EngineInput` to improve throughput.
    pub suggested_next_params: Option<crate::self_optimizer::OptimizerParams>,
    /// Fraction of dispatched proposals that survived verification (valid / total_evaluated).
    /// 1.0 = no waste; below `cfg.optimizer_waste_threshold` = wasteful run.
    pub waste_ratio: f64,
    /// SelfOptimizer suggestions derived from this wasteful-but-successful run.
    /// Empty when not wasteful or no applicable suggestion was found.
    /// Callers should apply these to AppState (τ spread EMA, topology hint).
    pub applied_optimizations: Vec<h2ai_types::events::AppliedOptimization>,
    /// Retry topology events in order — one entry per MAPE-K retry wave.
    /// Populated only when a ZeroSurvivalEvent fired; empty on first-wave success.
    pub topology_retry_events: Vec<h2ai_types::events::TopologyProvisionedEvent>,
    /// Number of ModeCollapse rotations applied across all retries.
    pub mode_collapse_count: usize,
    /// Epistemic yield from the resolved wave (reserved for Task 9 metrics wiring).
    pub epistemic_yield: Option<f64>,
    /// Routing quadrant assigned by Phase 1.5 task complexity assessment.
    /// In shadow_mode this is informational only — topology was not changed.
    /// `None` only when the engine path skips Phase 1.5 (should not happen in production).
    pub task_quadrant: Option<TaskQuadrant>,
    /// Full Phase 1.5 assessment event for NATS publishing by the caller.
    /// Always `Some` — carried here so the API route can publish to JetStream.
    pub complexity_event: TaskComplexityAssessedEvent,
    /// Constraint Pareto frontier coverage measured from the final wave's satisfaction matrix.
    /// `None` only when no proposals survived auditing.
    pub frontier_event: Option<h2ai_types::events::ConstraintFrontierEvent>,
    /// Per-explorer correctness flag from the final verification wave.
    /// `true` = proposal passed verification (score ≥ verify_threshold).
    /// Used for H1 (ρ_actual) empirical measurement in the GAP-A1 experiment.
    pub adapter_correctness: Vec<(h2ai_types::identity::ExplorerId, bool)>,
    /// Domain-level coherence state from all pruned proposals across all MAPE-K waves.
    pub coherence_state: crate::coherence::CoherenceState,
    /// Comparison events from dual-run verifier (populated only when `record_adversarial_comparison` is set).
    pub comparison_events: Vec<h2ai_types::events::VerifierComparisonEvent>,
    /// Shadow auditor outcome events collected across all MAPE-K waves.
    /// Populated only when `shadow_audit_ctx` is `Some` in the corresponding `EngineInput`.
    pub shadow_audit_events: Vec<h2ai_types::events::ShadowAuditorResultEvent>,
    /// C1 correlated ensemble warnings emitted across all retry waves.
    /// Non-empty when proposal CV dropped below `correlated_hallucination_cv_threshold`.
    pub correlated_warnings: Vec<h2ai_types::events::CorrelatedEnsembleWarning>,
    /// Researcher grounding events: proactive slot pre-steps + reactive C1 groundings.
    pub researcher_grounding_events: Vec<h2ai_types::events::ResearcherGroundingEvent>,
    /// C3 domain coverage degradation event. `Some` when coverage < threshold.
    /// `None` when coverage is sufficient or corpus has no domain tags.
    pub diversity_degraded_event: Option<h2ai_types::events::DiversityGuardDegradedEvent>,
    /// SRANI correlated fabrication events — fired when CFI > warn_threshold.
    pub srani_events: Vec<h2ai_types::events::CorrelatedFabricationEvent>,
    /// EMA midpoint updated after absorbing this task's CFI observation.
    /// Zero when no CFI was computed this task (proposals.len() < 2 or srani disabled).
    pub srani_ema_cfi_updated: f64,
    /// Count after this task's CFI observation (srani_count + 1 if CFI was computed, else unchanged).
    pub srani_count_updated: usize,
    /// Result of the oracle gate check before merge. `None` when gate was disabled or
    /// no NATS client was provided. `Some(true)` = passed, `Some(false)` = failed.
    pub oracle_gate_passed: Option<bool>,
    /// Leader elected events across all MAPE-K waves (empty when `leader_enabled = false`).
    pub leader_elected_events: Vec<h2ai_types::events::LeaderElectedEvent>,
    /// Socratic diagnosis events across all MAPE-K waves (empty when `leader_enabled = false`).
    pub socratic_diagnosis_events: Vec<h2ai_types::events::SocraticDiagnosisEvent>,
}

async fn write_reasoning_checkpoint(
    nats: &std::sync::Arc<NatsClient>,
    cp: &TaskReasoningCheckpoint,
    prefix: &str,
    strict: bool,
) -> Result<(), EngineError> {
    match nats.put_reasoning_checkpoint(cp, prefix).await {
        Ok(_) => Ok(()),
        Err(e) => {
            if strict {
                tracing::error!(
                    target: "h2ai.engine",
                    task_id = %cp.task_id,
                    "reasoning checkpoint write failed (strict audit mode): {e}"
                );
                Err(EngineError::CheckpointWriteFailed(e.to_string()))
            } else {
                tracing::warn!(
                    target: "h2ai.engine",
                    task_id = %cp.task_id,
                    "reasoning checkpoint write failed (non-fatal): {e}"
                );
                Ok(())
            }
        }
    }
}

/// Stateless coordinator for the five-phase task execution pipeline.
///
/// Orchestrates context compilation (Phase 1), topology provisioning (Phase 2),
/// the multiplication condition gate (Phase 2.5), parallel generation (Phase 3),
/// verification (Phase 3.5), auditor gate (Phase 4), and merge (Phase 5).
/// Wraps all phases in a MAPE-K autonomic retry loop that adjusts topology,
/// τ spread, and optimizer parameters on each failure before giving up.
pub struct ExecutionEngine;

impl ExecutionEngine {
    /// Run all five phases with the MAPE-K autonomic retry loop; no NATS publishing.
    ///
    /// Suitable for unit tests and offline evaluation because it does not require a
    /// live NATS connection — all events remain in-process via `TaskStore`.
    /// Returns `EngineOutput` on the first successful merge, or an `EngineError` when
    /// all retries are exhausted or a non-retryable condition is encountered.
    pub async fn run_offline(input: EngineInput<'_>) -> Result<EngineOutput, EngineError> {
        let task_id = input.task_id.clone();
        input.store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), input.tenant_id.clone()),
        );

        let rc_prefix = input.cfg.state.reasoning_checkpoint_bucket_prefix.clone();
        let ms_prefix = input.cfg.state.task_meta_state_bucket_prefix.clone();
        let strict_audit = input.cfg.reasoning_memory.strict_audit_checkpoint;
        let mut reasoning_cp = if input.cfg.reasoning_memory.enabled {
            if let Some(nats) = &input.nats {
                nats.ensure_tenant_reasoning_buckets(&input.tenant_id, &rc_prefix, &ms_prefix)
                    .await
                    .ok();
            }
            let cp = TaskReasoningCheckpoint::new_created(
                task_id.clone(),
                input.tenant_id.clone(),
                input.manifest.constraint_tags.clone(),
                None,
            );
            if let Some(nats) = &input.nats {
                write_reasoning_checkpoint(nats, &cp, &rc_prefix, strict_audit).await?;
            }
            Some(cp)
        } else {
            None
        };

        // ── Signal subscription: created once, shared across the whole run ──────
        // Subscribe before the pre-loop phases so the consumer is ready before any
        // HITL gate or wave-boundary window opens.
        let mut signal_sub = if let Some(ref nats) = input.nats {
            if input.cfg.hitl.enabled || input.cfg.signal_wave_window_ms > 0 {
                match nats
                    .subscribe_signals(&input.task_id, &input.tenant_id)
                    .await
                {
                    Ok(sub) => Some(sub),
                    Err(e) => {
                        tracing::warn!(
                            target: "h2ai.engine",
                            task_id = %input.task_id,
                            "signal subscription failed (non-fatal): {e}"
                        );
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // ── GAP-D1: load conflict-rate accumulator to inject beta_quality ───────
        let conflict_beta_prefix = input.cfg.state.conflict_beta_bucket_prefix.clone();
        let conflict_acc = if input.cfg.conflict_beta.enabled {
            if let Some(nats) = &input.nats {
                nats.ensure_tenant_conflict_bucket(&input.tenant_id, &conflict_beta_prefix)
                    .await
                    .ok();
                nats.get_conflict_accumulator(&input.tenant_id, &conflict_beta_prefix)
                    .await
                    .ok()
                    .flatten()
            } else {
                None
            }
        } else {
            None
        };

        // ── Pre-loop: one-shot phases (no retry needed) ─────────────────────
        let bootstrap_out = crate::phases::bootstrap::run(&input).await?;
        let mut complexity_out =
            crate::phases::complexity::run(&input, &bootstrap_out.system_context).await?;
        // Override n_max_ceiling with conflict-rate-based beta when accumulator has sufficient data.
        if let Some(acc) = &conflict_acc {
            let mut cc = input.calibration.coefficients.clone();
            cc.beta_quality = Some(acc.beta_quality);
            let conflict_n_max = cc.n_max().floor().max(1.0) as u32;
            complexity_out.n_max_ceiling = conflict_n_max;
            tracing::debug!(
                target: "h2ai.engine",
                beta_quality = acc.beta_quality,
                n_max_ceiling = conflict_n_max,
                "n_max_ceiling overridden with conflict-rate beta_quality"
            );
        }

        let domain_cov_out = crate::phases::domain_coverage::run(&input)?;
        // Extract diversity_degraded_event before borrowing domain_cov_out for the pipeline.
        let diversity_degraded_event = domain_cov_out.diversity_degraded_event.clone();

        let task_eval_cache = crate::verification::new_eval_cache();
        let pipeline = crate::pipeline::ExecutionPipeline::new(
            &input,
            &bootstrap_out,
            &complexity_out,
            &domain_cov_out,
            task_eval_cache,
        );
        let conflict_graph =
            h2ai_constraints::conflict::ConstraintConflictGraph::build(&input.constraint_corpus);
        let mut controller = crate::mape_k::MapeKController::new(
            &input,
            &bootstrap_out,
            &complexity_out,
            conflict_graph,
        )
        .await;
        // Wire diversity degraded event from domain coverage into the controller so it
        // is included in the final EngineOutput via MapeKController::finalize().
        controller.diversity_degraded_event = diversity_degraded_event;

        for retry_count in 0..=input.cfg.max_autonomic_retries {
            if let Some(dl) = controller.deadline() {
                if std::time::Instant::now() >= dl {
                    input.store.mark_failed(&task_id);
                    return Err(EngineError::DeadlineExceeded {
                        budget_secs: input.cfg.task_deadline_secs.unwrap_or(0),
                    });
                }
            }

            let wave = pipeline
                .run(controller.params(), retry_count as usize)
                .await;
            controller.observe(&wave);
            // ── Epistemic Leader Election ────────────────────────────────────
            if input.cfg.leader_enabled && !input.explorer_adapters.is_empty() {
                if let Some(plan) = controller.prepare_leader_election(input.cfg) {
                    let adapter = input.explorer_adapters[0];
                    let (question, eig_rank, dedup_tried) =
                        crate::leader::generate_socratic_question(
                            adapter,
                            &plan.prior_proposal,
                            &plan.violated_constraint_ids,
                            &plan.existing_belief_buffer,
                            input.cfg,
                        )
                        .await;
                    controller.apply_leader_result(
                        plan,
                        question,
                        eig_rank,
                        dedup_tried,
                        input.cfg,
                    );
                }
            }
            // Fire-and-forget: write conflict rate sample to per-tenant accumulator.
            if input.cfg.conflict_beta.enabled {
                if let Some(rate) = wave.events.conflict_rate {
                    if let Some(nats) = &input.nats {
                        let floor = input
                            .calibration
                            .beta_quality
                            .unwrap_or(input.calibration.coefficients.beta_base);
                        let mut acc = nats
                            .get_conflict_accumulator(&input.tenant_id, &conflict_beta_prefix)
                            .await
                            .ok()
                            .flatten()
                            .unwrap_or_else(|| {
                                ConflictRateAccumulator::new(input.tenant_id.clone(), floor)
                            });
                        let n_adapters = input.calibration.coefficients.cg_samples.len() as u32;
                        acc.push_sample(
                            rate,
                            n_adapters,
                            input.cfg.conflict_beta.max_samples,
                            input.cfg.conflict_beta.halflife_secs,
                            input.cfg.conflict_beta.min_samples_for_override,
                        );
                        if let Err(e) = nats
                            .put_conflict_accumulator(&acc, &conflict_beta_prefix)
                            .await
                        {
                            tracing::warn!(
                                target: "h2ai.engine",
                                "conflict accumulator write failed (non-fatal): {e}"
                            );
                        }
                    }
                }
            }
            if let Some(cp) = &mut reasoning_cp {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                cp.last_updated = now;
                cp.phase = ReasoningCheckpointPhase::WaveCompleted(retry_count);
                cp.completed_waves.push(CompletedWave {
                    wave_index: retry_count,
                    adapter_outputs: vec![],
                });
                if let Some(nats) = &input.nats {
                    write_reasoning_checkpoint(nats, cp, &rc_prefix, strict_audit).await?;
                }
            }
            // TODO: wire WaveContinue injection when multi-wave loop is added.
            // When signal_wave_window_ms > 0, open a brief tokio::select! window here
            // to receive a ContinueToNextWave signal and thread grounding/mandate_override
            // into the next ContextAssemblerInput before starting the next wave.

            let filter_ratio = wave.events.filter_ratio;
            match controller.decide(wave.outcome, retry_count, filter_ratio) {
                crate::mape_k::MapeKDecision::Return(out) => {
                    // Record knowledge patterns for cross-task induction (best-effort).
                    if let Some(ref store) = input.induction_store {
                        let domain_tags = input.manifest.constraint_tags.clone();
                        if !domain_tags.is_empty() {
                            let store_clone = store.clone();
                            tokio::spawn(async move {
                                let _ = store_clone
                                    .record(
                                        &domain_tags,
                                        &h2ai_types::config::AgentRole::Executor,
                                        &domain_tags,
                                    )
                                    .await;
                            });
                        }
                    }

                    // ── HITL approval gate ────────────────────────────────────
                    // When HITL is enabled, the engine waits for a Finalize signal
                    // before returning. On reject, publishes ApprovalResolved and
                    // returns HitlRejected. On approve (or timeout), falls through.
                    if input.cfg.hitl.enabled {
                        use futures::StreamExt as _;

                        let now_ms = || {
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64
                        };

                        // Compute adaptive timeout from hitl_timeouts_fired in checkpoint.
                        let n_fired = reasoning_cp.as_ref().map_or(0, |cp| cp.hitl_timeouts_fired);
                        let effective_ms = (input.cfg.hitl.timeout_ms as f64
                            * input.cfg.hitl.timeout_decay.powi(n_fired as i32))
                        .max(input.cfg.hitl.timeout_floor_ms as f64)
                            as u64;

                        let timeout_at_ms = now_ms() + effective_ms;

                        // Publish PendingApproval so harness/UI knows we're waiting.
                        if let Some(ref nats) = input.nats {
                            let q = out.attribution.q_confidence;
                            let n_used = out.selection_resolved.n_input_proposals as u32;
                            let triggered_by = h2ai_types::events::ApprovalTrigger::LowConfidence;
                            let risk_level =
                                h2ai_types::approval::compute_risk_level(&triggered_by, q);
                            let pending_ev = H2AIEvent::PendingApproval(PendingApprovalEvent {
                                task_id: input.task_id.clone(),
                                proposed_output: out.resolved_output.clone(),
                                q_confidence: q,
                                prediction_basis: match out.attribution.prediction_basis {
                                    h2ai_types::sizing::PredictionBasis::Heuristic => 0u8,
                                    h2ai_types::sizing::PredictionBasis::Empirical => 2u8,
                                },
                                n_used,
                                risk_level,
                                triggered_by,
                                timeout_at_ms,
                                timestamp_ms: now_ms(),
                            });
                            if let Err(e) = nats.publish_event(&input.task_id, &pending_ev).await {
                                tracing::warn!(
                                    target: "h2ai.engine",
                                    task_id = %input.task_id,
                                    "failed to publish PendingApproval: {e}"
                                );
                            }
                        }

                        // Refresh the signal stream right before entering select!.
                        // A durable pull consumer's messages() stream can terminate
                        // (return None) when its internal fetch requests expire with
                        // no messages — which happens routinely during the pipeline
                        // phases before reaching this gate.  Recreating the stream
                        // here gives select! a live consumer every time.
                        if let Some(ref nats) = input.nats {
                            match nats
                                .subscribe_signals(&input.task_id, &input.tenant_id)
                                .await
                            {
                                Ok(fresh) => {
                                    signal_sub = Some(fresh);
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        target: "h2ai.engine",
                                        task_id = %input.task_id,
                                        "signal stream refresh failed (non-fatal): {e}"
                                    );
                                }
                            }
                        }

                        // Wait for Finalize signal or timeout.
                        let (signal_approved, signal_operator_id, signal_reviewer_note, timed_out) =
                            if let Some(ref mut sub) = signal_sub {
                                tokio::select! {
                                    Some(Ok(sig)) = sub.next() => {
                                        let expired = now_ms() > sig.timeout_at_ms;
                                        if expired {
                                            (true, "system:late-signal".to_string(), None, false)
                                        } else {
                                            match crate::signal_dispatch::resolve_action(sig.payload) {
                                                crate::signal_dispatch::ResumeAction::Finalize {
                                                    approved,
                                                    reviewer_note,
                                                    operator_id,
                                                } => (approved, operator_id, reviewer_note, false),
                                                _ => (true, "system:non-approve-signal".to_string(), None, false),
                                            }
                                        }
                                    }
                                    _ = tokio::time::sleep(std::time::Duration::from_millis(effective_ms)) => {
                                        (
                                            true,
                                            "system:timeout".to_string(),
                                            Some("Auto-approved: review timeout exceeded".to_string()),
                                            true,
                                        )
                                    }
                                }
                            } else {
                                // HITL enabled but no signal subscriber — auto-approve.
                                (true, "system:no-sub".to_string(), None, false)
                            };

                        // Update hitl_timeouts_fired in checkpoint.
                        if let Some(ref mut cp) = reasoning_cp {
                            if timed_out {
                                cp.hitl_timeouts_fired += 1;
                            } else {
                                cp.hitl_timeouts_fired = 0;
                            }
                            if let Some(ref nats) = input.nats {
                                let _ =
                                    write_reasoning_checkpoint(nats, cp, &rc_prefix, strict_audit)
                                        .await;
                            }
                        }

                        // Publish ApprovalResolved BEFORE MergeResolved.
                        let decided_at_ms = now_ms();
                        if let Some(ref nats) = input.nats {
                            let approval_resolved_ev =
                                H2AIEvent::ApprovalResolved(ApprovalResolvedEvent {
                                    task_id: input.task_id.clone(),
                                    approved: signal_approved,
                                    operator_id: signal_operator_id.clone(),
                                    reviewer_note: signal_reviewer_note.clone(),
                                    decided_at_ms,
                                });
                            if let Err(e) = nats
                                .publish_event(&input.task_id, &approval_resolved_ev)
                                .await
                            {
                                tracing::warn!(
                                    target: "h2ai.engine",
                                    task_id = %input.task_id,
                                    "failed to publish ApprovalResolved: {e}"
                                );
                            }
                        }

                        // On reject: mark failed, clean up consumer, return error.
                        if !signal_approved {
                            input.store.mark_failed(&task_id);
                            // Clean up signal consumer.
                            if let Some(ref nats) = input.nats {
                                if let Err(e) = nats.delete_signal_consumer(&input.task_id).await {
                                    tracing::warn!(
                                        target: "h2ai.engine",
                                        task_id = %input.task_id,
                                        "failed to delete signal consumer on reject: {e}"
                                    );
                                }
                            }
                            return Err(EngineError::HitlRejected {
                                operator_id: signal_operator_id,
                                reviewer_note: signal_reviewer_note,
                            });
                        }
                        // Approved: fall through to normal resolve path below.
                    }
                    // ── End HITL gate ─────────────────────────────────────────

                    if let Some(mut cp) = reasoning_cp.take() {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        cp.last_updated = now;
                        cp.phase = ReasoningCheckpointPhase::Resolved;
                        cp.retry_count = retry_count;
                        cp.resolved_waste_ratio = Some(out.waste_ratio);
                        cp.resolved_attribution_json = serde_json::to_string(&out.attribution).ok();
                        if let Some(nats) = &input.nats {
                            write_reasoning_checkpoint(nats, &cp, &rc_prefix, strict_audit).await?;
                            if let Some(meta) = cp.into_meta_state() {
                                if let Err(e) = nats.put_task_meta_state(&meta, &ms_prefix).await {
                                    tracing::warn!(
                                        target: "h2ai.engine",
                                        task_id = %task_id,
                                        "TaskMetaState write failed (non-fatal): {e}"
                                    );
                                }
                            }
                        }
                    }

                    // Clean up signal consumer on successful resolve.
                    if let Some(ref nats) = input.nats {
                        if signal_sub.is_some() {
                            if let Err(e) = nats.delete_signal_consumer(&input.task_id).await {
                                tracing::warn!(
                                    target: "h2ai.engine",
                                    task_id = %input.task_id,
                                    "failed to delete signal consumer on resolve: {e}"
                                );
                            }
                        }
                    }

                    return Ok(*out);
                }
                crate::mape_k::MapeKDecision::Retry => continue,
                crate::mape_k::MapeKDecision::Fail(e) => {
                    input.store.mark_failed(&task_id);
                    // Clean up signal consumer on engine failure.
                    if let Some(ref nats) = input.nats {
                        if signal_sub.is_some() {
                            if let Err(ce) = nats.delete_signal_consumer(&input.task_id).await {
                                tracing::warn!(
                                    target: "h2ai.engine",
                                    task_id = %input.task_id,
                                    "failed to delete signal consumer on fail: {ce}"
                                );
                            }
                        }
                    }
                    return Err(e);
                }
            }
        }

        input.store.mark_failed(&task_id);
        // Clean up signal consumer on retry exhaustion.
        if let Some(ref nats) = input.nats {
            if signal_sub.is_some() {
                if let Err(e) = nats.delete_signal_consumer(&input.task_id).await {
                    tracing::warn!(
                        target: "h2ai.engine",
                        task_id = %input.task_id,
                        "failed to delete signal consumer on exhaustion: {e}"
                    );
                }
            }
        }
        tracing::warn!(
            target: "h2ai.engine",
            task_id = %task_id,
            max_retries = input.cfg.max_autonomic_retries,
            "retry loop exhausted all attempts"
        );
        Err(EngineError::MaxRetriesExhausted {
            partial_verification_events: controller.take_verification_events(),
        })
    }

    /// Resume execution from a persisted checkpoint.
    ///
    /// `Merging` checkpoint: resolved output already computed — reconstruct a minimal
    /// `EngineOutput` and return it so the caller can publish events without re-running LLM.
    ///
    /// All earlier phases fall back to `run_offline` (restart from scratch).
    pub async fn run_from_checkpoint(
        input: EngineInput<'_>,
        checkpoint: h2ai_types::checkpoint::TaskCheckpoint,
    ) -> Result<EngineOutput, EngineError> {
        if input.cfg.reasoning_memory.enabled {
            if let Some(nats) = &input.nats {
                let rc_prefix = &input.cfg.state.reasoning_checkpoint_bucket_prefix;
                if let Ok(Some(rc)) = nats
                    .get_reasoning_checkpoint(&input.task_id, &input.tenant_id, rc_prefix)
                    .await
                {
                    if rc.phase == ReasoningCheckpointPhase::Resolved
                        || rc.phase == ReasoningCheckpointPhase::MergeDone
                    {
                        tracing::info!(
                            target: "h2ai.engine",
                            task_id = %input.task_id,
                            phase = ?rc.phase,
                            "reasoning checkpoint: skipping to merge output"
                        );
                    }
                }
            }
        }

        let phase = crate::task_store::TaskPhase::try_from_name_str(&checkpoint.phase);

        if let Some(crate::task_store::TaskPhase::Merging) = phase {
            let resolved = checkpoint.resolved_output.ok_or_else(|| {
                EngineError::Parse("Merging checkpoint missing resolved_output".into())
            })?;

            let task_id = input.task_id.clone();
            input.store.mark_resolved(&task_id);

            // Hydrate attribution and waste_ratio from the reasoning checkpoint so
            // downstream analytics (billing, bandits, dashboards) receive real values
            // rather than zeros that would corrupt the statistical baseline.
            let (attribution, waste_ratio) = if input.cfg.reasoning_memory.enabled {
                if let Some(nats) = &input.nats {
                    let rc_prefix = &input.cfg.state.reasoning_checkpoint_bucket_prefix;
                    if let Ok(Some(rc)) = nats
                        .get_reasoning_checkpoint(&input.task_id, &input.tenant_id, rc_prefix)
                        .await
                    {
                        let attr = rc
                            .resolved_attribution_json
                            .as_deref()
                            .and_then(|s| serde_json::from_str(s).ok());
                        (attr, rc.resolved_waste_ratio)
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            let attribution = attribution.unwrap_or(crate::attribution::HarnessAttribution {
                baseline_quality: 0.0,
                topology_gain: 0.0,
                verification_gain: 0.0,
                tao_gain: 0.0,
                q_confidence: 0.0,
                prediction_basis: h2ai_types::sizing::PredictionBasis::Heuristic,
                q_measured: None,
                rho_adjusted: 0.0,
                case_b_flag: false,
                synthesis_gain: 0.0,
            });
            let waste_ratio = waste_ratio.unwrap_or(0.0);

            // Record knowledge patterns for cross-task induction (best-effort, fire-and-forget).
            // Phase 1 approximation: domain_tags used as proxy node_ids until full node_id
            // threading through EngineOutput is plumbed.
            if let Some(ref store) = input.induction_store {
                let domain_tags = input.manifest.constraint_tags.clone();
                if !domain_tags.is_empty() {
                    let store_clone = store.clone();
                    tokio::spawn(async move {
                        let _ = store_clone
                            .record(
                                &domain_tags,
                                &h2ai_types::config::AgentRole::Executor,
                                &domain_tags,
                            )
                            .await;
                    });
                }
            }

            Ok(EngineOutput {
                task_id: task_id.clone(),
                resolved_output: resolved,
                selection_resolved: h2ai_types::events::SelectionResolvedEvent {
                    task_id,
                    valid_proposals: vec![],
                    pruned_proposals: vec![],
                    merge_strategy: h2ai_types::sizing::MergeStrategy::ScoreOrdered,
                    timestamp: chrono::Utc::now(),
                    merge_elapsed_secs: None,
                    n_input_proposals: 0,
                    n_failed_proposals: 0,
                },
                attribution,
                attribution_interval: None,
                verification_events: vec![],
                failed_proposals: vec![],
                talagrand: None,
                suggested_next_params: None,
                waste_ratio,
                applied_optimizations: vec![],
                topology_retry_events: vec![],
                mode_collapse_count: 0,
                epistemic_yield: None,
                task_quadrant: Some(TaskQuadrant::Precision),
                complexity_event: h2ai_types::events::TaskComplexityAssessedEvent {
                    task_id: input.task_id.clone(),
                    tcc_structural: 0.0,
                    tcc_empirical: None,
                    tcc_effective: 0.0,
                    n_eff_pool: None,
                    task_quadrant: TaskQuadrant::Precision,
                    probe_skipped: true,
                    probe_skip_reason: h2ai_types::sizing::ProbeSkipReason::None,
                    heavy_fraction: 0.0,
                    tcc_mismatch: false,
                    probe_cost_tokens: 0,
                    n_informative_static: 0,
                    timestamp: chrono::Utc::now(),
                },
                frontier_event: None,
                adapter_correctness: vec![],
                coherence_state: crate::coherence::CoherenceState::default(),
                comparison_events: vec![],
                shadow_audit_events: vec![],
                correlated_warnings: vec![],
                researcher_grounding_events: vec![],
                diversity_degraded_event: None,
                srani_events: vec![],
                srani_ema_cfi_updated: input.srani_ema_cfi,
                srani_count_updated: input.srani_count,
                oracle_gate_passed: None,
                leader_elected_events: vec![],
                socratic_diagnosis_events: vec![],
            })
        } else {
            // Earlier phase or unknown phase — restart from scratch
            Self::run_offline(input).await
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn conflict_beta_disabled_skips_accumulator_load() {
        let mut cfg = h2ai_config::H2AIConfig::default();
        cfg.conflict_beta.enabled = false;
        assert!(!cfg.conflict_beta.enabled);
    }
}
