pub use crate::nats_dispatch_adapter::NatsDispatchConfig;
use crate::task_store::{TaskState, TaskStore};
use h2ai_autonomic::retry_accumulator::RetryAccumulator;
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_state::NatsClient;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{AuditorConfig, TaoConfig, VerificationConfig};
use h2ai_types::conflict::ConflictRateAccumulator;
use h2ai_types::events::{
    ApprovalResolvedEvent, BudgetExhaustedEvent, CalibrationCompletedEvent,
    CostThresholdWarningEvent, H2AIEvent, PendingApprovalEvent, ProposalFailedEvent,
    SelectionResolvedEvent, TaskComplexityAssessedEvent, VerificationScoredEvent,
};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::manifest::TaskManifest;
use h2ai_types::reasoning_checkpoint::{
    CompletedWave, ReasoningCheckpointPhase, TaskReasoningCheckpoint,
};
use h2ai_types::sizing::TaskQuadrant;
use thiserror::Error;

/// Appended to the verifier system prompt when `verifier_decomposition_enabled = true` and
/// the complexity probe rated the task at or above `decompose_threshold`.
/// Instructs the judge to decompose verification into sub-claims and tag uncomputable ones as
/// BEYOND_BUDGET rather than UNVERIFIED, so the MAPE-K controller can distinguish
/// "content rejected" from "computation limit reached".
const BEYOND_BUDGET_VERIFIER_ADDENDUM: &str = "\n\n\
    --- Sub-claim Verification Mode ---\n\
    This task requires multi-step proof verification. Decompose your verification \
    into sub-claims and label each as:\n\
    - VERIFIED: sub-claim checks out\n\
    - UNVERIFIED: sub-claim fails\n\
    - BEYOND_BUDGET: sub-claim requires more computation than available in this pass\n\
    Score = VERIFIED / (VERIFIED + UNVERIFIED). Report BEYOND_BUDGET items separately; \
    do NOT score them as 0.0.";

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
        /// Highest-scoring partial proposal from the entire session, for HITL surfacing.
        /// `None` when no partial passes existed (all proposals scored 0 on every check).
        best_partial_text: Option<String>,
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
    /// The provisioned ensemble is too small for the requested `OutlierResistant` fault bound.
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
    /// Domains currently in AND-vote mode, loaded from `AppState` at task dispatch.
    pub promoted_domains: std::collections::HashSet<String>,
    /// When true, shadow vote is always binding regardless of domain promotion history.
    pub strict: bool,
}

/// Fraction of `VerificationScoredEvent`s where `passed == true`.
/// Returns `1.0` for an empty slice (no information → assume stable).
pub fn consensus_agreement_rate_from_events(
    events: &[h2ai_types::events::VerificationScoredEvent],
) -> f64 {
    if events.is_empty() {
        return 1.0;
    }
    let passed = events.iter().filter(|e| e.passed).count();
    passed as f64 / events.len() as f64
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
    /// When Some, each explorer slot gets a `NatsDispatchAdapter` instead of
    /// drawing from `explorer_adapters`. `explorer_adapters` may be empty.
    pub nats_dispatch: Option<NatsDispatchConfig>,
    /// Adapter registry for profile-based routing.
    pub registry: &'a AdapterRegistry,
    /// Optional embedding model for Weiszfeld geometric median and cosine similarity.
    /// When `Some`, enables the Weiszfeld path in incoherent merge clusters.
    pub embedding_model: Option<&'a dyn EmbeddingModel>,
    /// Pre-task snapshot of `TaoMultiplierEstimator::multiplier()`.
    /// Used for `tao_per_turn_factor` in `AttributionInput` so attribution reflects
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
    /// When `None`, N selection falls back to `n_optimal_hint` from `EnsembleCalibration`.
    pub bandit_state: Option<std::sync::Arc<tokio::sync::RwLock<crate::bandit::BanditState>>>,
    /// Shadow auditor context disagreement measurement. `None` = shadow off.
    pub shadow_audit_ctx: Option<ShadowAuditCtx>,
    /// Optional researcher adapter for C1 grounding (proactive slot search + reactive retry).
    /// Uses `Arc` so it can be called from async closures inside the MAPE-K loop.
    /// When `None`, search-enabled slots and C1 retries fall back to hint-only.
    pub researcher_adapter: Option<std::sync::Arc<dyn IComputeAdapter>>,
    /// Current SRANI EMA midpoint (`ema_cfi`) loaded from NATS KV by tasks.rs.
    /// When count < 5, the engine substitutes `cfg.srani.cold_start_midpoint()`.
    pub srani_ema_cfi: f64,
    /// Number of tasks that have contributed a CFI observation to the EMA.
    pub srani_count: usize,
    /// Optional SRANI grounding chain. When `Some`, replaces the old negative-only hint
    /// with a positive grounding context (spec anchor + LLM researcher / web search).
    /// When `None`, falls back to `SpecAnchorGrounder` inline (zero I/O).
    pub srani_grounding_chain: Option<std::sync::Arc<crate::srani_grounding::SraniGroundingChain>>,
    /// Dedicated grounding chain gap researcher (DDG search + distiller).
    pub gap_research_chain: Option<std::sync::Arc<crate::srani_grounding::SraniGroundingChain>>,
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
    /// When None, `global_knowledge` and `topic_knowledge` remain None (existing behavior).
    pub knowledge_provider:
        Option<std::sync::Arc<dyn h2ai_knowledge::provider::KnowledgeProvider + Send + Sync>>,
    /// Optional induction store for cross-task knowledge boosting.
    /// When None, induction is skipped and pure BM25 is used.
    pub induction_store: Option<std::sync::Arc<crate::induction_store::InductionStore>>,
    /// ORCA conformal margin from `DriftMonitor::active_conformal_margin()`.
    /// Subtracted from `verification_config.threshold` at engine start. Zero = no active drift.
    pub conformal_margin: f64,
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
    /// Quality attribution snapshot (`q_confidence` + components) computed at resolve time.
    pub attribution: crate::attribution::HarnessAttribution,
    /// Bootstrap CI over `q_confidence` from CG sample variance. `None` when < 2 CG samples.
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
    /// `SelfOptimizer` suggestion for the next task run, computed from this run's quality.
    /// `None` only when no quality history was accumulated (should not happen on success).
    /// Callers may apply this to their next `EngineInput` to improve throughput.
    pub suggested_next_params: Option<crate::self_optimizer::OptimizerParams>,
    /// Fraction of dispatched proposals that survived verification (valid / `total_evaluated`).
    /// 1.0 = no waste; below `cfg.optimizer_waste_threshold` = wasteful run.
    pub waste_ratio: f64,
    /// `SelfOptimizer` suggestions derived from this wasteful-but-successful run.
    /// Empty when not wasteful or no applicable suggestion was found.
    /// Callers should apply these to `AppState` (τ spread EMA, topology hint).
    pub applied_optimizations: Vec<h2ai_types::events::AppliedOptimization>,
    /// Retry topology events in order — one entry per MAPE-K retry wave.
    /// Populated only when a `ZeroSurvivalEvent` fired; empty on first-wave success.
    pub topology_retry_events: Vec<h2ai_types::events::TopologyProvisionedEvent>,
    /// Number of `ModeCollapse` rotations applied across all retries.
    pub mode_collapse_count: usize,
    /// Epistemic yield from the resolved wave (reserved for Task 9 metrics wiring).
    pub epistemic_yield: Option<f64>,
    /// Routing quadrant assigned by Phase 1.5 task complexity assessment.
    /// In `shadow_mode` this is informational only — topology was not changed.
    /// `None` only when the engine path skips Phase 1.5 (should not happen in production).
    pub task_quadrant: Option<TaskQuadrant>,
    /// Full Phase 1.5 assessment event for NATS publishing by the caller.
    /// Always `Some` — carried here so the API route can publish to `JetStream`.
    pub complexity_event: TaskComplexityAssessedEvent,
    /// Constraint Pareto frontier coverage measured from the final wave's satisfaction matrix.
    /// `None` only when no proposals survived auditing.
    pub frontier_event: Option<h2ai_types::events::ConstraintFrontierEvent>,
    /// Per-explorer correctness flag from the final verification wave.
    /// `true` = proposal passed verification (score ≥ `verify_threshold`).
    /// Used for H1 (`ρ_actual`) empirical measurement in the experiment.
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
    /// SRANI correlated fabrication events — fired when CFI > `warn_threshold`.
    pub srani_events: Vec<h2ai_types::events::CorrelatedFabricationEvent>,
    /// EMA midpoint updated after absorbing this task's CFI observation.
    /// Zero when no CFI was computed this task (`proposals.len()` < 2 or srani disabled).
    pub srani_ema_cfi_updated: f64,
    /// Count after this task's CFI observation (`srani_count` + 1 if CFI was computed, else unchanged).
    pub srani_count_updated: usize,
    /// Result of the oracle gate check before merge. `None` when gate was disabled or
    /// no NATS client was provided. `Some(true)` = passed, `Some(false)` = failed.
    pub oracle_gate_passed: Option<bool>,
    /// Leader elected events across all MAPE-K waves (empty when `leader_enabled = false`).
    pub leader_elected_events: Vec<h2ai_types::events::LeaderElectedEvent>,
    /// Socratic diagnosis events across all MAPE-K waves (empty when `leader_enabled = false`).
    pub socratic_diagnosis_events: Vec<h2ai_types::events::SocraticDiagnosisEvent>,
    /// Fraction of verification events that passed. Fed to `DriftMonitor` by tasks.rs.
    pub consensus_agreement_rate: Option<f64>,
}

async fn write_reasoning_checkpoint(
    nats: &std::sync::Arc<NatsClient>,
    cp: &TaskReasoningCheckpoint,
    prefix: &str,
    strict: bool,
) -> Result<(), EngineError> {
    match nats.put_reasoning_checkpoint(cp, prefix).await {
        Ok(()) => Ok(()),
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
    pub async fn run_offline(mut input: EngineInput<'_>) -> Result<EngineOutput, EngineError> {
        // ORCA conformal margin: widen verification gate under active drift.
        if input.conformal_margin > 0.0 {
            input.verification_config.threshold =
                (input.verification_config.threshold - input.conformal_margin).max(0.0);
        }
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

        // ── load conflict-rate accumulator to inject beta_quality ───────
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
            let conflict_n_max = cc.n_max().floor().max(3.0) as u32;
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

        // ── pre-loop complexity probe ─────────────────────────────────────
        // One cheap LLM call before the restart loop so the result can be
        // wired into each controller iteration and `input.verification_config`
        // can be mutated freely (no pipeline borrow active here).
        //
        // Adapter resolution order:
        //   1. named adapter from registry (config string, e.g. "researcher")
        //   2. researcher adapter (preferred — cheapest, instruction-following)
        //   3. first explorer adapter (fallback)
        //   4. None → probe skipped, result absent on controller
        let probe_result: Option<h2ai_autonomic::complexity_probe::ComplexityProbeResult> =
            if input.cfg.complexity_routing.enabled {
                let probe_adapter: Option<&dyn h2ai_types::adapter::IComputeAdapter> = input
                    .registry
                    .get_by_name(&input.cfg.complexity_routing.complexity_probe_adapter)
                    .or(input.researcher_adapter.as_deref())
                    .or_else(|| input.explorer_adapters.first().copied());
                if let Some(adapter) = probe_adapter {
                    let probe = h2ai_autonomic::complexity_probe::ComplexityProbe::new(
                        input.cfg.complexity_routing.clone(),
                    );
                    let t0 = std::time::Instant::now();
                    let result = probe.run(&input.manifest.description, adapter).await;
                    let latency_ms = t0.elapsed().as_millis() as u64;
                    tracing::info!(
                        target: "h2ai.engine",
                        task_id = %task_id,
                        complexity = result.complexity,
                        rationale = %result.rationale,
                        decompose_recommended = result.decompose_recommended,
                        latency_ms,
                        "complexity probe completed"
                    );
                    if let Some(ref nats) = input.nats {
                        let ev = h2ai_types::events::H2AIEvent::ComplexityProbe(
                            h2ai_types::events::ComplexityProbeEvent {
                                task_id: task_id.clone(),
                                complexity: result.complexity,
                                rationale: result.rationale.clone(),
                                decompose_recommended: result.decompose_recommended,
                                probe_latency_ms: latency_ms,
                                timestamp: chrono::Utc::now(),
                            },
                        );
                        let _ = nats.publish_event(&input.task_id, &ev).await;
                    }
                    Some(result)
                } else {
                    tracing::debug!(
                        target: "h2ai.engine",
                        task_id = %task_id,
                        "complexity probe skipped — no researcher/explorer adapter"
                    );
                    None
                }
            } else {
                None
            };
        // Inject sub-claim BEYOND_BUDGET verifier addendum when the probe rated the task
        // as high-complexity and verifier_decomposition_enabled is set.  Must happen before
        // the restart loop so the pipeline sees the mutated system prompt.
        if let Some(ref probe) = probe_result {
            if input.cfg.complexity_routing.verifier_decomposition_enabled
                && probe.complexity >= input.cfg.complexity_routing.decompose_threshold
            {
                input
                    .verification_config
                    .evaluator_system_prompt
                    .push_str(BEYOND_BUDGET_VERIFIER_ADDENDUM);
            }
        }

        // Restart counter — capped at 1 to prevent infinite repair loops.
        let mut spec_ambiguous_restarts: u32 = 0;

        // ── restart loop ───────────────────────────────────────────────
        // Each iteration builds a fresh pipeline + controller from the (possibly
        // repaired) `input.constraint_corpus`.  The loop exits via `return` on
        // success/failure, or via `continue 'restart` after a successful spec
        // repair.  The pipeline is enclosed in an inner block so it is dropped
        // before `input.constraint_corpus` is mutated for the next iteration.
        'restart: loop {
            // ── Inner scope: pipeline borrows &input; dropped before repair ──────
            // Returns (None, controller) when the retry loop exhausts all attempts
            // normally, or (Some(sa_info), controller) on a SpecAmbiguous signal.
            let (spec_ambiguous_signal, complexity_overflow_graft_signal, mut controller) = {
                let task_eval_cache = crate::verification::new_eval_cache();
                let pipeline = crate::pipeline::ExecutionPipeline::new(
                    &input,
                    &bootstrap_out,
                    &complexity_out,
                    &domain_cov_out,
                    task_eval_cache,
                );
                let conflict_graph = h2ai_constraints::conflict::ConstraintConflictGraph::build(
                    &input.constraint_corpus,
                );
                let mut controller = crate::mape_k::MapeKController::new(
                    &input,
                    &bootstrap_out,
                    &complexity_out,
                    conflict_graph,
                )
                .await;
                // Wire diversity degraded event from domain coverage into the controller so it
                // is included in the final EngineOutput via MapeKController::finalize().
                controller.diversity_degraded_event = diversity_degraded_event.clone();
                if let Some(ref chain) = input.gap_research_chain {
                    controller.gap_grounding_chain = Some(chain.clone());
                }

                // Wire pre-loop probe result into controller (probe ran before 'restart loop).
                if let Some(ref probe) = probe_result {
                    controller.set_probe_result(probe.clone());
                }

                if input.cfg.complexity_routing.enabled && !controller.corpus_synthesis_viable {
                    tracing::warn!(
                        target: "h2ai.engine",
                        task_id = %task_id,
                        "complexity routing enabled but corpus has no binary_checks \
                         — ComplexityOverflow{{graft_first:true}} is suppressed; \
                         normal retry loop will run to exhaustion"
                    );
                }

                // Per-task OSP retry accumulator. Local variable — never stored in NATS KV or shared
                // state. Reset on Resolved (spec §A.3: state-leakage prevention).
                let mut osp_accumulator = RetryAccumulator::new();

                // Carry SpecAmbiguous info out of the inner block (pipeline-drop boundary).
                let mut spec_ambiguous_signal: Option<(String, usize, Vec<String>)> = None;
                // Signal the post-loop synthesis wave to run on ComplexityOverflow
                // with `graft_first = true`.  Decoupled from `synthesis_wave_enabled` so a
                // ceiling-detected task can run grafting even when the feature flag is off.
                let mut complexity_overflow_graft_signal = false;

                'wave: for retry_count in 0..=input.cfg.max_autonomic_retries {
                    if let Some(dl) = controller.deadline() {
                        if std::time::Instant::now() >= dl {
                            input.store.mark_failed(&task_id);
                            return Err(EngineError::DeadlineExceeded {
                                budget_secs: input.cfg.task_deadline_secs.unwrap_or(0),
                            });
                        }
                    }

                    // ── AgentDropout N-reduction ────────────────────────────────
                    // On retry ≥ 2, if N_eff from the previous wave was below threshold,
                    // reduce n_agents to avoid wasting tokens on correlated agents
                    // (Wang et al. ACL 2025, arXiv 2503.18891: −21.6% tokens, +1.14% perf).
                    let mut wave_params = controller.params();
                    // ── GAP-L1: TEE N escalation ─────────────────────────────────────────────
                    if input.cfg.tiered_exit.enabled {
                        let tee = &input.cfg.tiered_exit;
                        let base_n = if retry_count == 0 {
                            if input.cfg.complexity_routing.enabled {
                                if let Some(ref probe) = controller.probe_result {
                                    (probe.complexity as u32).clamp(tee.min_n, tee.max_n)
                                } else {
                                    tee.min_n
                                }
                            } else {
                                tee.min_n
                            }
                        } else {
                            tee.n_for_wave(retry_count, input.cfg.max_autonomic_retries)
                        };
                        wave_params.optimizer.n_agents = base_n;
                        tracing::debug!(
                            target: "h2ai.engine",
                            task_id = %task_id,
                            retry_count,
                            n_agents = base_n,
                            "TEE: set n_agents for wave"
                        );
                    }
                    if input.cfg.complexity_routing.agent_dropout.enabled && retry_count >= 2 {
                        let n_eff = controller.last_wave_n_eff();
                        let dropout_cfg = &input.cfg.complexity_routing.agent_dropout;
                        if n_eff < dropout_cfg.n_eff_dropout_threshold {
                            let n = wave_params.optimizer.n_agents as usize;
                            let drop = (n as f64 * (1.0 - n_eff)).floor() as usize;
                            let keep = n.saturating_sub(drop).max(3.min(n));
                            tracing::debug!(
                                target: "h2ai.engine",
                                task_id = %task_id,
                                retry_count,
                                n_eff,
                                n_before = n,
                                n_after = keep,
                                "AgentDropout: reducing n_agents due to low N_eff"
                            );
                            wave_params.optimizer.n_agents = keep as u32;
                        }
                    }
                    let wave = pipeline
                        .run(
                            wave_params,
                            retry_count as usize,
                            Some(&mut osp_accumulator),
                            input.cfg.osp.as_ref(),
                        )
                        .await;
                    controller.observe(&wave);

                    // ── GAP-H3: charge token budget ───────────────────────────────────────
                    {
                        let wave_cost = match &wave.outcome {
                            crate::mape_k::PipelineOutcome::Resolved(m) => m.wave_token_cost,
                            _ => wave.events.wave_token_cost,
                        };
                        controller.observe_wave_tokens(wave_cost);

                        if input.cfg.cost_guard.enabled {
                            let frac = input.cfg.cost_guard.fraction_used(controller.tokens_used());
                            if frac >= input.cfg.cost_guard.budget_warning_fraction {
                                let used = controller.tokens_used();
                                let budget = input.cfg.cost_guard.budget_tokens_per_task;
                                tracing::warn!(
                                    target: "h2ai.engine",
                                    task_id = %task_id,
                                    tokens_used = used,
                                    budget_tokens = budget,
                                    fraction_used = frac,
                                    "CostGuard: token budget warning threshold reached"
                                );
                                if let Some(ref nats) = input.nats {
                                    if let Err(e) = nats
                                        .publish_event(
                                            &task_id,
                                            &H2AIEvent::CostThresholdWarning(
                                                CostThresholdWarningEvent {
                                                    task_id: task_id.clone(),
                                                    tokens_used: used,
                                                    budget_tokens: budget,
                                                    fraction_used: frac,
                                                    timestamp: chrono::Utc::now(),
                                                },
                                            ),
                                        )
                                        .await
                                    {
                                        tracing::warn!(
                                            target: "h2ai.engine",
                                            task_id = %task_id,
                                            "failed to publish CostThresholdWarningEvent: {e}"
                                        );
                                    }
                                }
                            }
                        }
                    }

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
                                    .get_conflict_accumulator(
                                        &input.tenant_id,
                                        &conflict_beta_prefix,
                                    )
                                    .await
                                    .ok()
                                    .flatten()
                                    .unwrap_or_else(|| {
                                        ConflictRateAccumulator::new(input.tenant_id.clone(), floor)
                                    });
                                let n_adapters =
                                    input.calibration.coefficients.cg_samples.len() as u32;
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

                    // ── cold-check researcher (no-op when gap_i1.enabled = false) ──
                    controller.run_gap_i1_research().await;

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
                            // Fires when: hitl enabled AND NOT oracle task AND
                            // (require_approval flag OR q_confidence < threshold).
                            // Oracle tasks bypass — programmatic oracle verdict is sufficient.
                            let hitl_oracle_bypass = input.manifest.oracle.is_some();
                            let hitl_q = out.attribution.q_confidence;
                            let needs_hitl = input.cfg.hitl.enabled
                                && !hitl_oracle_bypass
                                && (input.manifest.require_approval
                                    || hitl_q < input.cfg.hitl.confidence_threshold);
                            if needs_hitl {
                                use futures::StreamExt as _;

                                let now_ms = || {
                                    std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_millis() as u64
                                };

                                // Compute adaptive timeout from hitl_timeouts_fired in checkpoint.
                                let n_fired =
                                    reasoning_cp.as_ref().map_or(0, |cp| cp.hitl_timeouts_fired);
                                let effective_ms = (input.cfg.hitl.timeout_ms as f64
                                    * input.cfg.hitl.timeout_decay.powi(n_fired as i32))
                                .max(input.cfg.hitl.timeout_floor_ms as f64)
                                    as u64;

                                let timeout_at_ms = now_ms() + effective_ms;

                                // Update in-memory store so GET /tasks/{id} reflects the HITL state.
                                input.store.set_awaiting_approval(&input.task_id);

                                // Publish PendingApproval so harness/UI knows we're waiting.
                                if let Some(ref nats) = input.nats {
                                    let q = out.attribution.q_confidence;
                                    let n_used = out.selection_resolved.n_input_proposals as u32;
                                    let triggered_by =
                                        h2ai_types::events::ApprovalTrigger::LowConfidence;
                                    let risk_level =
                                        h2ai_types::approval::compute_risk_level(&triggered_by, q);
                                    let pending_ev =
                                        H2AIEvent::PendingApproval(PendingApprovalEvent {
                                            task_id: input.task_id.clone(),
                                            proposed_output: out.resolved_output.clone(),
                                            q_confidence: q,
                                            prediction_basis: match out.attribution.prediction_basis
                                            {
                                                h2ai_types::sizing::PredictionBasis::Heuristic => {
                                                    0u8
                                                }
                                                h2ai_types::sizing::PredictionBasis::Empirical => {
                                                    2u8
                                                }
                                            },
                                            n_used,
                                            risk_level,
                                            triggered_by,
                                            timeout_at_ms,
                                            timestamp_ms: now_ms(),
                                        });
                                    if let Err(e) =
                                        nats.publish_event(&input.task_id, &pending_ev).await
                                    {
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
                                let (
                                    signal_approved,
                                    signal_operator_id,
                                    signal_reviewer_note,
                                    timed_out,
                                ) = if let Some(ref mut sub) = signal_sub {
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
                                        () = tokio::time::sleep(std::time::Duration::from_millis(effective_ms)) => {
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
                                        let _ = write_reasoning_checkpoint(
                                            nats,
                                            cp,
                                            &rc_prefix,
                                            strict_audit,
                                        )
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
                                        if let Err(e) =
                                            nats.delete_signal_consumer(&input.task_id).await
                                        {
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
                                cp.resolved_attribution_json =
                                    serde_json::to_string(&out.attribution).ok();
                                if let Some(nats) = &input.nats {
                                    write_reasoning_checkpoint(nats, &cp, &rc_prefix, strict_audit)
                                        .await?;
                                    if let Some(meta) = cp.into_meta_state() {
                                        if let Err(e) =
                                            nats.put_task_meta_state(&meta, &ms_prefix).await
                                        {
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
                                    if let Err(e) =
                                        nats.delete_signal_consumer(&input.task_id).await
                                    {
                                        tracing::warn!(
                                            target: "h2ai.engine",
                                            task_id = %input.task_id,
                                            "failed to delete signal consumer on resolve: {e}"
                                        );
                                    }
                                }
                            }

                            // Reset OSP accumulator on successful resolve (spec §A.3).
                            // NOT called on ZeroSurvival — accumulation must persist across retries.
                            osp_accumulator.reset();

                            let mut out = *out;
                            out.consensus_agreement_rate = Some(
                                consensus_agreement_rate_from_events(&out.verification_events),
                            );

                            // Publish TieredExitEvent if TEE gate fired.
                            if let Some(tee_evt) = controller.take_tee_event() {
                                if let Some(ref nats) = input.nats {
                                    if let Err(e) = nats
                                        .publish_event(&task_id, &H2AIEvent::TieredExit(tee_evt))
                                        .await
                                    {
                                        tracing::warn!(
                                            target: "h2ai.engine",
                                            task_id = %task_id,
                                            "failed to publish TieredExitEvent: {e}"
                                        );
                                    }
                                }
                            }

                            // ── GAP-H3: Publish BudgetExhaustedEvent ─────────────────────────────
                            if controller.take_budget_exhausted() {
                                tracing::info!(
                                    target: "h2ai.engine",
                                    task_id = %task_id,
                                    tokens_used = controller.tokens_used(),
                                    "CostGuard: budget exhausted; returning best available output"
                                );
                                if let Some(ref nats) = input.nats {
                                    if let Err(e) = nats
                                        .publish_event(
                                            &task_id,
                                            &H2AIEvent::BudgetExhausted(BudgetExhaustedEvent {
                                                task_id: task_id.clone(),
                                                tokens_used: controller.tokens_used(),
                                                budget_tokens: input
                                                    .cfg
                                                    .cost_guard
                                                    .budget_tokens_per_task,
                                                timestamp: chrono::Utc::now(),
                                            }),
                                        )
                                        .await
                                    {
                                        tracing::warn!(
                                            target: "h2ai.engine",
                                            task_id = %task_id,
                                            "failed to publish BudgetExhaustedEvent: {e}"
                                        );
                                    }
                                }
                            }

                            // ── GAP-H3: Publish ConvergenceGateEvent ─────────────────────────────
                            if let Some(cge_evt) = controller.take_convergence_event() {
                                tracing::info!(
                                    target: "h2ai.engine",
                                    task_id = %task_id,
                                    wave = cge_evt.wave,
                                    n_live = cge_evt.n_live,
                                    "ConvergenceGate: proposals converged; accepting output"
                                );
                                if let Some(ref nats) = input.nats {
                                    if let Err(e) = nats
                                        .publish_event(
                                            &task_id,
                                            &H2AIEvent::ConvergenceGate(cge_evt),
                                        )
                                        .await
                                    {
                                        tracing::warn!(
                                            target: "h2ai.engine",
                                            task_id = %task_id,
                                            "failed to publish ConvergenceGateEvent: {e}"
                                        );
                                    }
                                }
                            }

                            return Ok(out);
                        }
                        crate::mape_k::MapeKDecision::Retry => continue,
                        crate::mape_k::MapeKDecision::Fail(e) => {
                            input.store.mark_failed(&task_id);
                            // Clean up signal consumer on engine failure.
                            if let Some(ref nats) = input.nats {
                                if signal_sub.is_some() {
                                    if let Err(ce) =
                                        nats.delete_signal_consumer(&input.task_id).await
                                    {
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
                        crate::mape_k::MapeKDecision::SpecAmbiguous {
                            constraint_id,
                            check_index,
                            divergent_reasons,
                            ..
                        } => {
                            // Signal the outer restart loop to attempt spec repair.
                            spec_ambiguous_signal =
                                Some((constraint_id, check_index, divergent_reasons));
                            break 'wave;
                        }
                        crate::mape_k::MapeKDecision::ComplexityOverflow {
                            probe_score,
                            rationale,
                            graft_first,
                        } => {
                            // Task 6: complexity ceiling reached.  Either route to
                            // the post-loop synthesis/grafting wave (`graft_first = true`)
                            // or fail terminally (`graft_first = false` → HITL surface).
                            tracing::warn!(
                                target: "h2ai.engine",
                                task_id = %task_id,
                                probe_score,
                                %rationale,
                                graft_first,
                                "complexity overflow — breaking retry loop"
                            );
                            if let Some(ref nats) = input.nats {
                                let ev = h2ai_types::events::H2AIEvent::ComplexityCeilingDetected(
                                    h2ai_types::events::ComplexityCeilingDetectedEvent {
                                        task_id: task_id.clone(),
                                        retry_count,
                                        entropy: 0.0,
                                        retry_slope: 0.0,
                                        n_eff_cg_product: 0.0,
                                        signals_fired: if probe_score == 0 { 2 } else { 1 },
                                        timestamp: chrono::Utc::now(),
                                    },
                                );
                                let _ = nats.publish_event(&input.task_id, &ev).await;
                            }
                            if graft_first {
                                complexity_overflow_graft_signal = true;
                                break 'wave;
                            } else {
                                input.store.mark_failed(&task_id);
                                return Err(EngineError::MaxRetriesExhausted {
                                    partial_verification_events: controller
                                        .take_verification_events(),
                                    best_partial_text: controller
                                        .global_best_proposal
                                        .as_ref()
                                        .map(|(_, t)| t.clone()),
                                });
                            }
                        }
                    }
                } // end 'wave retry loop

                // Return the signal and the controller together so the outer scope can
                // run post-loop logic (synthesis wave, error return) with the controller
                // after the pipeline borrow on `input` has been released.
                (
                    spec_ambiguous_signal,
                    complexity_overflow_graft_signal,
                    controller,
                )
            }; // ── end inner pipeline-borrow scope ─────────────────────────────────

            // ── spec repair handler ───────────────────────────────────────
            // `input` is no longer borrowed by `pipeline` here; `constraint_corpus`
            // can be mutated if the repair succeeds.
            if let Some((constraint_id, check_index, divergent_reasons)) = spec_ambiguous_signal {
                // Guard: cap restarts at 1 and respect auto_repair_enabled flag.
                if spec_ambiguous_restarts >= 1 || !input.cfg.gap_k1.auto_repair_enabled {
                    tracing::warn!(
                        target: "h2ai.engine",
                        task_id = %task_id,
                        constraint_id = %constraint_id,
                        spec_ambiguous_restarts,
                        auto_repair_enabled = input.cfg.gap_k1.auto_repair_enabled,
                        "SpecAmbiguous: repair limit reached or disabled; failing task"
                    );
                    return Err(EngineError::MaxRetriesExhausted {
                        partial_verification_events: controller.take_verification_events(),
                        best_partial_text: None,
                    });
                }

                // Select repair adapter: prefer researcher, fall back to first explorer.
                let repair_adapter: Option<&dyn h2ai_types::adapter::IComputeAdapter> = input
                    .researcher_adapter
                    .as_deref()
                    .or_else(|| input.explorer_adapters.first().copied());

                let Some(repair_adapter) = repair_adapter else {
                    tracing::warn!(
                        target: "h2ai.engine",
                        task_id = %task_id,
                        "SpecAmbiguous: no adapter available for repair; failing task"
                    );
                    return Err(EngineError::MaxRetriesExhausted {
                        partial_verification_events: controller.take_verification_events(),
                        best_partial_text: None,
                    });
                };

                // Find the affected ConstraintDoc to extract check text and current version.
                let maybe_doc = input
                    .constraint_corpus
                    .iter()
                    .find(|d| d.id == constraint_id)
                    .cloned();

                let Some(doc) = maybe_doc else {
                    tracing::warn!(
                        target: "h2ai.engine",
                        task_id = %task_id,
                        constraint_id = %constraint_id,
                        "SpecAmbiguous: constraint not found in corpus; failing task"
                    );
                    return Err(EngineError::MaxRetriesExhausted {
                        partial_verification_events: controller.take_verification_events(),
                        best_partial_text: None,
                    });
                };

                let original_check_text = doc
                    .binary_checks
                    .get(check_index)
                    .cloned()
                    .unwrap_or_default();

                // Build a minimal SemanticSpec from the ConstraintDoc so NatsVersionedSource
                // can serve as the CAS-aware version store for the repair write.
                let spec_for_repair = h2ai_constraints::spec::SemanticSpec {
                    id: doc.id.clone(),
                    title: doc.description.clone(),
                    source_file: doc.source_file.clone(),
                    severity: doc.severity.clone(),
                    domains: doc.domains.clone(),
                    mandatory_for_tags: doc.mandatory_for_tags.clone(),
                    related_to: doc.related_to.clone(),
                    remediation_hint: doc.remediation_hint.clone(),
                    exclusions: vec![],
                    requirements: vec![],
                    orderings: vec![],
                    rubric: h2ai_constraints::spec::QualityRubric {
                        pass: doc.description.clone(),
                        partial: None,
                        fail: String::new(),
                        checks: doc.binary_checks.clone(),
                        failure_modes: vec![],
                        negative_examples: vec![],
                        positive_examples: vec![],
                    },
                    version: doc.version,
                    repair_provenance: doc.repair_provenance.clone(),
                };

                let inner_source = h2ai_constraints::source::InMemorySource {
                    specs: input
                        .constraint_corpus
                        .iter()
                        .map(|d| {
                            if d.id == constraint_id {
                                spec_for_repair.clone()
                            } else {
                                h2ai_constraints::spec::SemanticSpec {
                                    id: d.id.clone(),
                                    title: d.description.clone(),
                                    source_file: d.source_file.clone(),
                                    severity: d.severity.clone(),
                                    domains: d.domains.clone(),
                                    mandatory_for_tags: d.mandatory_for_tags.clone(),
                                    related_to: d.related_to.clone(),
                                    remediation_hint: d.remediation_hint.clone(),
                                    exclusions: vec![],
                                    requirements: vec![],
                                    orderings: vec![],
                                    rubric: h2ai_constraints::spec::QualityRubric {
                                        pass: d.description.clone(),
                                        partial: None,
                                        fail: String::new(),
                                        checks: d.binary_checks.clone(),
                                        failure_modes: vec![],
                                        negative_examples: vec![],
                                        positive_examples: vec![],
                                    },
                                    version: d.version,
                                    repair_provenance: d.repair_provenance.clone(),
                                }
                            }
                        })
                        .collect(),
                };

                let versioned_source = std::sync::Arc::new(
                    h2ai_constraints::nats_versioned::NatsVersionedSource::new_in_memory(
                        inner_source,
                    ),
                );

                let repair_input = h2ai_autonomic::spec_repair::RepairInput {
                    task_id: task_id.to_string(),
                    constraint_id: constraint_id.clone(),
                    check_index,
                    original_check_text,
                    divergent_reasons,
                    should_pass_example: doc.description.clone(),
                    should_prune_example: None,
                    current_version: doc.version,
                };

                let advisor =
                    h2ai_autonomic::spec_repair::SpecRepairAdvisor::new(input.cfg.gap_k1.clone());
                let outcome = advisor
                    .run(repair_input, versioned_source.clone(), repair_adapter)
                    .await;

                match outcome {
                    h2ai_autonomic::spec_repair::RepairOutcome::Repaired { new_version } => {
                        tracing::info!(
                            target: "h2ai.engine",
                            task_id = %task_id,
                            constraint_id = %constraint_id,
                            new_version,
                            "spec repair succeeded; reloading corpus and restarting"
                        );
                        // Reload updated SemanticSpec list from the versioned source and
                        // recompile each spec back into a ConstraintDoc.
                        use h2ai_constraints::source::ConstraintSource as _;
                        match versioned_source.load_all() {
                            Ok(updated_specs) => {
                                input.constraint_corpus = updated_specs
                                    .into_iter()
                                    .map(|s| s.into_constraint_doc())
                                    .collect();
                            }
                            Err(e) => {
                                tracing::warn!(
                                    target: "h2ai.engine",
                                    task_id = %task_id,
                                    error = %e,
                                    "versioned source reload failed after repair; keeping old corpus"
                                );
                            }
                        }
                        spec_ambiguous_restarts += 1;
                        continue 'restart;
                    }
                    h2ai_autonomic::spec_repair::RepairOutcome::Failed { best_score } => {
                        tracing::warn!(
                            target: "h2ai.engine",
                            task_id = %task_id,
                            constraint_id = %constraint_id,
                            best_score,
                            "spec repair failed; returning MaxRetriesExhausted"
                        );
                        return Err(EngineError::MaxRetriesExhausted {
                            partial_verification_events: controller.take_verification_events(),
                            best_partial_text: None,
                        });
                    }
                }
            }
            // ── End handler ────────────────────────────────────────────────

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

            // ── Synthesis Wave (terminal, never retries) ───────────────────
            // Also runs when a ComplexityOverflow with `graft_first = true`
            // signalled mid-loop, even if `synthesis_wave_enabled` is otherwise off.
            if input.cfg.synthesis_wave_enabled || complexity_overflow_graft_signal {
                let all_checks: Vec<String> = input
                    .constraint_corpus
                    .iter()
                    .flat_map(|d| d.binary_checks.iter().cloned())
                    .collect();
                let check_offsets: Vec<(String, usize, usize)> = {
                    let mut offsets = Vec::new();
                    let mut start = 0usize;
                    for doc in input.constraint_corpus.iter() {
                        let count = doc.binary_checks.len();
                        if count > 0 {
                            offsets.push((doc.id.clone(), start, count));
                            start += count;
                        }
                    }
                    offsets
                };
                if !all_checks.is_empty() {
                    let partial_passes = h2ai_autonomic::repair::select_orthogonal_partials(
                        controller.all_pruned(),
                        &all_checks,
                        &check_offsets,
                        3,
                        h2ai_autonomic::repair::partial_max_chars(
                            input.cfg.model_max_tokens,
                            3,
                            input.cfg.partial_pass_overhead_factor,
                        ),
                    );
                    if !partial_passes.is_empty() {
                        let synth_adapter = input
                            .synthesis_adapter
                            .or_else(|| input.explorer_adapters.first().copied());
                        if let Some(adapter) = synth_adapter {
                            use h2ai_types::adapter::ComputeRequest;
                            use h2ai_types::sizing::TauValue;
                            let tau = TauValue::new(input.cfg.synthesis_tau)
                                .unwrap_or_else(|_| TauValue::new(0.2).unwrap());

                            // Sort descending by score; seed from highest-scoring partial.
                            let mut sorted_partials = partial_passes.clone();
                            sorted_partials.sort_by(|a, b| {
                                b.score
                                    .partial_cmp(&a.score)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });

                            // Determine which synthesis path to take.
                            let final_output: Option<String> = if input
                                .cfg
                                .sequential_grafting_enabled
                                && sorted_partials.len() > 1
                            {
                                // ── Sequential Constraint Grafting ──────────────
                                let system_ctx = controller.system_context_with_rubric();
                                let mut base_text = sorted_partials[0].proposal_text.clone();
                                let mut base_partial = sorted_partials[0].clone();
                                let mut base_score = base_partial.score;
                                let max_rounds = input.cfg.sequential_grafting_max_rounds.max(1);
                                let mut rounds_used = 0usize;
                                // Track constraint IDs introduced by completed graft rounds
                                // to detect circular dependencies in the grafting sequence.
                                let mut grafted_ids: std::collections::HashSet<String> =
                                    std::collections::HashSet::new();

                                for candidate in sorted_partials.iter().skip(1) {
                                    if rounds_used >= max_rounds.saturating_sub(1) {
                                        break;
                                    }
                                    let missing = h2ai_autonomic::repair::missing_constraint_ids(
                                        &base_partial,
                                        candidate,
                                        &check_offsets,
                                    );
                                    if missing.is_empty() {
                                        continue;
                                    }
                                    // ── Over-decomposition guards ──────────────────────
                                    // 1. Redundancy: skip if candidate overlaps base by > 60%.
                                    if h2ai_autonomic::repair::graft_is_redundant(
                                        &base_partial,
                                        candidate,
                                        0.6,
                                    ) {
                                        tracing::debug!(
                                            target: "h2ai.engine",
                                            task_id = %task_id,
                                            "graft candidate skipped — shared constraint ratio > 0.6"
                                        );
                                        continue;
                                    }
                                    // 2. Cycle: skip if all missing IDs were already grafted.
                                    if h2ai_autonomic::repair::grafted_ids_cycle_detected(
                                        &missing,
                                        &grafted_ids,
                                    ) {
                                        tracing::debug!(
                                            target: "h2ai.engine",
                                            task_id = %task_id,
                                            "graft candidate skipped — circular dependency detected"
                                        );
                                        continue;
                                    }
                                    // 3. Token projection: skip if merged size > 130% base.
                                    if h2ai_autonomic::repair::graft_token_projection_exceeds(
                                        &base_text,
                                        &candidate.proposal_text,
                                        1.3,
                                    ) {
                                        tracing::debug!(
                                            target: "h2ai.engine",
                                            task_id = %task_id,
                                            "graft candidate skipped — token projection > 130% base"
                                        );
                                        continue;
                                    }
                                    let graft_ctx = h2ai_autonomic::repair::build_graft_context(
                                        &h2ai_autonomic::repair::GraftInput {
                                            base_text: &base_text,
                                            candidate_text: &candidate.proposal_text,
                                            constraint_ids: &missing,
                                            system_context: system_ctx,
                                        },
                                    );
                                    let req = ComputeRequest {
                                        system_context: graft_ctx,
                                        task: input.manifest.description.clone(),
                                        tau,
                                        max_tokens: input.cfg.synthesis_max_tokens,
                                    };
                                    let graft_resp = match adapter.execute(req).await {
                                        Ok(resp) => resp,
                                        Err(e) => {
                                            tracing::warn!(
                                                target: "h2ai.engine",
                                                task_id = %task_id,
                                                error = %e,
                                                "graft round LLM call failed; stopping early"
                                            );
                                            break;
                                        }
                                    };

                                    // ── Intermediate verification + rollback ───────────
                                    let graft_proposal = h2ai_types::events::ProposalEvent {
                                        task_id: task_id.clone(),
                                        explorer_id: h2ai_types::identity::ExplorerId::new(),
                                        tau,
                                        generation: u64::MAX,
                                        raw_output: graft_resp.output.clone(),
                                        token_cost: graft_resp.token_cost,
                                        adapter_kind: graft_resp.adapter_kind.clone(),
                                        timestamp: chrono::Utc::now(),
                                    };
                                    let verif = crate::verification::VerificationPhase::run(
                                        crate::verification::VerificationInput {
                                            proposals: vec![graft_proposal],
                                            constraint_corpus: &input.constraint_corpus,
                                            evaluator: input
                                                .explorer_adapters
                                                .first()
                                                .copied()
                                                .unwrap_or(adapter),
                                            config: controller.params().verification_config,
                                            eval_cache: crate::verification::new_eval_cache(),
                                            consensus_passes: 1,
                                        },
                                    )
                                    .await;

                                    let new_score: f64 = if let Some((_, compliance, _)) =
                                        verif.passed.first()
                                    {
                                        if compliance.is_empty() {
                                            1.0
                                        } else {
                                            compliance.iter().map(|r| r.score).sum::<f64>()
                                                / compliance.len() as f64
                                        }
                                    } else if let Some((_, compliance, _, _)) = verif.failed.first()
                                    {
                                        if compliance.is_empty() {
                                            0.0
                                        } else {
                                            compliance.iter().map(|r| r.score).sum::<f64>()
                                                / compliance.len() as f64
                                        }
                                    } else {
                                        0.0
                                    };

                                    if new_score >= base_score {
                                        base_text = graft_resp.output;
                                        base_score = new_score;
                                        for (idx, text, passed) in &candidate.check_results {
                                            if *passed {
                                                if let Some(entry) = base_partial
                                                    .check_results
                                                    .iter_mut()
                                                    .find(|(i, _, _)| i == idx)
                                                {
                                                    *entry = (*idx, text.clone(), true);
                                                }
                                            }
                                        }
                                        rounds_used += 1;
                                        grafted_ids.extend(missing.iter().cloned());
                                        tracing::debug!(
                                            target: "h2ai.engine",
                                            task_id = %task_id,
                                            round = rounds_used,
                                            new_score,
                                            "graft round accepted"
                                        );
                                    } else {
                                        tracing::warn!(
                                            target: "h2ai.engine",
                                            task_id = %task_id,
                                            new_score,
                                            base_score,
                                            "graft was destructive — rolling back"
                                        );
                                    }
                                }
                                Some(base_text)
                            } else {
                                // ── Single-shot synthesis (stable fallback) ────────────
                                let synthesis_ctx = h2ai_autonomic::repair::build_synthesis_context(
                                    h2ai_autonomic::repair::SynthesisInput {
                                        partial_passes: &sorted_partials,
                                        checks: &all_checks,
                                        system_context_with_rubric: controller
                                            .system_context_with_rubric(),
                                    },
                                );
                                let req = ComputeRequest {
                                    system_context: synthesis_ctx,
                                    task: input.manifest.description.clone(),
                                    tau,
                                    max_tokens: input.cfg.synthesis_max_tokens,
                                };
                                match adapter.execute(req).await {
                                    Ok(resp) => Some(resp.output),
                                    Err(_) => None,
                                }
                            };

                            // ── Terminal verification of final output ──────────────────
                            if let Some(output_text) = final_output {
                                let synth_proposal = h2ai_types::events::ProposalEvent {
                                    task_id: task_id.clone(),
                                    explorer_id: h2ai_types::identity::ExplorerId::new(),
                                    tau,
                                    generation: u64::MAX,
                                    raw_output: output_text.clone(),
                                    token_cost: 0,
                                    adapter_kind: adapter.kind().clone(),
                                    timestamp: chrono::Utc::now(),
                                };
                                let verif_out = crate::verification::VerificationPhase::run(
                                    crate::verification::VerificationInput {
                                        proposals: vec![synth_proposal.clone()],
                                        constraint_corpus: &input.constraint_corpus,
                                        evaluator: input
                                            .explorer_adapters
                                            .first()
                                            .copied()
                                            .unwrap_or(adapter),
                                        config: controller.params().verification_config,
                                        eval_cache: crate::verification::new_eval_cache(),
                                        consensus_passes: 1,
                                    },
                                )
                                .await;
                                if !verif_out.passed.is_empty() {
                                    tracing::info!(
                                        target: "h2ai.engine",
                                        task_id = %task_id,
                                        grafting = input.cfg.sequential_grafting_enabled,
                                        "synthesis wave succeeded"
                                    );
                                    use crate::mape_k::MergeOutput;
                                    use h2ai_types::events::SelectionResolvedEvent;
                                    use h2ai_types::sizing::{MergeStrategy, PredictionBasis};
                                    let explorer_id = synth_proposal.explorer_id.clone();
                                    let merge_out = MergeOutput {
                                        task_id: task_id.clone(),
                                        resolved_output: output_text,
                                        selection_resolved: true,
                                        selection_resolved_event: SelectionResolvedEvent {
                                            task_id: task_id.clone(),
                                            valid_proposals: vec![explorer_id.clone()],
                                            pruned_proposals: vec![],
                                            merge_strategy: MergeStrategy::ScoreOrdered,
                                            timestamp: chrono::Utc::now(),
                                            merge_elapsed_secs: None,
                                            n_input_proposals: 1,
                                            n_failed_proposals: 0,
                                        },
                                        attribution: crate::attribution::HarnessAttribution {
                                            baseline_quality: 0.0,
                                            topology_gain: 0.0,
                                            verification_gain: 0.0,
                                            tao_gain: 0.0,
                                            q_confidence: 1.0,
                                            prediction_basis: PredictionBasis::Heuristic,
                                            q_measured: None,
                                            rho_adjusted: 0.0,
                                            case_b_flag: false,
                                            synthesis_gain: 0.0,
                                        },
                                        attribution_interval: None,
                                        talagrand: None,
                                        suggested_next_params: None,
                                        waste_ratio: 0.0,
                                        applied_optimizations: vec![],
                                        epistemic_yield: None,
                                        frontier_event: None,
                                        adapter_correctness: vec![(explorer_id, true)],
                                        coherence_state: crate::coherence::CoherenceState::default(
                                        ),
                                        comparison_events: vec![],
                                        oracle_gate_passed: None,
                                        tau_values: vec![input.cfg.synthesis_tau],
                                        iteration_verification_events: vec![],
                                        wave_token_cost: 0,
                                        pairwise_cosine_mean: None,
                                    };
                                    input.store.mark_resolved(&task_id);
                                    let mut out = controller.finalize(merge_out);
                                    out.consensus_agreement_rate =
                                        Some(consensus_agreement_rate_from_events(
                                            &out.verification_events,
                                        ));
                                    return Ok(out);
                                }
                            }
                        }
                        // Synthesis failed or scored < 1.0: extract global best partial for HITL.
                        let best_partial_text = controller
                            .all_pruned()
                            .iter()
                            .filter_map(|e| {
                                h2ai_autonomic::repair::partial_pass_from_event(
                                    e,
                                    &all_checks,
                                    &check_offsets,
                                    h2ai_autonomic::repair::partial_max_chars(
                                        input.cfg.model_max_tokens,
                                        1,
                                        input.cfg.partial_pass_overhead_factor,
                                    ),
                                )
                            })
                            .max_by(|a, b| {
                                a.score
                                    .partial_cmp(&b.score)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|p| p.proposal_text);
                        return Err(EngineError::MaxRetriesExhausted {
                            partial_verification_events: controller.take_verification_events(),
                            best_partial_text,
                        });
                    }
                } else if complexity_overflow_graft_signal {
                    // Layer 2 in mape_k.rs should prevent this path from ever being reached
                    // (corpus_synthesis_viable=false blocks graft_first=true routing).
                    // If we land here, a future refactor has bypassed the guard — surface immediately.
                    tracing::error!(
                        target: "h2ai.engine",
                        task_id = %task_id,
                        "BUG: complexity_overflow_graft_signal is set but corpus has no \
                         binary_checks — Layer 2 guard in mape_k::handle_exit_reason was bypassed"
                    );
                }
            }

            return Err(EngineError::MaxRetriesExhausted {
                partial_verification_events: controller.take_verification_events(),
                best_partial_text: None,
            });
        } // end 'restart: loop
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

        if phase == Some(crate::task_store::TaskPhase::Merging) {
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
                consensus_agreement_rate: None,
            })
        } else {
            // Earlier phase or unknown stage — restart from scratch
            Self::run_offline(input).await
        }
    }
}

#[cfg(test)]
mod tiered_exit_engine_tests {
    use h2ai_config::{H2AIConfig, TieredExitConfig};

    fn tee_n_for_wave_standalone(cfg: &H2AIConfig, retry_count: u32) -> u32 {
        let tee = &cfg.tiered_exit;
        if retry_count == 0 {
            tee.min_n
        } else {
            tee.n_for_wave(retry_count, cfg.max_autonomic_retries)
        }
    }

    fn make_tee_cfg(min_n: u32, max_n: u32, max_retries: u32) -> H2AIConfig {
        H2AIConfig {
            tiered_exit: TieredExitConfig {
                enabled: true,
                min_n,
                max_n,
                ..TieredExitConfig::default()
            },
            max_autonomic_retries: max_retries,
            ..H2AIConfig::default()
        }
    }

    #[test]
    fn escalation_wave0_uses_min_n() {
        let cfg = make_tee_cfg(2, 6, 4);
        assert_eq!(tee_n_for_wave_standalone(&cfg, 0), 2);
    }

    #[test]
    fn escalation_wave4_uses_max_n() {
        let cfg = make_tee_cfg(2, 6, 4);
        assert_eq!(tee_n_for_wave_standalone(&cfg, 4), 6);
    }
}

#[cfg(test)]
mod beyond_budget_injection_tests {
    use super::BEYOND_BUDGET_VERIFIER_ADDENDUM;
    use h2ai_autonomic::complexity_probe::ComplexityProbeResult;
    use h2ai_config::ComplexityRoutingConfig;
    use h2ai_types::config::VerificationConfig;

    fn cfg_with_decompose_enabled(decompose_threshold: u8) -> ComplexityRoutingConfig {
        ComplexityRoutingConfig {
            enabled: true,
            verifier_decomposition_enabled: true,
            decompose_threshold,
            ..ComplexityRoutingConfig::default()
        }
    }

    fn inject(
        cfg: &ComplexityRoutingConfig,
        probe: &ComplexityProbeResult,
        vconfig: &mut VerificationConfig,
    ) {
        if cfg.verifier_decomposition_enabled && probe.complexity >= cfg.decompose_threshold {
            vconfig
                .evaluator_system_prompt
                .push_str(BEYOND_BUDGET_VERIFIER_ADDENDUM);
        }
    }

    #[test]
    fn addendum_appended_when_complexity_meets_threshold() {
        let cfg = cfg_with_decompose_enabled(4);
        let probe = ComplexityProbeResult {
            complexity: 4,
            rationale: "complex".into(),
            decompose_recommended: true,
        };
        let mut vconfig = VerificationConfig::default();
        let original_len = vconfig.evaluator_system_prompt.len();
        inject(&cfg, &probe, &mut vconfig);
        assert!(
            vconfig.evaluator_system_prompt.len() > original_len,
            "addendum must be appended when complexity >= threshold"
        );
        assert!(
            vconfig.evaluator_system_prompt.contains("BEYOND_BUDGET"),
            "appended text must contain BEYOND_BUDGET label"
        );
    }

    #[test]
    fn addendum_not_appended_when_complexity_below_threshold() {
        let cfg = cfg_with_decompose_enabled(4);
        let probe = ComplexityProbeResult {
            complexity: 3,
            rationale: "simple".into(),
            decompose_recommended: false,
        };
        let mut vconfig = VerificationConfig::default();
        let original = vconfig.evaluator_system_prompt.clone();
        inject(&cfg, &probe, &mut vconfig);
        assert_eq!(
            vconfig.evaluator_system_prompt, original,
            "prompt must not change when complexity < threshold"
        );
    }

    #[test]
    fn addendum_not_appended_when_verifier_decomposition_disabled() {
        let cfg = ComplexityRoutingConfig {
            enabled: true,
            verifier_decomposition_enabled: false,
            decompose_threshold: 4,
            ..ComplexityRoutingConfig::default()
        };
        let probe = ComplexityProbeResult {
            complexity: 5,
            rationale: "very complex".into(),
            decompose_recommended: true,
        };
        let mut vconfig = VerificationConfig::default();
        let original = vconfig.evaluator_system_prompt.clone();
        inject(&cfg, &probe, &mut vconfig);
        assert_eq!(
            vconfig.evaluator_system_prompt, original,
            "prompt must not change when verifier_decomposition_enabled = false"
        );
    }
}

#[cfg(test)]
mod cost_guard_engine_tests {
    use h2ai_types::events::{
        BudgetExhaustedEvent, ConvergenceGateEvent, CostThresholdWarningEvent, H2AIEvent,
    };

    #[test]
    fn cost_guard_event_variants_exist() {
        // Compile-time check: these variants must exist in H2AIEvent
        let _: fn(CostThresholdWarningEvent) -> H2AIEvent = H2AIEvent::CostThresholdWarning;
        let _: fn(BudgetExhaustedEvent) -> H2AIEvent = H2AIEvent::BudgetExhausted;
        let _: fn(ConvergenceGateEvent) -> H2AIEvent = H2AIEvent::ConvergenceGate;
    }
}
