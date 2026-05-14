use crate::complexity::assess_task_complexity;
use crate::diagnostics::TalagrandDiagnostic;
pub use crate::nats_dispatch_adapter::NatsDispatchConfig;
use crate::self_optimizer::{OptimizerParams, QualityMeasurement, SelfOptimizer, SuggestInput};
use crate::task_store::{TaskPhase, TaskState, TaskStore};
use crate::verification::extract_json_object;
use chrono::Utc;
use futures::future::join_all;
use h2ai_autonomic::checker::MultiplicationChecker;
use h2ai_autonomic::merger::{MergeEngine, MergeOutcome};
use h2ai_autonomic::planner::{ProvisionInput, TopologyPlanner};
use h2ai_autonomic::retry::{RetryAction, RetryPolicy};
use h2ai_config::{FamilyConstraint, H2AIConfig};
use h2ai_constraints::types::ConstraintDoc;
use h2ai_context::compaction::{compact, CompactionConfig};
use h2ai_context::compiler;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_state::semilattice::ProposalSet;
use h2ai_types::adapter::{AdapterRegistry, ComputeRequest, IComputeAdapter};
use h2ai_types::config::{
    AgentRole, AuditorConfig, RoleSpec, TaoConfig, TopologyKind, VerificationConfig,
};
use h2ai_types::events::{
    BranchPrunedEvent, CalibrationCompletedEvent, GenerationPhaseCompletedEvent,
    OracleGateResultEvent, ProposalEvent, ProposalFailedEvent, ProposalFailureReason,
    SelectionResolvedEvent, TaskBootstrappedEvent, TaskComplexityAssessedEvent,
    VerificationScoredEvent, ZeroSurvivalEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::TaskManifest;
use h2ai_types::sizing::TaskQuadrant;
use h2ai_types::sizing::{
    MergeStrategy, MultiplicationConditionFailure, PredictionBasis, RoleErrorCost, TauValue,
};
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
}

#[derive(serde::Deserialize)]
struct AuditResponse {
    approved: bool,
    reason: String,
    /// Constraint IDs the auditor identified as violated. Populated by the adversarial
    /// prompt template; empty when the auditor approves or uses a legacy prompt.
    #[serde(default)]
    violated: Vec<String>,
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
        input
            .store
            .insert(task_id.clone(), TaskState::new(task_id.clone()));

        // ── Phase 1: Bootstrap ──────────────────────────────────────────────
        let description = &input.manifest.description;
        // include_rubric=false: first-attempt context withholds LlmJudge rubric — explorer
        // reasons from domain expertise and the constraint requirement summary injected by
        // the compiler. The verifier retains the full rubric via ConstraintPredicate::LlmJudge.
        //
        // include_rubric=true: retry context exposes the full rubric so the model has all
        // information needed to address specific failing checks after the first attempt failed.
        let compiled = compiler::compile(description, &input.constraint_corpus, false);
        let compiled_with_rubric = compiler::compile(description, &input.constraint_corpus, true);

        let adr_keywords: Vec<String> = input
            .constraint_corpus
            .iter()
            .flat_map(|d: &ConstraintDoc| d.vocabulary().into_iter())
            .chain(input.manifest.constraints.iter().cloned())
            .collect();
        let compaction_cfg = CompactionConfig {
            max_tokens: input.cfg.max_context_tokens.unwrap_or(usize::MAX / 4),
            preserve_keywords: adr_keywords,
        };
        let system_context = compact(&compiled.system_context, &compaction_cfg);
        let system_context_with_rubric =
            compact(&compiled_with_rubric.system_context, &compaction_cfg);

        let _bootstrapped = TaskBootstrappedEvent {
            task_id: task_id.clone(),
            system_context: system_context.clone(),
            pareto_weights: input.manifest.pareto_weights.clone(),
            timestamp: Utc::now(),
        };

        let explorer_adapter_kind = input
            .explorer_adapters
            .first()
            .map(|a| a.kind().clone())
            .unwrap_or_else(|| input.auditor_config.adapter.clone());

        // ── Verifier/Explorer Family Conflict Gate ──────────────────────────
        // Enforced once before the MAPE-K loop: no retry can resolve a deployment
        // topology where the verification adapter shares a provider family with the
        // explorer pool. Such a configuration invalidates Condorcet independence —
        // the verifier cannot detect its own blind spots.
        //
        // Bypassed when `family_constraint = Disabled` (no check at all).
        // With `SingleFamilyOk` the gate is skipped but a WARN is emitted at startup.
        // With `RequireDiverse` the task fails when explorer/verifier share a family.
        if input.calibration.explorer_verification_family_match {
            match input.cfg.safety.family_constraint {
                FamilyConstraint::RequireDiverse => {
                    use h2ai_types::adapter::AdapterFamily;
                    let explorer_family = AdapterFamily::from(&explorer_adapter_kind).to_string();
                    let verifier_family =
                        AdapterFamily::from(input.verification_adapter.kind()).to_string();
                    input.store.mark_failed(&task_id);
                    return Err(EngineError::MultiplicationConditionFailed(
                        MultiplicationConditionFailure::VerifierExplorerFamilyConflict {
                            explorer_family,
                            verifier_family,
                        }
                        .to_string(),
                    ));
                }
                FamilyConstraint::SingleFamilyOk => {
                    tracing::warn!(
                        "single-family adapter pool: correlated hallucination protection degraded"
                    );
                }
                FamilyConstraint::Disabled => {}
            }
        }

        // ── Phase 1.5: Task Complexity Assessment ──────────────────────────
        // Classifies the task into a routing quadrant (Precision / Coverage /
        // Complex / Degenerate) using corpus metadata and calibration state.
        // In shadow_mode this is purely observational — the quadrant does not
        // change topology selection downstream. shadow_mode = false is the
        // production default (armed after GAP-A1; see reference.toml).
        let probe_adapter = input.explorer_adapters.first().copied();
        let complexity_assessment = assess_task_complexity(
            &input.constraint_corpus,
            &input.calibration,
            &input.cfg.task_complexity,
            task_id.clone(),
            probe_adapter.map(|a| (a as &dyn IComputeAdapter, system_context.as_str())),
        )
        .await;
        let assessed_quadrant = complexity_assessment.task_quadrant;
        // Store the assessment for NATS publishing after the engine returns.
        let complexity_event_for_output = complexity_assessment.clone();

        // Degenerate guard (non-shadow mode only): both TCC and pool N_eff are below
        // their thresholds. The pool cannot explore the solution space for this task;
        // fail immediately rather than wasting MAPE-K retries.
        if !input.cfg.task_complexity.shadow_mode && assessed_quadrant == TaskQuadrant::Degenerate {
            input.store.mark_failed(&task_id);
            return Err(EngineError::MultiplicationConditionFailed(
                MultiplicationConditionFailure::InsufficientPoolDiversity {
                    n_eff: input
                        .calibration
                        .eigen
                        .as_ref()
                        .map(|e| e.n_effective)
                        .unwrap_or(0.0),
                    threshold: input.cfg.task_complexity.n_eff_complex_threshold,
                }
                .to_string(),
            ));
        }

        let cg_mean = input.calibration.coefficients.cg_mean();
        let n_max_ceiling = input.calibration.coefficients.n_max().floor() as u32;

        // ── MAPE-K retry state ───────────────────────────────────────────────
        // When EnsembleCalibration is present use n_optimal (Condorcet-derived) as the
        // default ensemble size instead of the manifest count. n_max_ceiling (Amdahl) is
        // the hard ceiling regardless.
        // Calibration may suggest n_optimal > manifest.count (calibration uses max_n=9).
        // Treat manifest count as an explicit upper bound: the submitter chose it deliberately.
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
        let mut current_params = OptimizerParams {
            n_agents: initial_n_agents,
            max_turns: input.tao_config.max_turns as u32,
            verify_threshold: input.verification_config.threshold,
        };
        let mut tao_config = input.tao_config.clone();
        let mut verification_config = input.verification_config.clone();
        let mut force_topology: Option<TopologyKind> = None;
        let mut tried_topologies: Vec<TopologyKind> = Vec::new();
        let mut tau_reduction_factor: f64 = 1.0;
        // τ-spread expansion factor driven by Talagrand U-curve detection.
        // Starts at 1.0; increases by 20% per over-confident iteration, capped at tau_spread_max_factor.
        let mut tau_spread_factor: f64 = 1.0;
        let mut all_pruned: Vec<BranchPrunedEvent> = Vec::new();
        let mut tau_values_tried: Vec<Vec<f64>> = Vec::new();
        let mut quality_history: Vec<QualityMeasurement> = Vec::new();
        let mut all_verification_events: Vec<VerificationScoredEvent> = Vec::new();
        let mut all_failed_proposals: Vec<ProposalFailedEvent> = Vec::new();
        let mut all_comparison_events: Vec<h2ai_types::events::VerifierComparisonEvent> =
            Vec::new();
        let mut all_shadow_audit_events: Vec<h2ai_types::events::ShadowAuditorResultEvent> =
            Vec::new();
        let mut all_correlated_warnings: Vec<h2ai_types::events::CorrelatedEnsembleWarning> =
            Vec::new();
        let mut all_researcher_grounding_events: Vec<h2ai_types::events::ResearcherGroundingEvent> =
            Vec::new();
        let mut all_srani_events: Vec<h2ai_types::events::CorrelatedFabricationEvent> = Vec::new();
        let mut srani_ema_updated: f64 = input.srani_ema_cfi;
        let mut srani_count_updated: usize = input.srani_count;
        let mut last_multiplication_failure: Option<MultiplicationConditionFailure> = None;
        let task_eval_cache = crate::verification::new_eval_cache();
        let mut adapter_rotation_offset: usize = 0;
        let mut pending_tombstone: Option<String> = None;
        let mut retry_context: Option<String> = None;
        let mut srani_tier: usize = 0;
        let mut srani_last_wave_fired: bool = false;
        let mut topology_retry_events: Vec<h2ai_types::events::TopologyProvisionedEvent> =
            Vec::new();
        let mut mode_collapse_count: usize = 0;
        let task_deadline = input
            .cfg
            .task_deadline_secs
            .map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));

        // ── GAP-C3: Domain Coverage Pre-check ──────────────────────────────────
        // Slot domain assignments don't change between retries, so check once here.
        // Fires DiversityGuardDegradedEvent when coverage < domain_coverage_threshold.
        // Fails the task immediately when require_bivariate_cg = true.
        let diversity_degraded_event_for_output: Option<
            h2ai_types::events::DiversityGuardDegradedEvent,
        > = {
            let corpus_tags = crate::domain_coverage::corpus_domain_tags(&input.constraint_corpus);
            let coverage_score = crate::domain_coverage::compute_coverage_score(
                &input.manifest.explorers.slot_configs,
                &corpus_tags,
            );
            if coverage_score < input.cfg.domain_coverage_threshold {
                let slot_domains: Vec<String> = input
                    .manifest
                    .explorers
                    .slot_configs
                    .iter()
                    .flat_map(|s| s.constraint_domains.iter().cloned())
                    .collect();
                if input.cfg.safety.require_bivariate_cg {
                    input.store.mark_failed(&task_id);
                    return Err(EngineError::MultiplicationConditionFailed(format!(
                        "domain_coverage {coverage_score:.2} < threshold {:.2} (require_bivariate_cg=true)",
                        input.cfg.domain_coverage_threshold
                    )));
                }
                Some(h2ai_types::events::DiversityGuardDegradedEvent {
                    task_id: task_id.clone(),
                    reason: format!(
                        "slot domain coverage {coverage_score:.2} below threshold {:.2}",
                        input.cfg.domain_coverage_threshold
                    ),
                    coverage_score,
                    slot_domains,
                })
            } else {
                None
            }
        };

        for retry_count in 0..=input.cfg.max_autonomic_retries {
            if let Some(dl) = task_deadline {
                if std::time::Instant::now() >= dl {
                    input.store.mark_failed(&task_id);
                    return Err(EngineError::DeadlineExceeded {
                        budget_secs: input.cfg.task_deadline_secs.unwrap_or(0),
                    });
                }
            }
            // ── Phase 2: Topology Provisioning ─────────────────────────────
            // Phase 1.5 Precision override (non-shadow): use within-family τ-spread
            // with 2–3 slots and the calibration_tau_spread bounds. The explorer_adapter
            // is already within-family (single AdapterKind); τ diversity decorrelates
            // the samples structurally without crossing family boundaries.
            // TODO(gap-a1-multi-family): when multiple calibrated families are available,
            // select the family with the highest EnsembleCalibration::p_mean here instead
            // of the single explorer_adapter family. Requires per-family p_mean tracking
            // in CalibrationCompletedEvent (not yet implemented).
            let precision_active = !input.cfg.task_complexity.shadow_mode
                && assessed_quadrant == TaskQuadrant::Precision;
            let role_specs: Vec<RoleSpec> = if input.manifest.explorers.roles.is_empty() {
                let count = if precision_active {
                    // 2–3 slots: more than 1 provides synthesis benefit; cap at 3 to stay
                    // within the Self-MoA budget where within-family wins.
                    (n_max_ceiling as usize).clamp(2, input.cfg.precision_mode_max_slots) as u32
                } else {
                    current_params.n_agents.max(1)
                };
                let (tau_min_manifest, tau_max_manifest) = if precision_active {
                    // Use calibration τ-spread bounds for Precision — not the manifest
                    // values, which are set for multi-family Coverage tasks.
                    let s = input.cfg.calibration_tau_spread;
                    (s[0].clamp(0.05, 0.95), s[1].clamp(0.05, 0.95))
                } else {
                    (
                        input.manifest.explorers.tau_min.unwrap_or(0.2),
                        input.manifest.explorers.tau_max.unwrap_or(0.9),
                    )
                };
                // Apply τ-spread expansion (Talagrand U-curve feedback) around the manifest centre.
                let tau_center = (tau_max_manifest + tau_min_manifest) / 2.0;
                let half_spread = (tau_max_manifest - tau_min_manifest) / 2.0;
                let max_half = tau_center.min(1.0 - tau_center); // can't exceed [0,1]
                let expanded_half = (half_spread * tau_spread_factor).min(max_half);
                let tau_min = tau_center - expanded_half;
                let tau_max = tau_center + expanded_half;
                let step = if count > 1 {
                    (tau_max - tau_min) / (count - 1) as f64
                } else {
                    0.0
                };
                (0..count)
                    .map(|i| RoleSpec {
                        agent_id: format!("exp_{}", (b'A' + (i % 26) as u8) as char),
                        role: AgentRole::Executor,
                        tau: Some(
                            TauValue::new(
                                ((tau_min + step * i as f64) * tau_reduction_factor)
                                    .clamp(0.05, 0.95),
                            )
                            .unwrap_or_else(|_| TauValue::new(0.05).unwrap()),
                        ),
                        role_error_cost: None,
                    })
                    .collect()
            } else {
                input.manifest.explorers.roles.clone()
            };

            input
                .store
                .set_phase(&task_id, TaskPhase::Provisioning, 0, retry_count);

            // In shadow_mode: quadrant is observational only — pass None to preserve
            // current topology selection. When armed (shadow_mode=false), pass the
            // assessed quadrant so TopologyPlanner can apply self-MoA for Precision tasks.
            let effective_quadrant = if input.cfg.task_complexity.shadow_mode {
                None
            } else {
                Some(assessed_quadrant)
            };

            let (mut provisioned, _cg_collapse) = TopologyPlanner::provision(ProvisionInput {
                task_id: task_id.clone(),
                cc: &input.calibration.coefficients,
                pareto_weights: &input.manifest.pareto_weights,
                role_specs: &role_specs,
                review_gates: input.manifest.explorers.review_gates.clone(),
                auditor_config: input.auditor_config.clone(),
                explorer_adapter: explorer_adapter_kind.clone(),
                force_topology: force_topology.clone(),
                retry_count,
                cfg: input.cfg,
                eigen: input.calibration.eigen.as_ref(),
                task_quadrant: effective_quadrant,
            });
            provisioned.constraint_tombstone = pending_tombstone.clone();
            if retry_count > 0 {
                topology_retry_events.push(provisioned.clone());
            }

            tried_topologies.push(provisioned.topology_kind.clone());
            let explorer_count = provisioned.explorer_configs.len() as u32;
            current_params.n_agents = explorer_count;

            // Guard: OutlierResistant requires n ≥ 2f+3. Fail early rather than silently falling back.
            if let MergeStrategy::OutlierResistant { f }
            | MergeStrategy::MultiOutlierResistant { f, .. } = &provisioned.merge_strategy
            {
                let f = *f;
                let n = provisioned.explorer_configs.len();
                let required = MergeStrategy::min_krum_quorum(f);
                if n < required {
                    return Err(EngineError::InsufficientQuorum { n, f, required });
                }
            }

            // ── Phase 2.5: Multiplication Condition Gate ───────────────────
            input.store.set_phase(
                &task_id,
                TaskPhase::MultiplicationCheck,
                explorer_count,
                retry_count,
            );

            // Derive p_mean, rho_mean, and prediction_basis from EnsembleCalibration when available.
            // Fallback proxies when calibration is absent (Heuristic basis):
            //   p = 0.5 + CG_mean / 2  (accuracy proxy from output similarity)
            //   ρ = 1 - CG_mean        (correlation proxy from output similarity)
            let (p_mean, rho_mean, attribution_basis) = match &input.calibration.ensemble {
                Some(ec) => (ec.p_mean, ec.rho_mean, ec.prediction_basis),
                None => (
                    0.5 + cg_mean / 2.0,
                    (1.0 - cg_mean).clamp(0.0, 1.0),
                    PredictionBasis::Heuristic,
                ),
            };
            let baseline_competence = p_mean;
            let error_correlation = rho_mean;

            tracing::info!(
                target: "h2ai.engine",
                p_mean,
                rho_mean,
                cg_mean = input.calibration.coefficients.cg_mean(),
                theta_coord = input.calibration.coordination_threshold.value(),
                retry_count,
                "multiplication check"
            );
            if let Err(mc_event) = MultiplicationChecker::check(
                &task_id,
                &input.calibration.coefficients,
                &input.calibration.coordination_threshold,
                baseline_competence,
                error_correlation,
                retry_count,
                input.cfg,
            ) {
                tracing::warn!(
                    target: "h2ai.engine",
                    failure = ?mc_event.failure,
                    "multiplication condition failed"
                );
                last_multiplication_failure = Some(mc_event.failure.clone());
                let tau_values: Vec<f64> = provisioned
                    .explorer_configs
                    .iter()
                    .map(|ec| ec.tau.value())
                    .collect();
                tau_values_tried.push(tau_values);

                let zero_event = ZeroSurvivalEvent {
                    task_id: task_id.clone(),
                    retry_count,
                    timestamp: Utc::now(),
                    n_eff_cosine_actual: None,
                    failure_mode: None,
                };
                match RetryPolicy::decide(
                    &zero_event,
                    &tried_topologies,
                    all_pruned.clone(),
                    tau_values_tried.clone(),
                    last_multiplication_failure.clone(),
                ) {
                    RetryAction::Retry(next_topology) => {
                        force_topology = Some(next_topology);
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::RetryWithTauReduction {
                        topology,
                        tau_factor,
                    } => {
                        force_topology = Some(topology);
                        tau_reduction_factor *= tau_factor;
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::RetryWithHints { topology, hints } => {
                        force_topology = Some(topology);
                        if !hints.is_empty() {
                            let attempts_remaining =
                                input.cfg.max_autonomic_retries.saturating_sub(retry_count);
                            let hint_lines = hints
                                .iter()
                                .map(|h| format!("• {h}"))
                                .collect::<Vec<_>>()
                                .join("\n\n");
                            retry_context = Some(format!(
                                "{system_context_with_rubric}\n\n--- CONSTRAINT FEEDBACK (iteration {retry_count}) ---\n\
                                The following constraints were violated. Fix ALL of these in your next response:\n\n\
                                {hint_lines}\n\n\
                                {attempts_remaining} retry attempt(s) remaining.\n\
                                ---"
                            ));
                        }
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::Fail(reason) => {
                        tracing::warn!(
                            target: "h2ai.engine",
                            task_id = %task_id,
                            retry_count,
                            reason = ?reason,
                            last_multiplication_failure = ?last_multiplication_failure,
                            "retry policy decided Fail — giving up"
                        );
                        input.store.mark_failed(&task_id);
                        return Err(EngineError::MaxRetriesExhausted {
                            partial_verification_events: all_verification_events.clone(),
                        });
                    }
                }
            }

            // ── Phase 2.6: Pool Diversity Guard ────────────────────────────────
            if input.cfg.safety.diversity_threshold > 0.0 {
                let n_eff_prior = input.calibration.n_eff_cosine_prior;
                let threshold = 1.0 + input.cfg.safety.diversity_threshold;
                if n_eff_prior > 0.0 && n_eff_prior < threshold {
                    last_multiplication_failure = Some(
                        h2ai_types::sizing::MultiplicationConditionFailure::InsufficientPoolDiversity {
                            n_eff: n_eff_prior,
                            threshold: input.cfg.safety.diversity_threshold,
                        },
                    );
                    let tau_values: Vec<f64> = provisioned
                        .explorer_configs
                        .iter()
                        .map(|ec| ec.tau.value())
                        .collect();
                    tau_values_tried.push(tau_values);
                    let zero_event = ZeroSurvivalEvent {
                        task_id: task_id.clone(),
                        retry_count,
                        timestamp: Utc::now(),
                        n_eff_cosine_actual: Some(n_eff_prior),
                        failure_mode: Some(h2ai_types::events::FailureMode::ModeCollapse),
                    };
                    match RetryPolicy::decide(
                        &zero_event,
                        &tried_topologies,
                        all_pruned.clone(),
                        tau_values_tried.clone(),
                        last_multiplication_failure.clone(),
                    ) {
                        RetryAction::Retry(next_topology) => {
                            force_topology = Some(next_topology);
                            continue;
                        }
                        RetryAction::RetryWithTauReduction {
                            topology,
                            tau_factor,
                        } => {
                            force_topology = Some(topology);
                            tau_reduction_factor *= tau_factor;
                            continue;
                        }
                        RetryAction::RetryWithHints { topology, hints } => {
                            force_topology = Some(topology);
                            if !hints.is_empty() {
                                let attempts_remaining =
                                    input.cfg.max_autonomic_retries.saturating_sub(retry_count);
                                let hint_lines = hints
                                    .iter()
                                    .map(|h| format!("• {h}"))
                                    .collect::<Vec<_>>()
                                    .join("\n\n");
                                retry_context = Some(format!(
                                    "{system_context_with_rubric}\n\n--- CONSTRAINT FEEDBACK (iteration {retry_count}) ---\n\
                                    The following constraints were violated. Fix ALL of these in your next response:\n\n\
                                    {hint_lines}\n\n\
                                    {attempts_remaining} retry attempt(s) remaining.\n\
                                    ---"
                                ));
                            }
                            continue;
                        }
                        RetryAction::Fail(reason) => {
                            tracing::warn!(
                                target: "h2ai.engine",
                                task_id = %task_id,
                                retry_count,
                                reason = ?reason,
                                "retry policy decided Fail (diversity check) — giving up"
                            );
                            input.store.mark_failed(&task_id);
                            return Err(EngineError::MaxRetriesExhausted {
                                partial_verification_events: all_verification_events.clone(),
                            });
                        }
                    }
                }
            }
            // ───────────────────────────────────────────────────────────────────

            // ── Phase 3: Parallel Generation ───────────────────────────────
            input.store.set_phase(
                &task_id,
                TaskPhase::ParallelGeneration,
                explorer_count,
                retry_count,
            );

            use crate::nats_dispatch_adapter::NatsDispatchAdapter;
            use std::future::Future;
            use std::pin::Pin;
            use std::sync::Arc;
            type ExplorerFuture<'f> = Pin<
                Box<
                    dyn Future<
                            Output = Result<
                                (ProposalEvent, u8, Option<String>),
                                ProposalFailedEvent,
                            >,
                        > + Send
                        + 'f,
                >,
            >;
            // slot_configs are always populated by Phase 0 decomposition before EngineInput
            // is constructed. No fallback needed here.
            let effective_slot_configs: &[h2ai_types::manifest::ExplorerSlotConfig] =
                &input.manifest.explorers.slot_configs;

            let active_ctx: String = retry_context
                .as_deref()
                .unwrap_or(&system_context)
                .to_owned();

            // ── Proactive Researcher Pre-pass (GAP-C1 proactive path) ──────────
            // For slots with search_enabled=true, call the researcher adapter to fetch
            // current state-of-the-art grounding before generating proposals.
            let mut slot_groundings: Vec<Option<String>> =
                vec![None; provisioned.explorer_configs.len()];
            if let Some(ref researcher) = input.researcher_adapter {
                for idx in 0..provisioned.explorer_configs.len() {
                    let sc_opt = if effective_slot_configs.is_empty() {
                        None
                    } else {
                        Some(&effective_slot_configs[idx % effective_slot_configs.len()])
                    };
                    if sc_opt.map(|sc| sc.search_enabled).unwrap_or(false) {
                        let req = ComputeRequest {
                            system_context: active_ctx.clone(),
                            task: format!(
                                "Search for current state-of-the-art evidence relevant to: {}. \
                                 Return a concise grounding statement in 2-3 sentences that \
                                 the explorer should treat as established fact.",
                                input.manifest.description
                            ),
                            tau: TauValue::new(0.2).unwrap(),
                            max_tokens: 512,
                        };
                        if let Ok(resp) = researcher.execute(req).await {
                            all_researcher_grounding_events.push(
                                h2ai_types::events::ResearcherGroundingEvent {
                                    task_id: task_id.clone(),
                                    shared_assumption: String::new(),
                                    literature_summary: resp.output.clone(),
                                    slot: Some(format!("slot_{idx}")),
                                    source: h2ai_types::events::GroundingSource::LlmResearcher,
                                },
                            );
                            slot_groundings[idx] =
                                Some(format!("[STATE-OF-THE-ART]: {}", resp.output));
                        }
                    }
                }
            }
            // ─────────────────────────────────────────────────────────────────

            let futures_vec: Vec<ExplorerFuture<'_>> = provisioned
                .explorer_configs
                .iter()
                .enumerate()
                .map(|(idx, explorer_cfg)| {
                    let (slot_task, slot_system_ctx) = {
                        let configs = effective_slot_configs;
                        if configs.is_empty() {
                            (input.manifest.description.clone(), active_ctx.clone())
                        } else {
                            let sc = &configs[idx % configs.len()];
                            let cot = sc.cot_style.instruction();
                            let task = if cot.is_empty() {
                                input.manifest.description.clone()
                            } else {
                                format!("{}\n\n{}", cot, input.manifest.description)
                            };
                            let mut preamble = String::new();
                            if !sc.role_frame.is_empty() {
                                preamble.push_str(&sc.role_frame);
                            }
                            if !sc.focus_mandate.is_empty() {
                                if !preamble.is_empty() {
                                    preamble.push_str("\n\n");
                                }
                                preamble.push_str("[MANDATE]: ");
                                preamble.push_str(&sc.focus_mandate);
                            }
                            if !sc.rejection_criteria.is_empty() {
                                if !preamble.is_empty() {
                                    preamble.push_str("\n\n");
                                }
                                preamble.push_str(
                                    "[AFTER WRITING YOUR PROPOSAL, IDENTIFY THE BIGGEST RISK]: ",
                                );
                                preamble.push_str(&sc.rejection_criteria);
                            }
                            let base_ctx = if preamble.is_empty() {
                                active_ctx.clone()
                            } else {
                                format!("{}\n\n{}", preamble, active_ctx)
                            };
                            let ctx = if let Some(grounding) =
                                slot_groundings.get(idx).and_then(|g| g.as_ref())
                            {
                                format!("{}\n\n{}", grounding, base_ctx)
                            } else {
                                base_ctx
                            };
                            (task, ctx)
                        }
                    };
                    let effective_ctx = if let Some(ref tombstone) = pending_tombstone {
                        format!("{}\n\n{}", slot_system_ctx, tombstone)
                    } else {
                        slot_system_ctx
                    };
                    let req = ComputeRequest {
                        system_context: effective_ctx,
                        task: slot_task,
                        tau: explorer_cfg.tau,
                        max_tokens: input.cfg.explorer_max_tokens,
                    };
                    let explorer_id = explorer_cfg.explorer_id.clone();
                    let task_id_clone = task_id.clone();
                    let tao_cfg = tao_config.clone();
                    if let Some(ref nd_cfg) = input.nats_dispatch {
                        let arc = Arc::new(NatsDispatchAdapter::new(NatsDispatchConfig {
                            nats: nd_cfg.nats.clone(),
                            provider: nd_cfg.provider.clone(),
                            agent_descriptor: nd_cfg.agent_descriptor.clone(),
                            task_requirements: nd_cfg.task_requirements.clone(),
                            task_timeout: nd_cfg.task_timeout,
                            payload_store: nd_cfg.payload_store.clone(),
                            offload_threshold_bytes: nd_cfg.offload_threshold_bytes,
                        }));
                        let generation = retry_count as u64;
                        let fut: ExplorerFuture<'_> = Box::pin(async move {
                            use crate::tao_loop::{TaoInput, TaoLoop};
                            match TaoLoop::run(TaoInput {
                                task_id: task_id_clone.clone(),
                                explorer_id: explorer_id.clone(),
                                adapter: arc.as_ref(),
                                initial_request: req,
                                config: tao_cfg,
                                schema_config: None,
                                generation,
                            })
                            .await
                            {
                                Ok(tao_proposal) => Ok((
                                    tao_proposal.event,
                                    tao_proposal.tao_turns,
                                    tao_proposal.turn1_output,
                                )),
                                Err(e) => Err(ProposalFailedEvent {
                                    task_id: task_id_clone,
                                    explorer_id,
                                    reason: ProposalFailureReason::AdapterError(e.to_string()),
                                    timestamp: Utc::now(),
                                }),
                            }
                        });
                        fut
                    } else {
                        let pool_len = input.explorer_adapters.len();
                        let adapter_idx = (idx + adapter_rotation_offset) % pool_len;
                        let adapter = input.explorer_adapters[adapter_idx];
                        let generation = retry_count as u64;
                        let fut: ExplorerFuture<'_> = Box::pin(async move {
                            use crate::tao_loop::{TaoInput, TaoLoop};
                            match TaoLoop::run(TaoInput {
                                task_id: task_id_clone.clone(),
                                explorer_id: explorer_id.clone(),
                                adapter,
                                initial_request: req,
                                config: tao_cfg,
                                schema_config: None,
                                generation,
                            })
                            .await
                            {
                                Ok(tao_proposal) => Ok((
                                    tao_proposal.event,
                                    tao_proposal.tao_turns,
                                    tao_proposal.turn1_output,
                                )),
                                Err(e) => Err(ProposalFailedEvent {
                                    task_id: task_id_clone,
                                    explorer_id,
                                    reason: ProposalFailureReason::AdapterError(e.to_string()),
                                    timestamp: Utc::now(),
                                }),
                            }
                        });
                        fut
                    }
                })
                .collect();

            let results = join_all(futures_vec).await;

            let mut proposals: Vec<ProposalEvent> = Vec::new();
            let mut tao_turns_collected: Vec<u8> = Vec::new();
            let mut failed_proposals: Vec<ProposalFailedEvent> = Vec::new();
            let mut turn1_map: std::collections::HashMap<h2ai_types::identity::ExplorerId, String> =
                std::collections::HashMap::new();

            for result in results {
                match result {
                    Ok((proposal, turns, turn1_output)) => {
                        input.store.increment_completed(&task_id);
                        tao_turns_collected.push(turns);
                        if let Some(t1) = turn1_output {
                            turn1_map.insert(proposal.explorer_id.clone(), t1);
                        }
                        proposals.push(proposal);
                    }
                    Err(failed) => {
                        input.store.increment_completed(&task_id);
                        failed_proposals.push(failed);
                    }
                }
            }
            let failed_count = failed_proposals.len() as u32;
            all_failed_proposals.append(&mut failed_proposals);

            // Capture raw texts for epistemic yield / FailureMode classification.
            let all_raw_texts_this_wave: Vec<String> =
                proposals.iter().map(|p| p.raw_output.clone()).collect();

            let tao_turns_mean = if tao_turns_collected.is_empty() {
                1.0
            } else {
                tao_turns_collected.iter().map(|&t| t as f64).sum::<f64>()
                    / tao_turns_collected.len() as f64
            };

            let _gen_completed = GenerationPhaseCompletedEvent {
                task_id: task_id.clone(),
                total_explorers: explorer_count,
                successful: proposals.len() as u32,
                failed: failed_count,
                timestamp: Utc::now(),
            };

            // Collect tau values for this batch before verification
            let tau_values: Vec<f64> = provisioned
                .explorer_configs
                .iter()
                .map(|ec| ec.tau.value())
                .collect();

            // ── GAP-C1: Correlated Hallucination Detection ──────────────────
            // Check CV of pairwise Jaccard distances on raw proposal texts.
            // Low CV = proposals are semantically clustered → retry with grounding hint.
            if input.cfg.correlated_hallucination_cv_threshold > 0.0 && proposals.len() >= 2 {
                let proposal_texts: Vec<&str> =
                    proposals.iter().map(|p| p.raw_output.as_str()).collect();
                if let Some(signal) = crate::correlated_hallucination::compute_cv(&proposal_texts) {
                    if signal.cv < input.cfg.correlated_hallucination_cv_threshold
                        && signal.mean_jaccard_distance
                            < input.cfg.correlated_hallucination_min_jaccard_floor
                        && retry_count < input.cfg.max_autonomic_retries
                    {
                        all_correlated_warnings.push(
                            h2ai_types::events::CorrelatedEnsembleWarning {
                                task_id: task_id.clone(),
                                cv: signal.cv,
                                mean_jaccard_distance: signal.mean_jaccard_distance,
                                retry_count,
                            },
                        );

                        // Build grounding: call researcher (reactive path) if available
                        let grounding_hint = if let Some(ref researcher) = input.researcher_adapter
                        {
                            let proposal_summary = proposals
                                .iter()
                                // 300-char cap: debug context only — enough for correlation diagnosis
                                .map(|p| p.raw_output[..p.raw_output.len().min(300)].to_string())
                                .collect::<Vec<_>>()
                                .join("\n---\n");
                            let research_req = ComputeRequest {
                                system_context: system_context.clone(),
                                task: format!(
                                    "These AI proposals may share a common assumption.\
                                     \nPROPOSALS:\n{proposal_summary}\n\n\
                                     Search for current state-of-the-art evidence that \
                                     contradicts the shared assumption. Return JSON: \
                                     {{\"shared_assumption\": \"...\", \
                                       \"literature_summary\": \"...\", \
                                       \"grounding_statement\": \"...\"}}",
                                ),
                                tau: TauValue::new(0.3).unwrap(),
                                max_tokens: 1024,
                            };
                            match researcher.execute(research_req).await {
                                Ok(resp) => {
                                    #[derive(serde::Deserialize)]
                                    struct ResearchResult {
                                        shared_assumption: String,
                                        literature_summary: String,
                                        grounding_statement: String,
                                    }
                                    crate::verification::extract_json_object::<ResearchResult>(
                                        &resp.output,
                                    )
                                    .map(|r| {
                                        all_researcher_grounding_events
                                            .push(h2ai_types::events::ResearcherGroundingEvent {
                                            task_id: task_id.clone(),
                                            shared_assumption: r.shared_assumption,
                                            literature_summary: r.literature_summary.clone(),
                                            slot: None,
                                            source:
                                                h2ai_types::events::GroundingSource::LlmResearcher,
                                        });
                                        format!(
                                            "[EXTERNAL GROUNDING]: {}\n\
                                             Find the assumption all current proposals share \
                                             and propose a solution that contradicts it.",
                                            r.grounding_statement
                                        )
                                    })
                                }
                                Err(_) => None,
                            }
                        } else {
                            None
                        };

                        let hint = grounding_hint.unwrap_or_else(|| {
                            "Find the assumption all current proposals share that might be wrong. \
                             Propose a solution that directly contradicts it."
                                .to_string()
                        });

                        retry_context = Some(format!(
                            "{system_context_with_rubric}\n\n\
                             --- CORRELATED ENSEMBLE DETECTED (iteration {retry_count}) ---\n\
                             {hint}\n\
                             ---"
                        ));

                        tau_values_tried.push(tau_values.clone());
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                }
            }
            // ─────────────────────────────────────────────────────────────────

            // ── SRANI: Specification-Relative Architectural Noun Intersection ──
            // Orthogonal to C1: fires when diverse proposals share a fabricated entity.
            if input.cfg.srani.enabled && proposals.len() >= 2 {
                let proposal_texts: Vec<&str> =
                    proposals.iter().map(|p| p.raw_output.as_str()).collect();
                let task_spec = &input.manifest.description;
                if let Some(grounding) =
                    crate::specification_grounding::check_specification_grounding(
                        task_spec,
                        &proposal_texts,
                    )
                {
                    // Always update EMA regardless of whether the gate fires.
                    let new_ema = crate::srani_gate::update_ema(
                        input.srani_ema_cfi,
                        grounding.cfi,
                        input.cfg.srani.ema_alpha,
                    );
                    srani_ema_updated = new_ema;
                    srani_count_updated = input.srani_count + 1;

                    let (pressure, hint_injected) = if input.cfg.srani.adaptive {
                        let mu = if input.srani_count < 5 {
                            input.cfg.srani.cold_start_midpoint()
                        } else {
                            input.srani_ema_cfi
                        };
                        let p = crate::srani_gate::compute_injection_pressure(
                            grounding.cfi,
                            mu,
                            input.cfg.srani.temperature,
                        );
                        (p, p >= input.cfg.srani.gate_threshold)
                    } else {
                        // Legacy static-threshold path (adaptive=false).
                        let injected = grounding.cfi > input.cfg.srani.inject_threshold;
                        let p = if grounding.cfi > input.cfg.srani.warn_threshold {
                            1.0
                        } else {
                            0.0
                        };
                        (p, injected)
                    };

                    // Warn floor: 0.20 for adaptive, warn_threshold crossing for static.
                    let should_emit = if input.cfg.srani.adaptive {
                        pressure >= 0.20
                    } else {
                        grounding.cfi > input.cfg.srani.warn_threshold
                    };

                    if should_emit {
                        all_srani_events.push(h2ai_types::events::CorrelatedFabricationEvent {
                            task_id: task_id.clone(),
                            cfi: grounding.cfi,
                            injection_pressure: pressure,
                            shared_ungrounded_entities: grounding.shared_ungrounded.clone(),
                            proposal_count: grounding.proposal_count,
                            hint_injected,
                            timestamp: Utc::now(),
                        });
                        if hint_injected {
                            let grounding_ctx = crate::srani_grounding::GroundingContext {
                                fabricated_entities: grounding.shared_ungrounded.clone(),
                                task_description: input.manifest.description.clone(),
                            };
                            let chain_result = if let Some(ref chain) = input.srani_grounding_chain
                            {
                                chain.resolve(&grounding_ctx, srani_tier).await
                            } else {
                                use crate::srani_grounding::GroundingProvider;
                                crate::srani_grounding::SpecAnchorGrounder
                                    .ground(&grounding_ctx)
                                    .await
                            };
                            if let Some(ref result) = chain_result {
                                let hint = crate::srani_grounding::format_grounding_hint(
                                    result,
                                    &grounding.shared_ungrounded,
                                );
                                retry_context = Some(retry_context.unwrap_or_default() + &hint);
                                all_researcher_grounding_events.push(
                                    h2ai_types::events::ResearcherGroundingEvent {
                                        task_id: task_id.clone(),
                                        shared_assumption: grounding.shared_ungrounded.join(", "),
                                        literature_summary: result.grounding_statement.clone(),
                                        slot: None,
                                        source: result.source.clone(),
                                    },
                                );
                            } else {
                                let entities = grounding.shared_ungrounded.join(", ");
                                retry_context = Some(
                                    retry_context.unwrap_or_default()
                                        + &format!(
                                            "\n\n--- GROUNDING CONTEXT ---\n\
                                             Avoid (not in spec): {entities}\n\
                                             Design using spec-defined components only.\n---"
                                        ),
                                );
                            }
                            if srani_last_wave_fired {
                                srani_tier = srani_tier.saturating_add(1);
                            }
                            srani_last_wave_fired = true;
                        } else {
                            srani_last_wave_fired = false;
                        }
                    }
                }
            }

            // ── Phase 3.5: Verification Loop (LLM-as-Judge) ──────────────
            use crate::verification::{VerificationInput, VerificationPhase};
            let mut pruned: Vec<BranchPrunedEvent> = Vec::new();
            let mut iteration_verification_events: Vec<VerificationScoredEvent> = Vec::new();
            let ver_out = VerificationPhase::run(VerificationInput {
                proposals,
                constraint_corpus: &input.constraint_corpus,
                evaluator: input.verification_adapter,
                config: verification_config.clone(),
                eval_cache: std::sync::Arc::clone(&task_eval_cache),
                consensus_passes: input.cfg.verifier_consensus_passes,
            })
            .await;
            all_comparison_events.extend(ver_out.comparison_events.iter().cloned());

            // Diversity gate: post-verification — check constraint-satisfaction profile entropy.
            // Collapsed fingerprints signal collective hallucination; trigger MAPE-K retry.
            if matches!(
                crate::diversity::DiversityGuard::check(
                    &ver_out.passed,
                    input.cfg.safety.diversity_threshold
                ),
                crate::diversity::DiversityResult::Collapsed
            ) {
                tau_values_tried.push(tau_values);
                let zero_event = ZeroSurvivalEvent {
                    task_id: task_id.clone(),
                    retry_count,
                    timestamp: Utc::now(),
                    n_eff_cosine_actual: None,
                    failure_mode: None,
                };
                match RetryPolicy::decide(
                    &zero_event,
                    &tried_topologies,
                    all_pruned.clone(),
                    tau_values_tried.clone(),
                    last_multiplication_failure.clone(),
                ) {
                    RetryAction::Retry(next_topology) => {
                        force_topology = Some(next_topology);
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::RetryWithTauReduction {
                        topology,
                        tau_factor,
                    } => {
                        force_topology = Some(topology);
                        tau_reduction_factor *= tau_factor;
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::RetryWithHints { topology, hints } => {
                        force_topology = Some(topology);
                        if !hints.is_empty() {
                            let attempts_remaining =
                                input.cfg.max_autonomic_retries.saturating_sub(retry_count);
                            let hint_lines = hints
                                .iter()
                                .map(|h| format!("• {h}"))
                                .collect::<Vec<_>>()
                                .join("\n\n");
                            retry_context = Some(format!(
                                "{system_context_with_rubric}\n\n--- CONSTRAINT FEEDBACK (iteration {retry_count}) ---\n\
                                The following constraints were violated. Fix ALL of these in your next response:\n\n\
                                {hint_lines}\n\n\
                                {attempts_remaining} retry attempt(s) remaining.\n\
                                ---"
                            ));
                        }
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::Fail(reason) => {
                        tracing::warn!(
                            target: "h2ai.engine",
                            task_id = %task_id,
                            retry_count,
                            reason = ?reason,
                            last_multiplication_failure = ?last_multiplication_failure,
                            "retry policy decided Fail (all-pruned) — giving up"
                        );
                        input.store.mark_failed(&task_id);
                        return Err(EngineError::MaxRetriesExhausted {
                            partial_verification_events: all_verification_events.clone(),
                        });
                    }
                }
            }

            let mut proposals: Vec<ProposalEvent> = Vec::new();
            for (prop, results, any_cache_hit) in ver_out.passed {
                let score = h2ai_constraints::types::aggregate_compliance_score(&results);
                iteration_verification_events.push(VerificationScoredEvent {
                    task_id: task_id.clone(),
                    explorer_id: prop.explorer_id.clone(),
                    score,
                    reason: String::new(),
                    passed: true,
                    cache_hit: any_cache_hit,
                    timestamp: Utc::now(),
                });
                input.store.record_validation(&task_id, true);
                proposals.push(prop);
            }
            for (prop, results, violations, any_cache_hit) in ver_out.failed {
                let hard_gate = results.iter().all(|r| r.hard_passes());
                let soft = h2ai_constraints::types::aggregate_compliance_score(&results);
                let compliance = if hard_gate { soft } else { 0.0 };
                let score = compliance;
                iteration_verification_events.push(VerificationScoredEvent {
                    task_id: task_id.clone(),
                    explorer_id: prop.explorer_id.clone(),
                    score,
                    reason: violations
                        .iter()
                        .map(|v| v.constraint_id.clone())
                        .collect::<Vec<_>>()
                        .join(", "),
                    passed: false,
                    cache_hit: any_cache_hit,
                    timestamp: Utc::now(),
                });
                let error_cost = RoleErrorCost::new((1.0 - compliance).clamp(0.0, 1.0)).unwrap();
                let cost = provisioned
                    .explorer_configs
                    .iter()
                    .position(|ec| ec.explorer_id == prop.explorer_id)
                    .and_then(|idx| provisioned.role_error_costs.get(idx))
                    .cloned()
                    .unwrap_or(error_cost);
                tracing::info!(
                    target: "h2ai.engine",
                    explorer_id = %prop.explorer_id,
                    compliance = compliance,
                    hard_gate = hard_gate,
                    violated = ?violations.iter().map(|v| &v.constraint_id).collect::<Vec<_>>(),
                    "proposal pruned"
                );
                pruned.push(BranchPrunedEvent {
                    task_id: task_id.clone(),
                    explorer_id: prop.explorer_id,
                    reason: format!("verification compliance {compliance:.2}"),
                    constraint_error_cost: cost,
                    violated_constraints: violations,
                    timestamp: Utc::now(),
                });
                input.store.record_validation(&task_id, false);
            }

            // Build turn-1 proposals for Option B estimator feed.
            // Only accepted (passed) proposals that ran multiple TAO turns.
            let turn1_proposals_for_scoring: Vec<ProposalEvent> = proposals
                .iter()
                .filter_map(|prop| {
                    turn1_map
                        .get(&prop.explorer_id)
                        .map(|t1_output| ProposalEvent {
                            raw_output: t1_output.clone(),
                            ..prop.clone()
                        })
                })
                .collect();

            // ── Phase 4: Auditor Gate ──────────────────────────────────────
            input.store.set_phase(
                &task_id,
                TaskPhase::AuditorGate,
                explorer_count,
                retry_count,
            );

            let mut proposal_set = ProposalSet::new();
            let mut synthesis_candidates: Vec<ProposalEvent> = Vec::new();

            // Shadow auditor setup — domain and vote mode for this task.
            let task_domain = input
                .manifest
                .constraint_tags
                .first()
                .cloned()
                .unwrap_or_else(|| "default".to_string());
            let majority_vote_active = input
                .shadow_audit_ctx
                .as_ref()
                .map(|ctx| ctx.promoted_domains.contains(&task_domain))
                .unwrap_or(false);
            let mut shadow_events_this_wave: Vec<h2ai_types::events::ShadowAuditorResultEvent> =
                Vec::new();

            // In single-family mode the auditor is the same model as the explorer.
            // Adversarial self-evaluation produces systematic rejection bias — skip the
            // audit phase entirely and let all proposals through to the verifier.
            // Mock adapters are excluded: tests use Mock for both and must still run the audit.
            use h2ai_types::adapter::AdapterFamily as AF;
            let auditor_fam = input.auditor_adapter.family();
            let explorer_fam = input
                .explorer_adapters
                .first()
                .map(|a| a.family())
                .unwrap_or(auditor_fam.clone());
            let skip_audit = auditor_fam != AF::Mock && auditor_fam == explorer_fam;
            if skip_audit {
                tracing::info!(
                    target: "h2ai.engine",
                    task_id = %task_id,
                    family = %input.auditor_adapter.family(),
                    "single-family mode: skipping auditor (same family as explorer)"
                );
            }

            for proposal in proposals {
                let (primary_approved, audit_reason, audit_violated, shadow_result_opt) =
                    if skip_audit {
                        (true, String::new(), vec![], None)
                    } else {
                        let audit_prompt = input
                            .auditor_config
                            .prompt_template
                            .replace("{constraints}", &input.manifest.constraints.join(", "))
                            .replace("{proposal}", &proposal.raw_output);

                        let audit_prompt_str = audit_prompt;
                        let make_req = || ComputeRequest {
                            system_context: input.auditor_config.system_prompt.clone(),
                            task: audit_prompt_str.clone(),
                            tau: input.auditor_config.tau,
                            max_tokens: input.auditor_config.max_tokens,
                        };

                        // Run shadow concurrently with primary when shadow ctx is present.
                        let (primary_result, shadow_opt) = match input.shadow_audit_ctx.as_ref() {
                            Some(ctx) => {
                                let (p, s) = tokio::join!(
                                    input.auditor_adapter.execute(make_req()),
                                    ctx.adapter.execute(make_req())
                                );
                                (p, Some(s))
                            }
                            None => (input.auditor_adapter.execute(make_req()).await, None),
                        };

                        let audit_result =
                            primary_result.map_err(|e| EngineError::Adapter(e.to_string()))?;

                        let (approved, reason, violated) = match extract_json_object::<AuditResponse>(
                            &audit_result.output,
                        ) {
                            Some(r) => (r.approved, r.reason, r.violated),
                            None => {
                                tracing::warn!(
                                    task_id = %task_id,
                                    output = %audit_result.output,
                                    "auditor returned non-JSON; failing safe (treating as rejected)"
                                );
                                (false, "auditor parse failure".to_string(), vec![])
                            }
                        };
                        (approved, reason, violated, shadow_opt)
                    };

                // Extract shadow decision (None if shadow errored or absent).
                let shadow_approved_opt: Option<bool> = shadow_result_opt.and_then(|sr| {
                    sr.ok()
                        .and_then(|r| extract_json_object::<AuditResponse>(&r.output))
                        .map(|a| a.approved)
                });

                // Pruning decision.
                let rejected = if majority_vote_active {
                    // Promoted domain: both must approve (AND vote).
                    // If shadow errored (None), fall back to primary-only — shadow error ≠ rejection.
                    let shadow_vote = shadow_approved_opt.unwrap_or(primary_approved);
                    !(primary_approved && shadow_vote)
                } else {
                    // Shadow mode or no shadow: primary always decides.
                    !primary_approved
                };

                // Collect shadow event when shadow ran successfully.
                if let (Some(shadow_approved), Some(ctx)) =
                    (shadow_approved_opt, input.shadow_audit_ctx.as_ref())
                {
                    shadow_events_this_wave.push(h2ai_types::events::ShadowAuditorResultEvent {
                        task_id: task_id.clone(),
                        explorer_id: proposal.explorer_id.clone(),
                        primary_approved,
                        shadow_approved,
                        disagreement: primary_approved != shadow_approved,
                        domain: task_domain.clone(),
                        primary_family: input.auditor_adapter.family().to_string(),
                        shadow_family: ctx.adapter.family().to_string(),
                        timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
                    });
                }

                if rejected {
                    let explorer_id = proposal.explorer_id.clone();
                    let cost = provisioned
                        .explorer_configs
                        .iter()
                        .position(|ec| ec.explorer_id == explorer_id)
                        .and_then(|idx| provisioned.role_error_costs.get(idx))
                        .cloned()
                        .unwrap_or_else(|| RoleErrorCost::new(0.5).unwrap());
                    pruned.push(BranchPrunedEvent {
                        task_id: task_id.clone(),
                        explorer_id,
                        reason: audit_reason,
                        constraint_error_cost: cost,
                        violated_constraints: audit_violated
                            .iter()
                            .map(|id| h2ai_types::events::ConstraintViolation {
                                constraint_id: id.clone(),
                                score: 0.0,
                                severity_label: "Hard".to_string(),
                                remediation_hint: None,
                            })
                            .collect(),
                        timestamp: Utc::now(),
                    });
                    input.store.record_validation(&task_id, false);
                } else {
                    input.store.record_validation(&task_id, true);
                    let ver_score = iteration_verification_events
                        .iter()
                        .find(|e| e.explorer_id == proposal.explorer_id)
                        .map(|e| e.score)
                        .unwrap_or(0.0);
                    synthesis_candidates.push(proposal.clone());
                    proposal_set.insert_scored(proposal, ver_score);
                }
            }
            all_shadow_audit_events.extend(shadow_events_this_wave);

            // ── Constraint Frontier (Phase 4.5) ────────────────────────────
            // Build satisfaction matrix from surviving proposals × Static-tier constraints.
            // Compute pareto_coverage = participation_ratio(matrix) as a scalar measure of
            // how well the ensemble covered the constraint Pareto frontier.
            let frontier_event: Option<h2ai_types::events::ConstraintFrontierEvent> = {
                let static_constraints: Vec<&ConstraintDoc> = input
                    .constraint_corpus
                    .iter()
                    .filter(|d| d.tier() == h2ai_constraints::types::ConstraintTier::Static)
                    .collect();
                if !synthesis_candidates.is_empty() && !static_constraints.is_empty() {
                    let constraint_ids: Vec<String> =
                        static_constraints.iter().map(|c| c.id.clone()).collect();
                    let explorer_ids: Vec<h2ai_types::identity::ExplorerId> = synthesis_candidates
                        .iter()
                        .map(|p| p.explorer_id.clone())
                        .collect();
                    let satisfaction_matrix: Vec<Vec<f64>> = synthesis_candidates
                        .iter()
                        .map(|proposal| {
                            static_constraints
                                .iter()
                                .map(|c| {
                                    h2ai_constraints::eval::eval_sync(
                                        &c.predicate,
                                        &proposal.raw_output,
                                    )
                                })
                                .collect()
                        })
                        .collect();
                    let pareto_coverage =
                        crate::complexity::participation_ratio(&satisfaction_matrix);
                    Some(h2ai_types::events::ConstraintFrontierEvent {
                        task_id: task_id.clone(),
                        satisfaction_matrix,
                        constraint_ids,
                        explorer_ids,
                        pareto_coverage,
                        timestamp: chrono::Utc::now(),
                    })
                } else {
                    None
                }
            };

            // Per-explorer correctness from this wave's verification — used for H1 ρ_actual.
            let adapter_correctness: Vec<(h2ai_types::identity::ExplorerId, bool)> =
                iteration_verification_events
                    .iter()
                    .map(|e| (e.explorer_id.clone(), e.passed))
                    .collect();

            // ── Oracle Gate (Phase 3→4 transition) ─────────────────────────
            let oracle_gate_passed_flag: Option<bool> = if input.cfg.oracle_gate.enabled {
                if let Some(nats) = &input.nats_raw {
                    let gate_payload = serde_json::json!({
                        "task_id": &input.task_id,
                        "phase": 3,
                    });
                    let payload_bytes = serde_json::to_vec(&gate_payload).unwrap_or_default();
                    let timeout =
                        std::time::Duration::from_secs(input.cfg.oracle_gate.timeout_secs);
                    match tokio::time::timeout(
                        timeout,
                        nats.request(input.cfg.oracle_gate.subject.clone(), payload_bytes.into()),
                    )
                    .await
                    {
                        Ok(Ok(response)) => {
                            match serde_json::from_slice::<OracleGateResultEvent>(&response.payload)
                            {
                                Ok(result) => Some(result.gate_passed),
                                Err(_) => Some(input.cfg.oracle_gate.on_timeout == "pass"),
                            }
                        }
                        _ => Some(input.cfg.oracle_gate.on_timeout == "pass"),
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // If oracle gate explicitly failed, abort before merge.
            if oracle_gate_passed_flag == Some(false) {
                input.store.mark_failed(&task_id);
                return Err(EngineError::MaxRetriesExhausted {
                    partial_verification_events: all_verification_events.clone(),
                });
            }

            // ── Phase 5: Merge ──────────────────────────────────────────────
            input
                .store
                .set_phase(&task_id, TaskPhase::Merging, explorer_count, retry_count);

            let total_evaluated = proposal_set.len() + pruned.len();
            let filter_ratio = if total_evaluated > 0 {
                proposal_set.len() as f64 / total_evaluated as f64
            } else {
                1.0
            };

            let (mut attribution, attribution_interval) = {
                use crate::attribution::{
                    bootstrap_interval, AttributionInput, HarnessAttribution,
                };
                // Compute per-iteration Talagrand from current verification scores for S7 ρ correction.
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
                    verification_filter_ratio: filter_ratio,
                    tao_turns_mean,
                    tao_per_turn_factor: input.tao_multiplier,
                    prediction_basis: attribution_basis,
                    talagrand_state: iter_talagrand_state,
                    eigen_calibration: input.calibration.eigen.clone(),
                };
                let attr = HarnessAttribution::compute(&attr_input);
                let interval = {
                    let cg_samples = &input.calibration.coefficients.cg_samples;
                    if cg_samples.len() >= 2 {
                        Some(bootstrap_interval(&attr_input, cg_samples, 1000))
                    } else {
                        None
                    }
                };
                (attr, interval)
            };

            // ── Option B: feed TaoMultiplierEstimator with turn-1 vs final scores ──────
            if !turn1_proposals_for_scoring.is_empty() {
                use crate::verification::VerificationPhase;
                let turn1_scores = VerificationPhase::score_proposals(
                    turn1_proposals_for_scoring,
                    input.verification_adapter,
                    &verification_config,
                    &input.constraint_corpus,
                )
                .await;

                let mut est = input.tao_estimator.write().await;
                for (t1_prop, t1_score) in &turn1_scores {
                    if let Some(final_ev) = iteration_verification_events
                        .iter()
                        .find(|e| e.explorer_id == t1_prop.explorer_id && e.passed)
                    {
                        est.update(*t1_score, final_ev.score);
                    }
                }
            }
            // ─────────────────────────────────────────────────────────────────────────────

            // Accumulate all_pruned before moving pruned into resolve
            all_pruned.extend(pruned.iter().cloned());
            tau_values_tried.push(tau_values);

            // Coherence state after this wave — uncovered domains (from pruned) plus active
            // contradictions (from surviving proposals' satisfaction matrix when available).
            // Computed once here; all success and failure paths reuse it.
            let wave_coherence = {
                let base = crate::coherence::CoherenceState::from_pruned(
                    &input.constraint_corpus,
                    &all_pruned,
                );
                if let Some(ref fe) = frontier_event {
                    base.with_contradictions(
                        &input.constraint_corpus,
                        &fe.explorer_ids,
                        &fe.satisfaction_matrix,
                        &fe.constraint_ids,
                    )
                } else {
                    base
                }
            };
            tracing::trace!(
                target: "h2ai.coherence",
                retry = retry_count,
                uncovered_domains = ?wave_coherence.uncovered_domains,
                active_contradictions = wave_coherence.active_contradictions.len(),
                is_closed = wave_coherence.is_closed(),
                "coherence state after wave"
            );

            // ── Phase 5a: Synthesis (optional, replaces selection on success) ───────
            // Complex quadrant forces synthesis regardless of synthesis_enabled flag —
            // the task geometry demands cross-proposal reconciliation.
            let synthesis_forced = !input.cfg.task_complexity.shadow_mode
                && assessed_quadrant == TaskQuadrant::Complex;
            // When coherence is closed, all surviving proposals agree on every constraint domain.
            // Synthesis would reconcile nothing. Even in the Complex quadrant, coherence closure
            // implies structural agreement — synthesis adds no epistemic signal.
            let synthesis_bypass = wave_coherence.is_closed();
            if synthesis_bypass && (input.cfg.synthesis_enabled || synthesis_forced) {
                tracing::debug!(
                    target: "h2ai.coherence",
                    retry = retry_count,
                    n_candidates = synthesis_candidates.len(),
                    "synthesis bypassed: coherence closed, proposals already agree"
                );
            }
            let synthesis_output: Option<(String, f64)> = if !synthesis_bypass
                && (input.cfg.synthesis_enabled || synthesis_forced)
                && synthesis_candidates.len() >= input.cfg.synthesis_min_proposals
            {
                if let Some(synth_adapter) = input.synthesis_adapter {
                    use crate::synthesis::{SynthesisInput as SynthInput, SynthesisPhase};

                    let constraint_list = input.manifest.constraints.join("\n");
                    let synth_input = SynthInput {
                        task_description: &input.manifest.description,
                        constraint_list: &constraint_list,
                        proposals: &synthesis_candidates,
                        adapter: synth_adapter,
                        cfg: input.cfg,
                    };

                    match SynthesisPhase::run(synth_input).await {
                        Ok(synth_out) => {
                            // Re-verify the synthesis output through the full VerificationPhase
                            use crate::verification::{VerificationInput, VerificationPhase};
                            let synth_proposal = ProposalEvent {
                                task_id: task_id.clone(),
                                explorer_id: h2ai_types::identity::ExplorerId::new(),
                                tau: TauValue::new(input.cfg.synthesis_tau)
                                    .unwrap_or_else(|_| TauValue::new(0.2).unwrap()),
                                generation: 0,
                                raw_output: synth_out.synthesis_text.clone(),
                                token_cost: synth_out.synthesis_tokens,
                                adapter_kind: h2ai_types::config::AdapterKind::CloudGeneric {
                                    endpoint: "synthesis".into(),
                                    api_key_env: "NONE".into(),
                                    model: None,
                                },
                                timestamp: Utc::now(),
                            };

                            let re_ver = VerificationPhase::run(VerificationInput {
                                proposals: vec![synth_proposal],
                                constraint_corpus: &input.constraint_corpus,
                                evaluator: input.verification_adapter,
                                config: verification_config.clone(),
                                eval_cache: std::sync::Arc::clone(&task_eval_cache),
                                consensus_passes: input.cfg.verifier_consensus_passes,
                            })
                            .await;
                            all_comparison_events.extend(re_ver.comparison_events.iter().cloned());

                            if !re_ver.passed.is_empty() {
                                // Compute synthesis_gain: Q(synthesis) - max(Q(individuals))
                                let indiv_scores = VerificationPhase::score_proposals(
                                    synthesis_candidates.clone(),
                                    input.verification_adapter,
                                    &verification_config,
                                    &input.constraint_corpus,
                                )
                                .await;
                                let max_indiv = indiv_scores
                                    .iter()
                                    .map(|(_, s)| *s)
                                    .fold(f64::NEG_INFINITY, f64::max)
                                    .max(0.0);
                                let synth_score = re_ver
                                    .passed
                                    .first()
                                    .map(|(_, results, _)| {
                                        h2ai_constraints::types::aggregate_compliance_score(results)
                                    })
                                    .unwrap_or(0.0);
                                let synthesis_gain = synth_score - max_indiv;

                                tracing::debug!(
                                    task_id = %task_id,
                                    synthesis_gain,
                                    "synthesis re-verification passed"
                                );

                                Some((synth_out.synthesis_text, synthesis_gain))
                            } else {
                                tracing::warn!(
                                    task_id = %task_id,
                                    "synthesis re-verification failed; falling back to selection chain"
                                );
                                None
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                task_id = %task_id,
                                error = %e,
                                "synthesis phase error; falling back to selection chain"
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
            // ── End Phase 5a ─────────────────────────────────────────────────────────

            // When synthesis produced a valid re-verified output, bypass selection chain
            // and return directly without calling MergeEngine::resolve.
            if let Some((synthesis_text, synthesis_gain)) = synthesis_output {
                attribution.synthesis_gain = synthesis_gain;
                quality_history.push(QualityMeasurement {
                    params: current_params.clone(),
                    q_confidence: attribution.q_confidence + synthesis_gain,
                });
                let suggested_next = SelfOptimizer::suggest(SuggestInput {
                    current: &current_params,
                    history: &quality_history,
                    n_max_ceiling,
                    n_optimal: input
                        .calibration
                        .ensemble
                        .as_ref()
                        .map(|ec| ec.n_optimal as u32),
                    p_mean,
                    rho_mean,
                    filter_ratio,
                    cfg: input.cfg,
                });
                let waste_ratio = filter_ratio;
                let selection_resolved = SelectionResolvedEvent {
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
                input.store.mark_resolved(&task_id);
                if let Some(ref bandit_arc) = input.bandit_state {
                    let n_used = current_params.n_agents;
                    let tier3_score = Some(attribution.q_confidence.clamp(0.0, 1.0));
                    let mut bandit = bandit_arc.write().await;
                    bandit.update(n_used, None, tier3_score);
                    bandit.apply_optimizer_hint(n_used, suggested_next.n_agents);
                }
                let run_scores: Vec<f64> =
                    all_verification_events.iter().map(|e| e.score).collect();
                let talagrand = TalagrandDiagnostic::from_verification_scores(&[run_scores]);
                return Ok(EngineOutput {
                    task_id,
                    resolved_output: synthesis_text,
                    selection_resolved,
                    attribution,
                    attribution_interval,
                    verification_events: all_verification_events,
                    failed_proposals: all_failed_proposals.clone(),
                    talagrand,
                    suggested_next_params: Some(suggested_next),
                    waste_ratio,
                    applied_optimizations: vec![],
                    topology_retry_events: topology_retry_events.clone(),
                    mode_collapse_count,
                    epistemic_yield: None,
                    task_quadrant: Some(assessed_quadrant),
                    complexity_event: complexity_event_for_output.clone(),
                    frontier_event: frontier_event.clone(),
                    adapter_correctness: adapter_correctness.clone(),
                    coherence_state: wave_coherence,
                    comparison_events: all_comparison_events.clone(),
                    shadow_audit_events: all_shadow_audit_events.clone(),
                    correlated_warnings: all_correlated_warnings.clone(),
                    researcher_grounding_events: all_researcher_grounding_events.clone(),
                    diversity_degraded_event: diversity_degraded_event_for_output.clone(),
                    srani_events: all_srani_events.clone(),
                    srani_ema_cfi_updated: srani_ema_updated,
                    srani_count_updated,
                    oracle_gate_passed: oracle_gate_passed_flag,
                });
            }

            let outcome = MergeEngine::resolve(
                task_id.clone(),
                proposal_set,
                pruned,
                provisioned.merge_strategy.clone(),
                retry_count,
                input.embedding_model,
            )
            .await;

            all_verification_events.extend(iteration_verification_events.clone());

            // Talagrand τ feedback: after each iteration, update τ-spread expansion factor.
            // Using the current iteration's scores — typically Insufficient (< 20 runs) for a
            // single task iteration, but may trigger on longer sessions with accumulated data.
            {
                let iter_scores: Vec<f64> = iteration_verification_events
                    .iter()
                    .map(|e| e.score)
                    .collect();
                if let Some(diag) = TalagrandDiagnostic::from_verification_scores(&[iter_scores]) {
                    tau_spread_factor = diag
                        .tau_expansion_factor(tau_spread_factor, input.cfg.tau_spread_max_factor);
                }
            }

            match outcome {
                MergeOutcome::Resolved {
                    selection_resolved,
                    resolved,
                } => {
                    quality_history.push(QualityMeasurement {
                        params: current_params.clone(),
                        q_confidence: attribution.q_confidence,
                    });
                    let suggested_next = SelfOptimizer::suggest(SuggestInput {
                        current: &current_params,
                        history: &quality_history,
                        n_max_ceiling,
                        n_optimal: input
                            .calibration
                            .ensemble
                            .as_ref()
                            .map(|ec| ec.n_optimal as u32),
                        p_mean,
                        rho_mean,
                        filter_ratio,
                        cfg: input.cfg,
                    });

                    let waste_ratio = filter_ratio;
                    let applied_optimizations = if waste_ratio < input.cfg.optimizer_waste_threshold
                    {
                        // N changes are bandit-owned; only record threshold suggestions.
                        let mut opts = Vec::new();
                        if (suggested_next.verify_threshold - current_params.verify_threshold).abs()
                            > 1e-9
                        {
                            opts.push(h2ai_types::events::AppliedOptimization {
                                kind: h2ai_types::events::OptimizationKind::TauSpreadAdjusted,
                                reason: format!(
                                    "waste_ratio={:.2} < threshold {:.2}; \
                                         tighten verify_threshold to reduce pruned proposals",
                                    waste_ratio, input.cfg.optimizer_waste_threshold
                                ),
                                before: format!("{:.3}", current_params.verify_threshold),
                                after: format!("{:.3}", suggested_next.verify_threshold),
                            });
                        }
                        opts
                    } else {
                        vec![]
                    };

                    input.store.mark_resolved(&task_id);

                    // Compute epistemic yield synchronously so the result is returned to callers.
                    // When an embedding model is available, use cosine n_eff (eigenvalue-based).
                    // Otherwise fall back to a Jaccard-diversity approximation:
                    //   yield = mean_jaccard_distance × (n_survivors / n_requested)
                    // This approximation degrades gracefully — it's 0 when proposals are identical
                    // and approaches 1 when they are maximally diverse and all survive.
                    let epistemic_yield: Option<f64> = {
                        let surviving_texts: Vec<String> = synthesis_candidates
                            .iter()
                            .map(|p| p.raw_output.clone())
                            .collect();
                        let n_requested = all_raw_texts_this_wave.len().max(1);
                        if let Some(model) = input.embedding_model {
                            let n_eff = h2ai_autonomic::epistemic::compute_n_eff_cosine(
                                &surviving_texts,
                                model,
                                input.cfg.eigen_n_eff_delta,
                            );
                            let yield_ratio = n_eff / n_requested as f64;
                            tracing::debug!(
                                n_eff_cosine_actual = n_eff,
                                yield_ratio,
                                "EpistemicYield computed (cosine)"
                            );
                            Some(yield_ratio.clamp(0.0, 1.0))
                        } else if surviving_texts.len() >= 2 {
                            let refs: Vec<&str> =
                                surviving_texts.iter().map(|s| s.as_str()).collect();
                            let mean_jaccard = crate::correlated_hallucination::compute_cv(&refs)
                                .map(|s| s.mean_jaccard_distance)
                                .unwrap_or(0.0);
                            let survival_rate = surviving_texts.len() as f64 / n_requested as f64;
                            let yield_approx = mean_jaccard * survival_rate;
                            tracing::debug!(
                                mean_jaccard,
                                survival_rate,
                                yield_approx,
                                "EpistemicYield computed (jaccard fallback)"
                            );
                            Some(yield_approx.clamp(0.0, 1.0))
                        } else {
                            None
                        }
                    };

                    if let Some(ref bandit_arc) = input.bandit_state {
                        let n_used = current_params.n_agents;
                        let tier3_score = Some(attribution.q_confidence.clamp(0.0, 1.0));
                        let mut bandit = bandit_arc.write().await;
                        bandit.update(n_used, None, tier3_score);
                        bandit.apply_optimizer_hint(n_used, suggested_next.n_agents);
                    }
                    let run_scores: Vec<f64> =
                        all_verification_events.iter().map(|e| e.score).collect();
                    let talagrand = TalagrandDiagnostic::from_verification_scores(&[run_scores]);
                    return Ok(EngineOutput {
                        task_id,
                        resolved_output: resolved.resolved_output,
                        selection_resolved,
                        attribution,
                        attribution_interval,
                        verification_events: all_verification_events,
                        failed_proposals: all_failed_proposals.clone(),
                        talagrand,
                        suggested_next_params: Some(suggested_next),
                        waste_ratio,
                        applied_optimizations,
                        topology_retry_events: topology_retry_events.clone(),
                        mode_collapse_count,
                        epistemic_yield,
                        task_quadrant: Some(assessed_quadrant),
                        complexity_event: complexity_event_for_output.clone(),
                        frontier_event: frontier_event.clone(),
                        adapter_correctness: adapter_correctness.clone(),
                        coherence_state: wave_coherence,
                        comparison_events: all_comparison_events.clone(),
                        shadow_audit_events: all_shadow_audit_events.clone(),
                        correlated_warnings: all_correlated_warnings.clone(),
                        researcher_grounding_events: all_researcher_grounding_events.clone(),
                        diversity_degraded_event: diversity_degraded_event_for_output.clone(),
                        srani_events: all_srani_events.clone(),
                        srani_ema_cfi_updated: srani_ema_updated,
                        srani_count_updated,
                        oracle_gate_passed: oracle_gate_passed_flag,
                    });
                }
                MergeOutcome::ZeroSurvival(mut zero_event) => {
                    // GAP-A4 #4: coherence-closed early exit.
                    // If is_closed() AND proposals survived verification (synthesis_candidates
                    // non-empty), the auditor is the blocker — constraint coverage is complete
                    // and retrying cannot improve it. Skip further retries.
                    // Guard: when synthesis_candidates is empty (all proposals verifier-rejected),
                    // is_closed() is vacuously true (no domains were tracked) and the verifier
                    // may accept a better proposal on retry.
                    if wave_coherence.is_closed() && !synthesis_candidates.is_empty() {
                        tracing::info!(
                            target: "h2ai.coherence",
                            retry = retry_count,
                            uncovered_domains = ?wave_coherence.uncovered_domains,
                            active_contradictions = wave_coherence.active_contradictions.len(),
                            "coherence closed at ZeroSurvival — auditor is the blocker, \
                             constraint coverage complete; skipping further retries"
                        );
                        input.store.mark_failed(&task_id);
                        return Err(EngineError::MaxRetriesExhausted {
                            partial_verification_events: all_verification_events.clone(),
                        });
                    }

                    // Compute epistemic diagnostics synchronously on the MAPE-K path.
                    let detected_failure_mode = if let Some(model) = input.embedding_model {
                        let n_eff = h2ai_autonomic::epistemic::compute_n_eff_cosine(
                            &all_raw_texts_this_wave,
                            model,
                            input.cfg.eigen_n_eff_delta,
                        );
                        let failure = h2ai_autonomic::epistemic::classify_failure_mode(
                            n_eff,
                            all_raw_texts_this_wave.len().max(1),
                            input.cfg.safety.diversity_threshold,
                        );
                        zero_event.n_eff_cosine_actual = Some(n_eff);
                        zero_event.failure_mode = Some(failure.clone());
                        Some(failure)
                    } else {
                        None
                    };

                    // Apply FailureMode routing before RetryPolicy topology selection.
                    match &detected_failure_mode {
                        Some(h2ai_types::events::FailureMode::ModeCollapse) => {
                            let pool_len = input.explorer_adapters.len().max(1);
                            adapter_rotation_offset = (adapter_rotation_offset + 1) % pool_len;
                            mode_collapse_count += 1;
                            pending_tombstone = None;
                        }
                        Some(h2ai_types::events::FailureMode::ConstrainedExploration) => {
                            let all_violations: Vec<h2ai_types::events::ConstraintViolation> =
                                all_pruned
                                    .iter()
                                    .flat_map(|p| p.violated_constraints.iter().cloned())
                                    .collect();
                            pending_tombstone =
                                h2ai_autonomic::epistemic::synthesize_tombstone(&all_violations);
                        }
                        Some(h2ai_types::events::FailureMode::CorrelatedHallucination {
                            ..
                        }) => {
                            // C1 retries are handled directly before Phase 3.5 — no extra routing needed here.
                        }
                        None => {}
                    }

                    match RetryPolicy::decide(
                        &zero_event,
                        &tried_topologies,
                        all_pruned.clone(),
                        tau_values_tried.clone(),
                        last_multiplication_failure.clone(),
                    ) {
                        RetryAction::Retry(next_topology) => {
                            force_topology = Some(next_topology);
                            Self::apply_optimizer(
                                &mut current_params,
                                &mut tao_config,
                                &mut verification_config,
                                &quality_history,
                                n_max_ceiling,
                                cg_mean,
                                filter_ratio,
                                input.calibration.ensemble.as_ref(),
                                input.cfg,
                            );
                        }
                        RetryAction::RetryWithTauReduction {
                            topology,
                            tau_factor,
                        } => {
                            force_topology = Some(topology);
                            tau_reduction_factor *= tau_factor;
                            Self::apply_optimizer(
                                &mut current_params,
                                &mut tao_config,
                                &mut verification_config,
                                &quality_history,
                                n_max_ceiling,
                                cg_mean,
                                filter_ratio,
                                input.calibration.ensemble.as_ref(),
                                input.cfg,
                            );
                        }
                        RetryAction::RetryWithHints { topology, hints } => {
                            force_topology = Some(topology);
                            if !hints.is_empty() {
                                let attempts_remaining =
                                    input.cfg.max_autonomic_retries.saturating_sub(retry_count);
                                let hint_lines = hints
                                    .iter()
                                    .map(|h| format!("• {h}"))
                                    .collect::<Vec<_>>()
                                    .join("\n\n");
                                retry_context = Some(format!(
                                    "{system_context_with_rubric}\n\n--- CONSTRAINT FEEDBACK (iteration {retry_count}) ---\n\
                                    The following constraints were violated. Fix ALL of these in your next response:\n\n\
                                    {hint_lines}\n\n\
                                    {attempts_remaining} retry attempt(s) remaining.\n\
                                    ---"
                                ));
                            }
                            Self::apply_optimizer(
                                &mut current_params,
                                &mut tao_config,
                                &mut verification_config,
                                &quality_history,
                                n_max_ceiling,
                                cg_mean,
                                filter_ratio,
                                input.calibration.ensemble.as_ref(),
                                input.cfg,
                            );
                        }
                        RetryAction::Fail(reason) => {
                            tracing::warn!(
                                target: "h2ai.engine",
                                task_id = %task_id,
                                retry_count,
                                reason = ?reason,
                                "retry policy decided Fail (post-phase5) — giving up"
                            );
                            input.store.mark_failed(&task_id);
                            return Err(EngineError::MaxRetriesExhausted {
                                partial_verification_events: all_verification_events.clone(),
                            });
                        }
                    }
                }
            }
        }

        tracing::warn!(
            target: "h2ai.engine",
            task_id = %task_id,
            max_retries = input.cfg.max_autonomic_retries,
            last_multiplication_failure = ?last_multiplication_failure,
            all_pruned_count = all_pruned.len(),
            "retry loop exhausted all attempts — giving up"
        );
        input.store.mark_failed(&task_id);
        Err(EngineError::MaxRetriesExhausted {
            partial_verification_events: all_verification_events,
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
        let phase = crate::task_store::TaskPhase::try_from_name_str(&checkpoint.phase);

        if let Some(crate::task_store::TaskPhase::Merging) = phase {
            let resolved = checkpoint.resolved_output.ok_or_else(|| {
                EngineError::Parse("Merging checkpoint missing resolved_output".into())
            })?;

            let task_id = input.task_id.clone();
            input.store.mark_resolved(&task_id);

            // Return minimal EngineOutput with saved resolved output.
            // Aggregated fields (verification, attribution, etc.) are zeroed —
            // the engine did not re-run so there are no new measurements.
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
                },
                attribution: crate::attribution::HarnessAttribution {
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
                },
                attribution_interval: None,
                verification_events: vec![],
                failed_proposals: vec![],
                talagrand: None,
                suggested_next_params: None,
                waste_ratio: 0.0,
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
            })
        } else {
            // Earlier phase or unknown phase — restart from scratch
            Self::run_offline(input).await
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_optimizer(
        current_params: &mut OptimizerParams,
        tao_config: &mut TaoConfig,
        verification_config: &mut VerificationConfig,
        history: &[QualityMeasurement],
        n_max_ceiling: u32,
        cg_mean: f64,
        filter_ratio: f64,
        ensemble: Option<&h2ai_types::sizing::EnsembleCalibration>,
        cfg: &H2AIConfig,
    ) {
        let (p_mean, rho_mean) = match ensemble {
            Some(ec) => (ec.p_mean, ec.rho_mean),
            None => (0.5 + cg_mean / 2.0, (1.0 - cg_mean).clamp(0.0, 1.0)),
        };
        let n_optimal = ensemble.map(|ec| ec.n_optimal as u32);
        let suggested = SelfOptimizer::suggest(SuggestInput {
            current: current_params,
            history,
            n_max_ceiling,
            n_optimal,
            p_mean,
            rho_mean,
            filter_ratio,
            cfg,
        });
        if suggested.max_turns != current_params.max_turns {
            tao_config.max_turns = suggested.max_turns as u8;
        }
        if (suggested.verify_threshold - current_params.verify_threshold).abs() > 1e-9 {
            verification_config.threshold = suggested.verify_threshold;
        }
        *current_params = suggested;
    }
}
