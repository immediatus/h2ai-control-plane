use crate::config::{
    AdapterKind, AuditorConfig, ExplorerConfig, ParetoWeights, ReviewGate, TopologyKind,
};
use crate::identity::{ExplorerId, SubtaskId, TaskId};
use crate::sizing::{
    CoherencyCoefficients, CoordinationThreshold, EigenCalibration, EnsembleCalibration,
    MergeStrategy, MultiplicationConditionFailure, PredictionBasis, ProbeSkipReason, RoleErrorCost,
    TaskQuadrant, TauValue,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Quality level of the current calibration state.
///
/// Used in Phase 1.5 bootstrap guard: when `Bootstrap`, synthetic priors are the only
/// source and the N-probe sampling is bypassed (routes to Coverage unconditionally).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CalibrationQuality {
    /// Calibration has run against real adapters; priors are empirically grounded.
    #[default]
    Domain,
    /// Only synthetic priors available (no real adapter data yet).
    Bootstrap,
}

/// How CG(i,j) was computed during calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CgMode {
    /// CG is mean pairwise Hamming distance between constraint satisfaction profiles.
    /// Falls back to `cfg.calibration_cg_fallback` when no constraint corpus is provided.
    #[default]
    ConstraintProfile,
    /// CG is the fraction of calibration prompts where cosine(embed_i, embed_j) > θ_agree.
    /// Semantically robust: paraphrase-insensitive, matches the theoretical specification.
    /// Requires the `fastembed-embed` feature and an `EmbeddingModel` in `AppState`.
    EmbeddingCosine,
}

/// Classifies why all proposals were pruned in a MAPE-K zero-survival wave.
///
/// Computed synchronously from cosine N_eff before re-provisioning.
/// Drives retry routing: ConstrainedExploration injects a tombstone;
/// ModeCollapse rotates the adapter selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureMode {
    /// Agents explored diverse solution areas but none satisfied constraints.
    /// Retry: same topology, inject Constraint Violation Tombstone.
    ConstrainedExploration,
    /// Agents converged on a shared hallucination (N_eff ≈ 1).
    /// Retry: rotate adapter selection or widen τ_spread.
    ModeCollapse,
}

/// Emitted asynchronously after `MergeResolvedEvent` — does not block task close.
///
/// Measures semantic independence of the surviving proposals. `yield_ratio` uses
/// `N_requested` as the denominator (not `N_responded`) — financial yield: you paid
/// for N adapters, you received `n_eff_cosine_actual` independent perspectives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicYieldEvent {
    pub task_id: TaskId,
    pub n_eff_cosine_actual: f64,
    pub n_eff_prior: f64,
    /// n_eff_actual / N_requested
    pub yield_ratio: f64,
    pub adapters: Vec<String>,
}

/// Emitted when the calibration harness finishes measuring α, β₀, and CG for the adapter pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCompletedEvent {
    pub calibration_id: TaskId,
    pub coefficients: CoherencyCoefficients,
    pub coordination_threshold: CoordinationThreshold,
    /// Condorcet-based ensemble calibration. `None` when < 2 adapters ran calibration
    /// (falls back to config defaults).
    pub ensemble: Option<EnsembleCalibration>,
    /// Eigenvalue-based calibration (from pairwise CG matrix). `None` when fewer than 2 adapters.
    pub eigen: Option<EigenCalibration>,
    pub timestamp: DateTime<Utc>,
    /// β₀ derived from timing the pairwise CG measurement loop during calibration.
    /// Captures coherence-drag baseline; the CG coupling then adjusts for divergence severity.
    /// `None` when fewer than 2 adapters ran calibration.
    #[serde(default)]
    pub pairwise_beta: Option<f64>,
    /// How CG was computed: constraint Hamming distance profile or fallback.
    /// Defaults to `ConstraintProfile` when deserialising events written before this field was added.
    #[serde(default)]
    pub cg_mode: CgMode,
    /// Distinct non-Mock adapter families present in the calibration pool (sorted).
    /// Empty when all adapters are Mock (test-only deployments).
    #[serde(default)]
    pub adapter_families: Vec<String>,
    /// True when explorer and verification adapters are from the same non-Mock family.
    /// LLM self-preference judge bias is likely; consider routing verification to a different family.
    #[serde(default)]
    pub explorer_verification_family_match: bool,
    /// True when all non-Mock adapters belong to a single family.
    /// Weiszfeld BFT correlated hallucination protection is degraded.
    #[serde(default)]
    pub single_family_warning: bool,
    /// Lower bound of N_max one-σ confidence interval (CG_mean − cg_std_dev).
    /// Equals `n_max()` when only one CG sample exists.
    #[serde(default)]
    pub n_max_lo: f64,
    /// Upper bound of N_max one-σ confidence interval (CG_mean + cg_std_dev).
    /// `n_max_lo ≤ n_max() ≤ n_max_hi`. Wide interval = high CG measurement variance.
    #[serde(default)]
    pub n_max_hi: f64,
    /// Pool-level semantic independence measured at calibration time via cosine N_eff.
    /// Used as the Bayesian prior at task provisioning. `0.0` when no EmbeddingModel
    /// is present (fallback formula: 1.0 + cg_fallback × (N − 1) is computed in the harness).
    #[serde(default)]
    pub n_eff_cosine_prior: f64,
    /// Whether this calibration is empirically grounded (`Domain`) or synthetic-prior only
    /// (`Bootstrap`). Phase 1.5 skips the N-probe path when `Bootstrap`.
    /// Defaults to `Domain` so existing serialised events deserialise correctly.
    #[serde(default)]
    pub calibration_quality: CalibrationQuality,
}

/// Point-in-time snapshot of a task's in-memory state for crash-recovery replay optimization.
/// Stored in NATS KV at key `snapshots/{task_id}/latest`.
/// Recovery loads this snapshot then replays only events with sequence > `last_sequence`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSnapshot {
    pub task_id: TaskId,
    /// NATS JetStream sequence number of the last event included in this snapshot.
    pub last_sequence: u64,
    /// Serialized `TaskState` as JSON — stored as a string to avoid a circular crate dependency.
    pub task_state_json: String,
    pub taken_at: DateTime<Utc>,
}

/// Emitted when a task is initialised: system context compiled and locked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBootstrappedEvent {
    pub task_id: TaskId,
    pub system_context: String,
    pub pareto_weights: ParetoWeights,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the planner selects topology, explorer roles, and merge strategy for a retry iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyProvisionedEvent {
    pub task_id: TaskId,
    pub topology_kind: TopologyKind,
    pub explorer_configs: Vec<ExplorerConfig>,
    pub auditor_config: AuditorConfig,
    pub n_max: f64,
    pub interface_n_max: Option<f64>,
    #[serde(alias = "kappa_eff")]
    pub beta_eff: f64,
    pub role_error_costs: Vec<RoleErrorCost>,
    pub merge_strategy: MergeStrategy,
    pub coordination_threshold: CoordinationThreshold,
    pub review_gates: Vec<ReviewGate>,
    pub retry_count: u32,
    pub timestamp: DateTime<Utc>,
    /// Dense constraint violation summary injected on `ConstrainedExploration` retries.
    /// Contains constraint IDs and c_i weights only — never raw proposal text.
    /// `None` on wave 1 and on `ModeCollapse` retries.
    #[serde(default)]
    pub constraint_tombstone: Option<String>,
}

/// Emitted when the multiplication condition gate rejects the current topology on a given retry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplicationConditionFailedEvent {
    pub task_id: TaskId,
    pub failure: MultiplicationConditionFailure,
    pub retry_count: u32,
    pub timestamp: DateTime<Utc>,
}

/// Why an explorer failed to produce a proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalFailureReason {
    /// The adapter did not respond within the per-turn deadline.
    Timeout,
    /// The adapter process was killed by the OOM killer; the message is the signal detail.
    OomPanic(String),
    /// The adapter returned an error; the message contains the error description.
    AdapterError(String),
}

/// Emitted when an explorer completes a TAO loop and produces a raw output proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub tau: TauValue,
    /// TAO retry-loop generation counter. First attempt = 0; each MAPE-K retry increments by 1.
    /// Used by `ProposalSet` as the primary LUB key: higher generation always supersedes lower.
    #[serde(default)]
    pub generation: u64,
    pub raw_output: String,
    pub token_cost: u64,
    pub adapter_kind: AdapterKind,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when an explorer's TAO loop terminates without producing a usable proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalFailedEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub reason: ProposalFailureReason,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when all explorers in Phase 3 have finished (or timed out), summarising success/failure counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPhaseCompletedEvent {
    pub task_id: TaskId,
    pub total_explorers: u32,
    pub successful: u32,
    pub failed: u32,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the verification phase starts evaluating a specific explorer's proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub timestamp: DateTime<Utc>,
}

/// A single constraint that a proposal violated during the verification phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_id: String,
    /// Predicate score [0,1]; 0 = total violation.
    pub score: f64,
    /// "Hard", "Soft", or "Advisory"
    pub severity_label: String,
    pub remediation_hint: Option<String>,
}

/// Emitted when an explorer's proposal is eliminated by the verification or auditor gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchPrunedEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub reason: String,
    pub constraint_error_cost: RoleErrorCost,
    pub violated_constraints: Vec<ConstraintViolation>,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when all proposals for a retry iteration were pruned, triggering MAPE-K retry logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroSurvivalEvent {
    pub task_id: TaskId,
    pub retry_count: u32,
    pub timestamp: DateTime<Utc>,
    /// Effective independent adapters computed from cosine similarity on failed proposals.
    /// `None` when no `EmbeddingModel` is present in `AppState`.
    #[serde(default)]
    pub n_eff_cosine_actual: Option<f64>,
    /// MAPE-K failure classification. `None` when no EmbeddingModel is available.
    #[serde(default)]
    pub failure_mode: Option<FailureMode>,
}

/// Emitted when CG_embed falls below `cg_collapse_threshold`.
/// The planner forces N_max=1 — no ensemble benefit is possible when coordination quality collapses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCoordinationQualityEvent {
    pub task_id: TaskId,
    pub cg_embed: f64,
    pub forced_n_max: u32,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the maximum role error cost exceeds the BFT threshold, signalling consensus-mode merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRequiredEvent {
    pub task_id: TaskId,
    pub max_role_error_cost: RoleErrorCost,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the merge engine finishes selecting surviving proposals.
///
/// The CRDT semilattice resolves to a single winning proposal by selection; content synthesis,
/// if enabled, is a separate Phase 5a operation. This event records which proposals survived
/// and which were pruned, the merge strategy used, and the merge timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResolvedEvent {
    pub task_id: TaskId,
    pub valid_proposals: Vec<ExplorerId>,
    pub pruned_proposals: Vec<(ExplorerId, String)>,
    pub merge_strategy: MergeStrategy,
    pub timestamp: DateTime<Utc>,
    /// Wall-clock seconds consumed by MergeEngine::resolve() for this event.
    /// `None` for events reconstructed from older serialised logs.
    #[serde(default)]
    pub merge_elapsed_secs: Option<f64>,
    /// Number of proposals (valid + pruned) that entered resolve().
    #[serde(default)]
    pub n_input_proposals: usize,
}

/// Emitted when the merge engine produces the final resolved output string for a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResolvedEvent {
    pub task_id: TaskId,
    pub resolved_output: String,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the MAPE-K loop exhausts all retries without producing a resolved output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFailedEvent {
    pub task_id: TaskId,
    pub pruned_events: Vec<BranchPrunedEvent>,
    pub topologies_tried: Vec<TopologyKind>,
    pub tau_values_tried: Vec<Vec<f64>>,
    pub multiplication_condition_failure: Option<MultiplicationConditionFailure>,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when a review gate fires and routes a proposal to a reviewer explorer for approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewGateTriggeredEvent {
    pub task_id: TaskId,
    pub gate_id: String,
    pub blocked_explorer_id: ExplorerId,
    pub reviewer_explorer_id: ExplorerId,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when a reviewer explorer rejects the proposal at a review gate, blocking it from merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewGateBlockedEvent {
    pub task_id: TaskId,
    pub gate_id: String,
    pub blocked_explorer_id: ExplorerId,
    pub reviewer_explorer_id: ExplorerId,
    pub rejection_reason: String,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when active subtask count approaches `interface_n_max`, warning of interface saturation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceSaturationWarningEvent {
    pub task_id: TaskId,
    pub active_subtasks: u32,
    pub interface_n_max: f64,
    pub saturation_ratio: f64,
    pub timestamp: DateTime<Utc>,
}

/// Emitted after each TAO loop turn, recording the observation and whether the turn passed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaoIterationEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub turn: u8,
    pub observation: String,
    pub passed: bool,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the LLM-as-Judge verifier assigns a compliance score to a proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationScoredEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub score: f64,
    pub reason: String,
    pub passed: bool,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the orchestrator creates a decomposition plan for a multi-step task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskPlanCreatedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub subtask_count: usize,
    pub timestamp: DateTime<Utc>,
}

/// Covers both approved and rejected outcomes — use `approved` field to distinguish.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskPlanReviewedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub approved: bool,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when an individual subtask begins execution within a wave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskStartedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub subtask_id: SubtaskId,
    pub description: String,
    pub wave: usize,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when an individual subtask finishes successfully, recording token cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskCompletedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub subtask_id: SubtaskId,
    pub token_cost: u64,
    pub timestamp: DateTime<Utc>,
}

/// Category of self-optimizer suggestion applied on a wasteful-but-successful run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationKind {
    /// SelfOptimizer suggested adjusting the verify_threshold to reduce wasted proposals.
    TauSpreadAdjusted,
    /// SelfOptimizer suggested switching topology (stored as a one-shot hint in AppState).
    TopologyHintSet,
}

/// One self-optimizer suggestion that was applied on a completed task run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedOptimization {
    pub kind: OptimizationKind,
    pub reason: String,
    /// Human-readable description of the parameter before the suggestion.
    pub before: String,
    /// Human-readable description of the parameter after the suggestion.
    pub after: String,
}

/// Quality attribution snapshot for a completed task.
///
/// Published alongside `SelectionResolved` on the success path.
/// `q_confidence` is the heuristic/empirical confidence estimate from the CG/USL/CJT chain —
/// it measures how confident the system is in its output, not whether the output is correct.
/// `q_measured` (when present) is the Tier 1 oracle result (actual correctness).
/// The interval fields are `None` when fewer than 2 CG calibration samples are available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAttributionEvent {
    pub task_id: TaskId,
    /// Heuristic or empirical confidence estimate from CG/USL/CJT chain.
    /// This is a confidence score (system's self-assessment), not oracle-grounded quality.
    /// See `prediction_basis` for whether it is `Heuristic` or `Empirical`.
    pub q_confidence: f64,
    /// Fraction of Tier 1 oracle tests passed. `None` when no oracle ran.
    #[serde(default)]
    pub q_measured: Option<f64>,
    /// 5th percentile of the bootstrap or conformal interval.
    #[serde(default)]
    pub q_interval_lo: Option<f64>,
    /// 95th percentile of the bootstrap or conformal interval.
    #[serde(default)]
    pub q_interval_hi: Option<f64>,
    /// Source of quality predictions: `Heuristic` or `Empirical`.
    pub prediction_basis: PredictionBasis,
    /// Fraction of dispatched proposals that survived verification (valid / total_evaluated).
    /// 1.0 = no waste; below `optimizer_waste_threshold` = wasteful run.
    #[serde(default = "default_waste_ratio")]
    pub waste_ratio: f64,
    /// SelfOptimizer suggestions applied on this successful-but-wasteful run.
    /// Empty when the run was not wasteful or no applicable suggestions existed.
    #[serde(default)]
    pub applied_optimizations: Vec<AppliedOptimization>,
    pub timestamp: DateTime<Utc>,
}

fn default_waste_ratio() -> f64 {
    1.0
}

/// Emitted at the end of Phase 1.5 (Task Complexity Assessment).
///
/// Records the full complexity signal chain: structural prior → optional empirical probe →
/// effective TCC → quadrant classification. Always emitted even in shadow mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskComplexityAssessedEvent {
    pub task_id: TaskId,
    /// TCC_structural: zero-cost prior from corpus metadata (formula-based, no LLM calls).
    pub tcc_structural: f64,
    /// TCC_empirical: participation ratio from N-probe satisfaction matrix.
    /// `None` when probe was skipped (see `probe_skip_reason`).
    #[serde(default)]
    pub tcc_empirical: Option<f64>,
    /// TCC_effective = max(tcc_structural, tcc_empirical) + mismatch_penalty.
    /// Equals tcc_structural when probe was skipped.
    pub tcc_effective: f64,
    /// Pool-level N_eff from the most recent calibration (eigenvalue participation ratio).
    /// `None` when EigenCalibration was not available at calibration time.
    #[serde(default)]
    pub n_eff_pool: Option<f64>,
    /// Routing quadrant before shadow_mode override.
    pub task_quadrant: TaskQuadrant,
    /// Whether the N-probe mini-generation step was skipped.
    pub probe_skipped: bool,
    /// Reason the probe was skipped; `None` when probe ran.
    #[serde(default)]
    pub probe_skip_reason: ProbeSkipReason,
    /// Fraction of Heavy-tier constraints (OracleExecution) in the corpus.
    pub heavy_fraction: f64,
    /// True when tcc_empirical diverges from tcc_structural by > 0.3 (signal mismatch).
    pub tcc_mismatch: bool,
    /// Total tokens consumed by the probe mini-generation calls (0 when probe skipped).
    pub probe_cost_tokens: u64,
    /// Number of Static-tier constraints that produced informative variation in the probe.
    pub n_informative_static: usize,
    pub timestamp: DateTime<Utc>,
}

/// Emitted after Phase 3.5 verification; records the full constraint satisfaction matrix
/// across all surviving proposals for Pareto frontier coverage analysis.
///
/// Used for H3 validation in the GAP-A1 experiment: measures whether cross-family
/// committees actually cover more of the constraint Pareto frontier than Self-MoA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintFrontierEvent {
    pub task_id: TaskId,
    /// satisfaction_matrix[i][j] = score of proposal i on constraint j ∈ [0, 1].
    pub satisfaction_matrix: Vec<Vec<f64>>,
    /// Constraint IDs in column order.
    pub constraint_ids: Vec<String>,
    /// Explorer IDs in row order.
    pub explorer_ids: Vec<ExplorerId>,
    /// Participation ratio (Σλ)²/Σλ² of the column-space eigenvalues.
    /// Measures how much of the constraint Pareto frontier was covered by the ensemble.
    pub pareto_coverage: f64,
    pub timestamp: DateTime<Utc>,
}

/// Discriminated union of all events published to the NATS event stream by the orchestrator.
///
/// Serialised with an `event_type` tag and a `payload` content field for downstream consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "payload")]
pub enum H2AIEvent {
    /// Wraps [`CalibrationCompletedEvent`]: calibration harness finished.
    CalibrationCompleted(CalibrationCompletedEvent),
    /// Emitted when calibration fails (e.g. LLM adapter unreachable).
    CalibrationFailed {
        calibration_id: String,
        reason: String,
    },
    /// Wraps [`TaskBootstrappedEvent`]: task context compiled and J_eff gate passed.
    TaskBootstrapped(TaskBootstrappedEvent),
    /// Wraps [`TopologyProvisionedEvent`]: planner selected topology and explorer roles.
    TopologyProvisioned(TopologyProvisionedEvent),
    /// Wraps [`MultiplicationConditionFailedEvent`]: multiplication condition gate rejected the topology.
    MultiplicationConditionFailed(MultiplicationConditionFailedEvent),
    /// Wraps [`ProposalEvent`]: an explorer completed its TAO loop and produced output.
    Proposal(ProposalEvent),
    /// Wraps [`ProposalFailedEvent`]: an explorer's TAO loop terminated without usable output.
    ProposalFailed(ProposalFailedEvent),
    /// Wraps [`GenerationPhaseCompletedEvent`]: all explorers in Phase 3 finished.
    GenerationPhaseCompleted(GenerationPhaseCompletedEvent),
    /// Wraps [`ReviewGateTriggeredEvent`]: a review gate routed a proposal to a reviewer.
    ReviewGateTriggered(ReviewGateTriggeredEvent),
    /// Wraps [`ReviewGateBlockedEvent`]: a reviewer rejected a proposal at a review gate.
    ReviewGateBlocked(ReviewGateBlockedEvent),
    /// Wraps [`ValidationEvent`]: verifier started scoring an explorer's proposal.
    Validation(ValidationEvent),
    /// Wraps [`BranchPrunedEvent`]: an explorer's proposal was eliminated by verification or the auditor.
    BranchPruned(BranchPrunedEvent),
    /// Wraps [`ZeroSurvivalEvent`]: all proposals were pruned, triggering MAPE-K retry.
    ZeroSurvival(ZeroSurvivalEvent),
    /// Wraps [`InterfaceSaturationWarningEvent`]: active subtask count is approaching `interface_n_max`.
    InterfaceSaturationWarning(InterfaceSaturationWarningEvent),
    /// Wraps [`ConsensusRequiredEvent`]: error costs exceed the BFT threshold, switching to consensus merge.
    ConsensusRequired(ConsensusRequiredEvent),
    /// Wraps [`SelectionResolvedEvent`]: merge engine finished selecting surviving proposals.
    SelectionResolved(SelectionResolvedEvent),
    /// Wraps [`MergeResolvedEvent`]: final resolved output string produced for the task.
    MergeResolved(MergeResolvedEvent),
    /// Wraps [`TaskFailedEvent`]: MAPE-K loop exhausted retries without resolving.
    TaskFailed(TaskFailedEvent),
    /// Wraps [`TaoIterationEvent`]: one TAO loop turn completed with its observation and pass/fail status.
    TaoIteration(TaoIterationEvent),
    /// Wraps [`VerificationScoredEvent`]: LLM-as-Judge assigned a compliance score to a proposal.
    VerificationScored(VerificationScoredEvent),
    /// Wraps [`SubtaskPlanCreatedEvent`]: orchestrator created a decomposition plan.
    SubtaskPlanCreated(SubtaskPlanCreatedEvent),
    /// Wraps [`SubtaskPlanReviewedEvent`]: reviewer approved or rejected a decomposition plan.
    SubtaskPlanReviewed(SubtaskPlanReviewedEvent),
    /// Wraps [`SubtaskStartedEvent`]: an individual subtask began execution.
    SubtaskStarted(SubtaskStartedEvent),
    /// Wraps [`SubtaskCompletedEvent`]: an individual subtask finished successfully.
    SubtaskCompleted(SubtaskCompletedEvent),
    /// Wraps [`TaskAttributionEvent`]: quality attribution snapshot for a completed task.
    TaskAttribution(TaskAttributionEvent),
    /// Wraps [`EpistemicYieldEvent`]: semantic independence of surviving proposals (async, post-merge).
    EpistemicYield(EpistemicYieldEvent),
    /// Wraps [`TaskComplexityAssessedEvent`]: Phase 1.5 task complexity and routing quadrant.
    TaskComplexityAssessed(TaskComplexityAssessedEvent),
    /// Wraps [`ConstraintFrontierEvent`]: Pareto frontier coverage of constraint satisfaction matrix.
    ConstraintFrontier(ConstraintFrontierEvent),
}

impl H2AIEvent {
    pub fn subject(&self, task_id: &TaskId) -> String {
        format!("h2ai.tasks.{}", task_id)
    }
}

#[cfg(test)]
mod bivariate_types_tests {
    use super::*;

    #[test]
    fn epistemic_yield_event_roundtrip() {
        use crate::identity::TaskId;
        let ev = EpistemicYieldEvent {
            task_id: TaskId::new(),
            n_eff_cosine_actual: 2.3,
            n_eff_prior: 2.8,
            yield_ratio: 0.77,
            adapters: vec!["anthropic-a".into(), "openai-b".into()],
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: EpistemicYieldEvent = serde_json::from_str(&json).unwrap();
        assert!((back.n_eff_cosine_actual - 2.3).abs() < 1e-9);
        assert_eq!(back.adapters.len(), 2);
    }

    #[test]
    fn failure_mode_serde() {
        let fm = FailureMode::ModeCollapse;
        let s = serde_json::to_string(&fm).unwrap();
        let back: FailureMode = serde_json::from_str(&s).unwrap();
        assert_eq!(back, FailureMode::ModeCollapse);
    }

    #[test]
    fn zero_survival_event_new_fields_default_to_none() {
        let json = r#"{"task_id":"00000000-0000-0000-0000-000000000000","retry_count":0,"timestamp":"2026-01-01T00:00:00Z"}"#;
        let ev: ZeroSurvivalEvent = serde_json::from_str(json).unwrap();
        assert!(ev.n_eff_cosine_actual.is_none());
        assert!(ev.failure_mode.is_none());
    }

    #[test]
    fn topology_provisioned_constraint_tombstone_defaults_none() {
        // Simulate old serialised event that doesn't carry the new field
        let json = r#"{"task_id":"00000000-0000-0000-0000-000000000000","topology_kind":"Ensemble","explorer_configs":[],"auditor_config":{"adapter":{"CloudGeneric":{"endpoint":"x","api_key_env":"X"}},"prompt_template":"","tau":0.2,"max_tokens":1000},"n_max":3.0,"interface_n_max":null,"beta_eff":0.03,"role_error_costs":[],"merge_strategy":"ScoreOrdered","coordination_threshold":0.1,"review_gates":[],"retry_count":0,"timestamp":"2026-01-01T00:00:00Z"}"#;
        let ev: TopologyProvisionedEvent = serde_json::from_str(json).unwrap();
        assert!(ev.constraint_tombstone.is_none());
    }
}
