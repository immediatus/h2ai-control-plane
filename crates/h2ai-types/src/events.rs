use crate::config::{
    AdapterKind, AuditorConfig, ExplorerConfig, ParetoWeights, ReviewGate, TopologyKind,
};
use crate::identity::{ExplorerId, SubtaskId, TaskId};
use crate::physics::{
    CoherencyCoefficients, CoordinationThreshold, EigenCalibration, EnsembleCalibration,
    MergeStrategy, MultiplicationConditionFailure, PredictionBasis, RoleErrorCost, TauValue,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// How CG(i,j) was computed during calibration.
///
/// `EmbeddingCosine` means an embedding model was available and CG is the fraction of
/// calibration prompts where `cosine(embed_i, embed_j) > cg_agreement_threshold` — the
/// semantically correct measurement per the blog spec.
///
/// `TokenJaccard` is the fallback when no embedding model is configured: mean per-prompt
/// token Jaccard similarity. Downstream quality predictions are less accurate in this mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CgMode {
    /// CG is an embedding cosine agreement rate (semantically correct).
    EmbeddingCosine,
    /// CG is mean token Jaccard similarity (fallback — configure an embedding model for accuracy).
    #[default]
    TokenJaccard,
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
    /// β₀ derived from timing the pairwise CG reconciliation loop during calibration.
    /// More accurate than I/O-timing-derived β₀ for semantic reconciliation cost.
    /// `None` when fewer than 2 adapters ran calibration.
    #[serde(default)]
    pub pairwise_beta: Option<f64>,
    /// How CG was computed: embedding cosine agreement rate (accurate) or token Jaccard (fallback).
    /// Defaults to `TokenJaccard` when deserialising events written before this field was added.
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

/// Emitted when a task is initialised: system context compiled and J_eff computed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBootstrappedEvent {
    pub task_id: TaskId,
    pub system_context: String,
    pub pareto_weights: ParetoWeights,
    pub j_eff: f64,
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

/// Emitted when the merge engine selects surviving proposals and computes the semilattice join.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemilatticeCompiledEvent {
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
/// Published alongside `SemilatticeCompiled` on the success path.
/// `q_predicted` is the heuristic/empirical estimate; `q_measured` (when present)
/// is the Tier 1 oracle result. The interval fields are `None` when fewer than
/// 2 CG calibration samples are available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAttributionEvent {
    pub task_id: TaskId,
    /// Heuristic or empirical Q_total estimate from CG/USL/CJT chain.
    pub q_predicted: f64,
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

/// Discriminated union of all events published to the NATS event stream by the orchestrator.
///
/// Serialised with an `event_type` tag and a `payload` content field for downstream consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "payload")]
pub enum H2AIEvent {
    /// Wraps [`CalibrationCompletedEvent`]: calibration harness finished.
    CalibrationCompleted(CalibrationCompletedEvent),
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
    /// Wraps [`SemilatticeCompiledEvent`]: merge engine finished selecting and joining proposals.
    SemilatticeCompiled(SemilatticeCompiledEvent),
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
}

impl H2AIEvent {
    pub fn subject(&self, task_id: &TaskId) -> String {
        format!("h2ai.tasks.{}", task_id)
    }
}
