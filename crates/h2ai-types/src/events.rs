use crate::config::{
    AdapterKind, AuditorConfig, ExplorerConfig, ParetoWeights, ReviewGate, TopologyKind,
};
use crate::identity::{ExplorerId, SubtaskId, TaskId};
use crate::physics::{
    CoherencyCoefficients, CoordinationThreshold, EigenCalibration, EnsembleCalibration,
    MergeStrategy, MultiplicationConditionFailure, RoleErrorCost, TauValue,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBootstrappedEvent {
    pub task_id: TaskId,
    pub system_context: String,
    pub pareto_weights: ParetoWeights,
    pub j_eff: f64,
    pub timestamp: DateTime<Utc>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplicationConditionFailedEvent {
    pub task_id: TaskId,
    pub failure: MultiplicationConditionFailure,
    pub retry_count: u32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalFailureReason {
    Timeout,
    OomPanic(String),
    AdapterError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub tau: TauValue,
    pub raw_output: String,
    pub token_cost: u64,
    pub adapter_kind: AdapterKind,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalFailedEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub reason: ProposalFailureReason,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPhaseCompletedEvent {
    pub task_id: TaskId,
    pub total_explorers: u32,
    pub successful: u32,
    pub failed: u32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_id: String,
    /// Predicate score [0,1]; 0 = total violation.
    pub score: f64,
    /// "Hard", "Soft", or "Advisory"
    pub severity_label: String,
    pub remediation_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchPrunedEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub reason: String,
    pub constraint_error_cost: RoleErrorCost,
    pub violated_constraints: Vec<ConstraintViolation>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroSurvivalEvent {
    pub task_id: TaskId,
    pub retry_count: u32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRequiredEvent {
    pub task_id: TaskId,
    pub max_role_error_cost: RoleErrorCost,
    pub timestamp: DateTime<Utc>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResolvedEvent {
    pub task_id: TaskId,
    pub resolved_output: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFailedEvent {
    pub task_id: TaskId,
    pub pruned_events: Vec<BranchPrunedEvent>,
    pub topologies_tried: Vec<TopologyKind>,
    pub tau_values_tried: Vec<Vec<f64>>,
    pub multiplication_condition_failure: Option<MultiplicationConditionFailure>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewGateTriggeredEvent {
    pub task_id: TaskId,
    pub gate_id: String,
    pub blocked_explorer_id: ExplorerId,
    pub reviewer_explorer_id: ExplorerId,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewGateBlockedEvent {
    pub task_id: TaskId,
    pub gate_id: String,
    pub blocked_explorer_id: ExplorerId,
    pub reviewer_explorer_id: ExplorerId,
    pub rejection_reason: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceSaturationWarningEvent {
    pub task_id: TaskId,
    pub active_subtasks: u32,
    pub interface_n_max: f64,
    pub saturation_ratio: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaoIterationEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub turn: u8,
    pub observation: String,
    pub passed: bool,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationScoredEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub score: f64,
    pub reason: String,
    pub passed: bool,
    pub timestamp: DateTime<Utc>,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskStartedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub subtask_id: SubtaskId,
    pub description: String,
    pub wave: usize,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskCompletedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub subtask_id: SubtaskId,
    pub token_cost: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "payload")]
pub enum H2AIEvent {
    CalibrationCompleted(CalibrationCompletedEvent),
    TaskBootstrapped(TaskBootstrappedEvent),
    TopologyProvisioned(TopologyProvisionedEvent),
    MultiplicationConditionFailed(MultiplicationConditionFailedEvent),
    Proposal(ProposalEvent),
    ProposalFailed(ProposalFailedEvent),
    GenerationPhaseCompleted(GenerationPhaseCompletedEvent),
    ReviewGateTriggered(ReviewGateTriggeredEvent),
    ReviewGateBlocked(ReviewGateBlockedEvent),
    Validation(ValidationEvent),
    BranchPruned(BranchPrunedEvent),
    ZeroSurvival(ZeroSurvivalEvent),
    InterfaceSaturationWarning(InterfaceSaturationWarningEvent),
    ConsensusRequired(ConsensusRequiredEvent),
    SemilatticeCompiled(SemilatticeCompiledEvent),
    MergeResolved(MergeResolvedEvent),
    TaskFailed(TaskFailedEvent),
    TaoIteration(TaoIterationEvent),
    VerificationScored(VerificationScoredEvent),
    SubtaskPlanCreated(SubtaskPlanCreatedEvent),
    SubtaskPlanReviewed(SubtaskPlanReviewedEvent),
    SubtaskStarted(SubtaskStartedEvent),
    SubtaskCompleted(SubtaskCompletedEvent),
}

impl H2AIEvent {
    pub fn subject(&self, task_id: &TaskId) -> String {
        format!("h2ai.tasks.{}", task_id)
    }
}
