use crate::config::{
    AdapterKind, AuditorConfig, ExplorerConfig, ParetoWeights, ReviewGate, TopologyKind,
};
use crate::identity::{ExplorerId, TaskId};
use crate::physics::{
    CoherencyCoefficients, CoordinationThreshold, MergeStrategy, MultiplicationConditionFailure,
    RoleErrorCost,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCompletedEvent {
    pub calibration_id: TaskId,
    pub coefficients: CoherencyCoefficients,
    pub coordination_threshold: CoordinationThreshold,
    pub timestamp: DateTime<Utc>,
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
    pub kappa_eff: f64,
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
    pub tau: f64,
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
pub struct BranchPrunedEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub reason: String,
    pub constraint_error_cost: RoleErrorCost,
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
}

impl H2AIEvent {
    pub fn subject(&self, task_id: &TaskId) -> String {
        format!("h2ai.tasks.{}", task_id)
    }
}
