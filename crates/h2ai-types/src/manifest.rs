use crate::config::{AgentRole, ParetoWeights, ReviewGate, RoleSpec};
use crate::identity::TenantId;
use crate::sizing::OracleSpec;
use serde::{Deserialize, Serialize};

/// Chain-of-thought reasoning style injected as a per-slot instruction prefix.
///
/// Different styles impose structurally different reasoning paths, making
/// simultaneous failures statistically less likely across slots even when the
/// same underlying model is used. This is the primary structural mitigation for
/// the correlated-hallucination failure mode: when N slots all receive the same
/// prompt, correlated errors propagate to all proposals; style diversity breaks
/// that correlation by forcing different cognitive paths through the same problem.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CotStyle {
    /// No prefix — standard model behavior.
    #[default]
    None,
    /// "Think step by step" — elicits sequential decomposition.
    StepByStep,
    /// Start from the desired outcome and work backward.
    BackwardChaining,
    /// Break to fundamental principles; avoid analogies and convention.
    FirstPrinciples,
    /// Identify the strongest objections and failure modes first, then resolve them.
    DevilsAdvocate,
}

impl CotStyle {
    pub fn instruction(&self) -> &'static str {
        match self {
            CotStyle::None => "",
            CotStyle::StepByStep => {
                "Think step by step before giving your final answer. Show your reasoning explicitly."
            }
            CotStyle::BackwardChaining => {
                "Start from the desired outcome and work backward: what conditions, \
                 steps, and preconditions are required to achieve it?"
            }
            CotStyle::FirstPrinciples => {
                "Break this problem down to its fundamental principles. Do not rely on \
                 analogies, conventions, or prior patterns. Reason from first principles."
            }
            CotStyle::DevilsAdvocate => {
                "Before proposing a solution, identify the strongest objections and \
                 most likely failure modes. Build your answer to address these directly."
            }
        }
    }
}

/// Per-slot prompt strategy configuration.
///
/// When `slot_configs` is populated in `ExplorerRequest`, slot `i` uses
/// `slot_configs[i % slot_configs.len()]`. Empty vec → all slots use the
/// default `ComputeRequest` with no framing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplorerSlotConfig {
    /// Role framing injected at the start of the system context for this slot.
    #[serde(default)]
    pub role_frame: String,
    /// CoT instruction style prepended to the task description.
    #[serde(default)]
    pub cot_style: CotStyle,
    /// What constraint domains this slot is responsible for covering.
    /// Injected as "[MANDATE]: ..." after role_frame when non-empty.
    #[serde(default)]
    pub focus_mandate: String,
    /// The specific failure mode this slot should actively try to find.
    /// Injected as "[FIND]: ..." after role_frame when non-empty.
    #[serde(default)]
    pub rejection_criteria: String,
    /// Constraint corpus domain tags this slot covers.
    /// Assigned by the decomposition LLM; used to compute the C3 domain coverage score.
    /// Empty when the decomposition predates this field or corpus has no domain tags.
    #[serde(default)]
    pub constraint_domains: Vec<String>,
    /// When `true`, this slot runs a researcher pre-step before generating proposals.
    /// The researcher fetches current state-of-the-art grounding and injects it into
    /// the explorer prompt. Assigned by the decomposition LLM for tasks requiring
    /// current external knowledge (library versions, security advisories, regulations).
    #[serde(default)]
    pub search_enabled: bool,
    /// Knowledge profile role for this slot. Determines RAPTOR retrieval mode,
    /// PPR hops, and domain-tag filtering. Defaults to Executor when unset.
    #[serde(default)]
    pub agent_role: AgentRole,
}

/// POST /tasks request body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskManifest {
    pub description: String,
    pub pareto_weights: ParetoWeights,
    #[serde(default)]
    pub topology: TopologyRequest,
    pub explorers: ExplorerRequest,
    #[serde(default)]
    pub constraints: Vec<String>,
    pub context: Option<String>,
    #[serde(default)]
    pub oracle: Option<OracleSpec>,
    /// When `true`, operator requires human review before output is delivered,
    /// regardless of q_confidence. Defaults to `false`.
    #[serde(default)]
    pub require_approval: bool,
    /// Domain tags for automatic constraint routing via the wiki context map.
    ///
    /// Example: `["eu_data", "financial_report"]` causes the wiki to resolve
    /// all constraints mapped to those domains. Explicit `constraints` IDs are
    /// always included regardless of tags.
    #[serde(default)]
    pub constraint_tags: Vec<String>,
    /// When `true`, run both standard and adversarial verifiers and emit
    /// `VerifierComparisonEvent` for every scored proposal. Does NOT affect
    /// pruning decisions — the standard score always wins. Off by default;
    /// enable only for A/B measurement runs.
    #[serde(default)]
    pub measure_verifier_ab: bool,
    /// Tenant scope for this task. Defaults to `TenantId::default_tenant()` when absent.
    #[serde(default)]
    pub tenant_id: TenantId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyRequest {
    /// "auto" | "ensemble" | "hierarchical_tree" | "team_swarm_hybrid"
    #[serde(default = "default_topology_kind")]
    pub kind: String,
    pub branching_factor: Option<u8>,
}

impl Default for TopologyRequest {
    fn default() -> Self {
        Self {
            kind: default_topology_kind(),
            branching_factor: None,
        }
    }
}

fn default_topology_kind() -> String {
    "auto".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerRequest {
    pub count: usize,
    pub tau_min: Option<f64>,
    pub tau_max: Option<f64>,
    #[serde(default)]
    pub roles: Vec<RoleSpec>,
    #[serde(default)]
    pub review_gates: Vec<ReviewGate>,
    /// Additional slots appended to the LLM-derived decomposition result.
    /// When non-empty, these slots are merged with the decomposition output and the
    /// combined set is pruned by orthogonality to N_max. They add operator context —
    /// they do NOT bypass decomposition. Leave empty unless adding task-specific experts.
    #[serde(default)]
    pub slot_configs: Vec<ExplorerSlotConfig>,
    /// Adapter diversity IDs for this task. Each element maps to `pool[id % pool.len()]`.
    /// When empty, defaults to `[0, 1, ..., count-1]` (one distinct slot per explorer).
    /// Example: `[1, 2, 3]` → 3 explorers, each on a different pool adapter (if pool.len() ≥ 3).
    /// Example: `[1, 1, 2, 2, 3]` → 5 explorers, IDs 1 and 2 run twice.
    #[serde(default)]
    pub diversity_ids: Vec<u32>,
}

/// POST /tasks 202 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAccepted {
    pub task_id: String,
    pub status: String,
    pub events_url: String,
    pub topology_kind: String,
    pub n_max: f64,
    pub interface_n_max: Option<f64>,
}

/// GET /tasks/{id} response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatusResponse {
    pub task_id: String,
    pub status: String,
    pub phase: u8,
    pub phase_name: String,
    pub explorers_completed: u32,
    pub explorers_total: u32,
    pub proposals_valid: u32,
    pub proposals_pruned: u32,
    pub autonomic_retries: u32,
}

/// POST /tasks/{id}/merge request body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRequest {
    pub resolution: MergeResolution,
    #[serde(default)]
    pub selected_proposals: Vec<String>,
    pub synthesis_notes: Option<String>,
    pub final_output: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeResolution {
    Select,
    Synthesize,
    Reject,
}

/// POST /calibrate 202 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationAccepted {
    pub calibration_id: String,
    pub status: String,
    pub events_url: String,
    pub adapter_count: usize,
}
