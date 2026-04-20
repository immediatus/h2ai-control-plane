use crate::config::{ParetoWeights, ReviewGate, RoleSpec};
use serde::{Deserialize, Serialize};

/// POST /tasks request body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskManifest {
    pub description: String,
    pub pareto_weights: ParetoWeights,
    pub topology: TopologyRequest,
    pub explorers: ExplorerRequest,
    #[serde(default)]
    pub constraints: Vec<String>,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyRequest {
    /// "auto" | "ensemble" | "hierarchical_tree" | "team_swarm_hybrid"
    #[serde(default = "default_topology_kind")]
    pub kind: String,
    pub branching_factor: Option<u8>,
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
}

/// POST /tasks 202 response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAccepted {
    pub task_id: String,
    pub status: String,
    pub events_url: String,
    pub j_eff: f64,
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
