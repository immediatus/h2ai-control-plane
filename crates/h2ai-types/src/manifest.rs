use crate::config::{ParetoWeights, ReviewGate, RoleSpec};
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
    /// Example: `"You are a skeptical senior engineer reviewing a proposed solution."`
    #[serde(default)]
    pub role_frame: String,
    /// CoT instruction style prepended to the task description.
    #[serde(default)]
    pub cot_style: CotStyle,
}

impl ExplorerSlotConfig {
    /// Returns a set of four structurally diverse slot configs for N≤4 ensembles.
    ///
    /// Covers the four most decorrelated reasoning strategies. For N>4, configs
    /// repeat modulo 4 — diversity saturates at 4 distinct strategies.
    pub fn diverse_defaults() -> Vec<ExplorerSlotConfig> {
        vec![
            ExplorerSlotConfig {
                role_frame: String::new(),
                cot_style: CotStyle::StepByStep,
            },
            ExplorerSlotConfig {
                role_frame: String::new(),
                cot_style: CotStyle::DevilsAdvocate,
            },
            ExplorerSlotConfig {
                role_frame: String::new(),
                cot_style: CotStyle::FirstPrinciples,
            },
            ExplorerSlotConfig {
                role_frame: String::new(),
                cot_style: CotStyle::BackwardChaining,
            },
        ]
    }
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
    /// Per-slot prompt strategy overrides. Slot `i` uses `slot_configs[i % len]`.
    /// When empty, all slots use default framing (no role_frame, no CoT prefix).
    /// Populate with `ExplorerSlotConfig::diverse_defaults()` to activate
    /// structural prompt diversity for correlated-hallucination mitigation.
    #[serde(default)]
    pub slot_configs: Vec<ExplorerSlotConfig>,
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
