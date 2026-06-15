use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGapRecord {
    pub constraint_id: String,
    pub check_idx: usize,
    pub incorrect_concept: String,
    pub gap_query: String,
    pub pass_rate_across_waves: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSynthesis {
    pub check_id: (String, usize),
    pub incorrect_pattern: String,
    pub correct_pattern: String,
    pub mechanistic_reason: String,
    pub source: Option<String>,
    pub confidence: f64,
    /// Wave index at which this synthesis was first injected into proposal generation context.
    #[serde(default)]
    pub injected_at_wave: Option<u32>,
    /// Mean per-check pass rate immediately before this synthesis was injected.
    #[serde(default)]
    pub pre_injection_pass_rate: Option<f64>,
    /// Mean per-check pass rate for each wave after injection.
    #[serde(default)]
    pub post_injection_pass_rates: Vec<f64>,
}
