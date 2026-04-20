use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigLoadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Parse(#[from] serde_json::Error),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct H2AIConfig {
    // Context compiler gate
    pub j_eff_gate: f64,
    // Physics thresholds
    pub bft_threshold: f64,
    pub coordination_threshold_max: f64,
    pub min_baseline_competence: f64,
    pub max_error_correlation: f64,
    // Agent role default τ (temperature/creativity) values
    pub tau_coordinator: f64,
    pub tau_executor: f64,
    pub tau_evaluator: f64,
    pub tau_synthesizer: f64,
    // Agent role default error-cost values
    pub cost_coordinator: f64,
    pub cost_executor: f64,
    pub cost_evaluator: f64,
    pub cost_synthesizer: f64,
    // Context compaction budget (None = no compaction limit)
    pub max_context_tokens: Option<usize>,
    // Explorer generation token budget
    #[serde(default = "default_explorer_max_tokens")]
    pub explorer_max_tokens: u64,
    // Calibration probe token budget
    #[serde(default = "default_calibration_max_tokens")]
    pub calibration_max_tokens: u64,
    // Calibration probe tau (temperature for diversity during measurement)
    #[serde(default = "default_calibration_tau")]
    pub calibration_tau: f64,
    // SelfOptimizer: how much to lower verify_threshold per step
    #[serde(default = "default_optimizer_threshold_step")]
    pub optimizer_threshold_step: f64,
    // SelfOptimizer: minimum verify_threshold floor
    #[serde(default = "default_optimizer_threshold_floor")]
    pub optimizer_threshold_floor: f64,
}

fn default_explorer_max_tokens() -> u64 { 1024 }
fn default_calibration_max_tokens() -> u64 { 256 }
fn default_calibration_tau() -> f64 { 0.5 }
fn default_optimizer_threshold_step() -> f64 { 0.1 }
fn default_optimizer_threshold_floor() -> f64 { 0.3 }

impl Default for H2AIConfig {
    fn default() -> Self {
        Self {
            j_eff_gate: 0.4,
            bft_threshold: 0.85,
            coordination_threshold_max: 0.3,
            min_baseline_competence: 0.5,
            max_error_correlation: 0.9,
            tau_coordinator: 0.05,
            tau_executor: 0.40,
            tau_evaluator: 0.10,
            tau_synthesizer: 0.80,
            cost_coordinator: 0.1,
            cost_executor: 0.5,
            cost_evaluator: 0.9,
            cost_synthesizer: 0.1,
            max_context_tokens: None,
            explorer_max_tokens: 1024,
            calibration_max_tokens: 256,
            calibration_tau: 0.5,
            optimizer_threshold_step: 0.1,
            optimizer_threshold_floor: 0.3,
        }
    }
}

impl H2AIConfig {
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigLoadError> {
        let contents = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&contents)?;
        Ok(config)
    }
}
