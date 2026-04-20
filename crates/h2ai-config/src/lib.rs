use h2ai_types::config::AdapterKind;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// A named adapter configuration entry. Operators define a list of profiles in
/// H2AIConfig; callers reference them by name to avoid scattering AdapterKind
/// values throughout application startup code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterProfile {
    pub name: String,
    pub kind: AdapterKind,
}

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
    /// Byzantine fault tolerance bound for Krum/Multi-Krum.
    /// 0 = Krum disabled (ConsensusMedian used instead at high error costs).
    /// n = tolerate up to n Byzantine explorers; requires at least 2n+3 explorers.
    #[serde(default)]
    pub krum_fault_tolerance: usize,
    /// Role error cost threshold above which Krum is preferred over ConsensusMedian.
    /// Only has effect when krum_fault_tolerance > 0.
    #[serde(default = "default_krum_threshold")]
    pub krum_threshold: f64,
    pub coordination_threshold_max: f64,
    pub min_baseline_competence: f64,
    pub max_error_correlation: f64,
    // Agent role default τ values
    pub tau_coordinator: f64,
    pub tau_executor: f64,
    pub tau_evaluator: f64,
    pub tau_synthesizer: f64,
    // Agent role default error-cost values
    pub cost_coordinator: f64,
    pub cost_executor: f64,
    pub cost_evaluator: f64,
    pub cost_synthesizer: f64,
    // Context compaction budget (None = no limit)
    pub max_context_tokens: Option<usize>,
    #[serde(default = "default_explorer_max_tokens")]
    pub explorer_max_tokens: u64,
    #[serde(default = "default_calibration_max_tokens")]
    pub calibration_max_tokens: u64,
    #[serde(default = "default_calibration_tau")]
    pub calibration_tau: f64,
    #[serde(default = "default_optimizer_threshold_step")]
    pub optimizer_threshold_step: f64,
    #[serde(default = "default_optimizer_threshold_floor")]
    pub optimizer_threshold_floor: f64,
    #[serde(default = "default_max_autonomic_retries")]
    pub max_autonomic_retries: u32,
    /// α contention constant: fraction of work that must serialize regardless of
    /// parallelism. This is a calibration heuristic — set per deployment based on
    /// observed adapter coordination overhead. Default 0.12 matches typical LLM workloads.
    #[serde(
        default = "default_alpha_contention",
        alias = "calibration_alpha_single_adapter"
    )]
    pub alpha_contention: f64,
    #[serde(default = "default_calibration_cg_fallback")]
    pub calibration_cg_fallback: f64,
    /// β₀ (beta_base_default) — base coherency cost per agent pair for this deployment tier.
    /// Used as calibration fallback when fewer than 3 adapters are available.
    /// Default 0.01 = AI-agents tier (blog calibration table: α=0.15, β₀=0.01, CG=0.4 → N_max≈6).
    /// Use 0.005 for human-team tier, 0.0003 for CPU-core tier.
    #[serde(default = "default_beta_base", alias = "kappa_eff_factor")]
    pub beta_base_default: f64,
    #[serde(default = "default_tao_per_turn_factor")]
    pub tao_per_turn_factor: f64,
    /// Jaccard threshold above which all pairwise proposals are considered uniformly
    /// hallucinated → ZeroSurvival → MAPE-K retry. Default 0.85 (active).
    #[serde(default = "default_diversity_threshold")]
    pub diversity_threshold: f64,
    #[serde(default)]
    pub task_deadline_secs: Option<u64>,
    /// Maximum number of tasks that may execute concurrently. Requests beyond this
    /// limit receive 503 Service Unavailable. Default 8.
    #[serde(default = "default_max_concurrent_tasks")]
    pub max_concurrent_tasks: usize,
    /// Named adapter profiles. Reference by name via AdapterFactory::build_from_profiles.
    /// Names must be unique; build_from_profiles returns the first match — duplicate names
    /// cause the second entry to be silently ignored.
    #[serde(default)]
    pub adapter_profiles: Vec<AdapterProfile>,
    /// If non-zero, overrides the CG-mean–derived accuracy proxy (`0.5 + CG_mean / 2`)
    /// with a directly measured per-adapter baseline accuracy.
    /// Set by running `scripts/baseline_eval.py` and pasting the result.
    /// A value of 0.0 means "use the CG-mean proxy".
    #[serde(default)]
    pub baseline_accuracy_proxy: f64,
}

fn default_explorer_max_tokens() -> u64 {
    1024
}
fn default_calibration_max_tokens() -> u64 {
    256
}
fn default_calibration_tau() -> f64 {
    0.5
}
fn default_optimizer_threshold_step() -> f64 {
    0.1
}
fn default_optimizer_threshold_floor() -> f64 {
    0.3
}
fn default_max_autonomic_retries() -> u32 {
    2
}
fn default_alpha_contention() -> f64 {
    0.12
}
fn default_calibration_cg_fallback() -> f64 {
    0.7
}
fn default_beta_base() -> f64 {
    0.01
}
fn default_tao_per_turn_factor() -> f64 {
    0.6
}
fn default_diversity_threshold() -> f64 {
    0.85
}
fn default_krum_threshold() -> f64 {
    0.95
}
fn default_max_concurrent_tasks() -> usize {
    8
}

impl Default for H2AIConfig {
    fn default() -> Self {
        Self {
            j_eff_gate: 0.4,
            bft_threshold: 0.85,
            krum_fault_tolerance: 0,
            krum_threshold: 0.95,
            coordination_threshold_max: 0.3,
            min_baseline_competence: 0.3,
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
            max_autonomic_retries: 2,
            alpha_contention: 0.12,
            calibration_cg_fallback: 0.7,
            beta_base_default: 0.01,
            tao_per_turn_factor: 0.6,
            diversity_threshold: 0.85,
            task_deadline_secs: None,
            max_concurrent_tasks: 8,
            adapter_profiles: Vec::new(),
            baseline_accuracy_proxy: 0.0,
        }
    }
}

impl H2AIConfig {
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigLoadError> {
        let contents = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&contents)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_base_default_loads_from_kappa_eff_factor_alias() {
        let json = r#"{"j_eff_gate":0.4,"bft_threshold":0.85,"coordination_threshold_max":0.3,"min_baseline_competence":0.3,"max_error_correlation":0.9,"tau_coordinator":0.05,"tau_executor":0.4,"tau_evaluator":0.1,"tau_synthesizer":0.8,"cost_coordinator":0.1,"cost_executor":0.5,"cost_evaluator":0.9,"cost_synthesizer":0.1,"kappa_eff_factor":0.019}"#;
        let cfg: H2AIConfig = serde_json::from_str(json).unwrap();
        assert!(
            (cfg.beta_base_default - 0.019).abs() < 1e-10,
            "kappa_eff_factor alias must deserialize into beta_base_default, got {}",
            cfg.beta_base_default
        );
    }
}
