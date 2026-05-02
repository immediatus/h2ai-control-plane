pub mod prompts;

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
    #[error("config error: {0}")]
    Config(#[from] config::ConfigError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct H2AIConfig {
    pub j_eff_gate: f64,
    pub bft_threshold: f64,
    pub krum_fault_tolerance: usize,
    pub krum_threshold: f64,
    pub coordination_threshold_max: f64,
    pub min_baseline_competence: f64,
    pub max_error_correlation: f64,
    pub tau_coordinator: f64,
    pub tau_executor: f64,
    pub tau_evaluator: f64,
    pub tau_synthesizer: f64,
    pub cost_coordinator: f64,
    pub cost_executor: f64,
    pub cost_evaluator: f64,
    pub cost_synthesizer: f64,
    /// None = no limit. Omit from override file to leave unlimited.
    #[serde(default)]
    pub max_context_tokens: Option<usize>,
    pub explorer_max_tokens: u64,
    pub calibration_max_tokens: u64,
    pub calibration_tau: f64,
    pub optimizer_threshold_step: f64,
    pub optimizer_threshold_floor: f64,
    pub max_autonomic_retries: u32,
    #[serde(alias = "calibration_alpha_single_adapter")]
    pub alpha_contention: f64,
    pub calibration_cg_fallback: f64,
    #[serde(alias = "kappa_eff_factor")]
    pub beta_base_default: f64,
    pub tao_per_turn_factor: f64,
    pub diversity_threshold: f64,
    /// None = no deadline. Omit from override file to leave unlimited.
    #[serde(default)]
    pub task_deadline_secs: Option<u64>,
    pub max_concurrent_tasks: usize,
    pub adapter_profiles: Vec<AdapterProfile>,
    pub context_pressure_gamma: f64,
    pub baseline_accuracy_proxy: f64,
    pub calibration_adapter_count: usize,
    pub calibration_tau_spread: [f64; 2],
    pub cg_collapse_threshold: f64,
    pub cg_agreement_threshold: f64,
    pub embedding_model_name: EmbeddingModelName,
    pub bandit_phase0_k: u32,
    pub bandit_phase1_k: u32,
    pub bandit_epsilon: f64,
    pub bandit_soft_reset_decay: f64,
    pub tau_spread_max_factor: f64,
    pub auto_baseline_eval: bool,
    pub auto_baseline_eval_min_tasks: u32,
    pub allow_single_family: bool,
    pub optimizer_waste_threshold: f64,
    pub scheduler_policy: SchedulerPolicy,
    pub scheduler_spillover_threshold: usize,
    /// Byte length above which system_context is offloaded to the payload store.
    /// Default 524288 (512 KB) — half of NATS 1 MB default limit.
    pub payload_offload_threshold_bytes: usize,
    /// Events published per task before a state snapshot is written to NATS KV.
    /// Reduces crash-recovery replay time. Default 50. 0 disables snapshotting.
    pub snapshot_interval_events: usize,
    /// NATS server URL used by the API server, agent binary, and integration tests.
    pub nats_url: String,
}

/// Agent scheduling policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchedulerPolicy {
    /// Route to the lowest cost tier that has headroom below `scheduler_spillover_threshold`.
    /// Spills to the next tier when all agents in the preferred tier are saturated.
    /// Falls back to globally least-loaded when every tier is saturated.
    #[default]
    CostAwareSpillover,
    /// Original policy: cheapest tier always wins regardless of queue depth.
    LeastLoaded,
}

/// Embedding model selection for `FastEmbedModel`.
///
/// All variants are supported by fastembed-rs and downloaded to `~/.cache/fastembed/` on
/// first use. Models are L2-normalised; cosine similarity equals dot product.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EmbeddingModelName {
    /// sentence-transformers/all-MiniLM-L6-v2 — 22 MB, ~8 ms/sentence CPU.
    /// Default: smallest footprint, good STS performance.
    #[default]
    AllMiniLmL6V2,
    /// BAAI/bge-small-en-v1.5 — 109 MB, ~5 ms/sentence CPU.
    /// Better MTEB STS scores than AllMiniLmL6V2; recommended for production deployments.
    BgeSmallEnV1_5,
}

impl Default for H2AIConfig {
    fn default() -> Self {
        Self::load_layered(None).expect("embedded reference.toml is always valid")
    }
}

impl H2AIConfig {
    /// Load configuration using the three-layer stack (later layers win):
    ///
    /// 1. Embedded `reference.toml` — all defaults, always present
    /// 2. `override_path` file — operator-provided TOML with only changed fields
    /// 3. `H2AI__<FIELD>` env vars — highest priority, per-field overrides
    ///
    /// Returns `Err` if `override_path` is `Some` but the file does not exist or
    /// contains invalid TOML, or if a field has a wrong type.
    pub fn load_layered(override_path: Option<&Path>) -> Result<Self, ConfigLoadError> {
        use config::{Config, Environment, File, FileFormat};

        let mut builder = Config::builder().add_source(File::from_str(
            include_str!("../reference.toml"),
            FileFormat::Toml,
        ));

        if let Some(path) = override_path {
            builder = builder.add_source(File::from(path).required(true));
        }

        builder = builder.add_source(
            Environment::with_prefix("H2AI")
                .separator("__")
                .try_parsing(true),
        );

        Ok(builder.build()?.try_deserialize()?)
    }

    /// Load configuration from a complete JSON file.
    ///
    /// Unlike `load_layered`, this does NOT merge with `reference.toml` — the JSON
    /// must contain all required fields. Partial JSON will fail deserialization.
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigLoadError> {
        let contents = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&contents)?)
    }
}
