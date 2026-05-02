pub mod prompts;

use h2ai_types::config::AdapterKind;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Named adapter configuration entry used for TaskProfile routing.
///
/// Operators populate `H2AIConfig::adapter_profiles` with these entries so that
/// model backends are configured once and referenced by name throughout the
/// application, avoiding scattered `AdapterKind` values at startup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterProfile {
    /// Unique human-readable identifier for this profile (e.g. `"claude-sonnet"`).
    pub name: String,
    /// Backend kind and its connection parameters (API key env var, model string, etc.).
    pub kind: AdapterKind,
}

/// Error returned by `H2AIConfig::load_layered` and `H2AIConfig::load_from_file`.
#[derive(Debug, Error)]
pub enum ConfigLoadError {
    /// File could not be read from disk.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON content failed to deserialize into `H2AIConfig`.
    #[error("JSON parse error: {0}")]
    Parse(#[from] serde_json::Error),
    /// TOML parsing or field-type mismatch in the layered config stack.
    #[error("config error: {0}")]
    Config(#[from] config::ConfigError),
}

/// Single configuration authority for the H2AI Control Plane runtime.
///
/// All fields are populated by `load_layered()`, which merges the embedded
/// `reference.toml` defaults, an optional operator override file, and
/// `H2AI__<FIELD>` environment variables (highest priority wins).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct H2AIConfig {
    /// Minimum Jaccard overlap between task description and constraint vocabulary before a task is accepted. Range [0, 1]; lower values are more permissive.
    pub j_eff_gate: f64,
    /// BFT consensus threshold — fraction of agents that must agree for a result to be accepted. Range [0, 1].
    pub bft_threshold: f64,
    /// Number of Byzantine explorers Krum/Multi-Krum will tolerate; `0` disables Krum and falls back to `ConsensusMedian`. Requires at least `2n+3` agents when nonzero.
    pub krum_fault_tolerance: usize,
    /// Role error-cost threshold above which Krum is preferred over `ConsensusMedian`. Only relevant when `krum_fault_tolerance > 0`.
    pub krum_threshold: f64,
    /// Maximum coordination threshold θ derived from calibration. Range [0, 1].
    pub coordination_threshold_max: f64,
    /// Minimum baseline competence p₀ required to pass the multiplication condition. Range [0, 1].
    pub min_baseline_competence: f64,
    /// Maximum error correlation ρ tolerated before the multiplication condition fails. Range [0, 1].
    pub max_error_correlation: f64,
    /// Default temperature τ for the coordinator role; controls output diversity. Range [0, 1].
    pub tau_coordinator: f64,
    /// Default temperature τ for the executor role; controls output diversity. Range [0, 1].
    pub tau_executor: f64,
    /// Default temperature τ for the evaluator role; controls output diversity. Range [0, 1].
    pub tau_evaluator: f64,
    /// Default temperature τ for the synthesizer role; controls output diversity. Range [0, 1].
    pub tau_synthesizer: f64,
    /// Semilattice merge weight for coordinator role errors. Range [0, 1]; higher = penalised more.
    pub cost_coordinator: f64,
    /// Semilattice merge weight for executor role errors. Range [0, 1]; higher = penalised more.
    pub cost_executor: f64,
    /// Semilattice merge weight for evaluator role errors. Range [0, 1]; higher = penalised more.
    pub cost_evaluator: f64,
    /// Semilattice merge weight for synthesizer role errors. Range [0, 1]; higher = penalised more.
    pub cost_synthesizer: f64,
    /// Maximum tokens kept after context compaction. `None` means no limit; omit from the override file to leave unlimited.
    #[serde(default)]
    pub max_context_tokens: Option<usize>,
    /// Maximum tokens per explorer generation call.
    pub explorer_max_tokens: u64,
    /// Maximum tokens per calibration probe call.
    pub calibration_max_tokens: u64,
    /// Temperature used for all calibration adapter probes. Range [0, 1].
    pub calibration_tau: f64,
    /// Step size for `verify_threshold` reduction suggestions from the self-optimizer. Range (0, 1).
    pub optimizer_threshold_step: f64,
    /// Floor for `verify_threshold` reductions; the threshold never drops below this value.
    pub optimizer_threshold_floor: f64,
    /// Maximum autonomic MAPE-K retry iterations before a task is declared failed.
    pub max_autonomic_retries: u32,
    /// USL α contention constant: fraction of work that must serialise regardless of parallelism. Also accepts the alias `calibration_alpha_single_adapter`.
    #[serde(alias = "calibration_alpha_single_adapter")]
    pub alpha_contention: f64,
    /// CG (coordination gain) fallback value used when fewer than 3 adapters ran calibration.
    pub calibration_cg_fallback: f64,
    /// USL β₀ base coherency cost per agent pair; deployment-tier specific (e.g. `0.039` for AI agents). Also accepts the alias `kappa_eff_factor`.
    #[serde(alias = "kappa_eff_factor")]
    pub beta_base_default: f64,
    /// Quality factor gained per TAO loop turn; heuristic prior that converges after ~20 Tier 1 oracle tasks.
    pub tao_per_turn_factor: f64,
    /// EMA smoothing factor α for `TaoMultiplierEstimator` drift tracking. Smaller values weight history more; half-life ≈ ln(2) / α samples.
    pub tao_estimator_ema_alpha: f64,
    /// Jaccard similarity threshold above which all pairwise proposals are treated as uniformly hallucinated, triggering a MAPE-K retry. Range [0, 1].
    pub diversity_threshold: f64,
    /// Hard deadline in seconds for a single task end-to-end. `None` means no deadline; omit from the override file to leave unlimited.
    #[serde(default)]
    pub task_deadline_secs: Option<u64>,
    /// Maximum number of concurrent task executions; requests beyond this limit receive HTTP 503.
    pub max_concurrent_tasks: usize,
    /// Named adapter profiles available for TaskProfile routing.
    pub adapter_profiles: Vec<AdapterProfile>,
    /// Context pressure sensitivity γ: scales how much a full context window raises β. `0` disables the effect; `0.5` doubles β at 100% context fill. Range [0, 1].
    pub context_pressure_gamma: f64,
    /// Per-adapter baseline accuracy proxy. `0.0` uses the CG-mean proxy (`0.5 + CG_mean / 2`); set to an empirically measured value via `scripts/baseline_eval.py`.
    pub baseline_accuracy_proxy: f64,
    /// Number of adapter instances spawned during calibration. Minimum 3 for a valid USL two-point fit; fewer falls back to `alpha_contention` and `beta_base_default`.
    pub calibration_adapter_count: usize,
    /// τ spread `[min, max]` for calibration instances; instances are spaced linearly across this range. The spread may expand up to `tau_spread_max_factor` when Talagrand detects over-confidence.
    pub calibration_tau_spread: [f64; 2],
    /// CG collapse threshold: when CG_embed drops below this value the planner forces N_max = 1. Default `0.10` — below 10 % pairwise reconciliation is undefined.
    pub cg_collapse_threshold: f64,
    /// Cosine similarity threshold for counting two adapter outputs as "in agreement" when computing CG_embed.
    pub cg_agreement_threshold: f64,
    /// Embedding model used for CG cosine agreement measurement; requires the `fastembed-embed` Cargo feature.
    pub embedding_model_name: EmbeddingModelName,
    /// Tasks completed before activating the bandit (Phase 0 — pure exploration); during Phase 0 N = N_max_USL unconditionally.
    pub bandit_phase0_k: u32,
    /// Tasks completed before switching from ε-greedy to pure Thompson Sampling (Phase 1).
    pub bandit_phase1_k: u32,
    /// ε for Phase 1 ε-greedy: probability of selecting a random arm each task. Range [0, 1].
    pub bandit_epsilon: f64,
    /// Soft-reset decay factor applied to the learned posterior when the adapter version hash changes. `0.3` blends 30 % toward the initial prior.
    pub bandit_soft_reset_decay: f64,
    /// Maximum τ-spread expansion factor when Talagrand detects over-confidence. `2.0` means the spread can at most double.
    pub tau_spread_max_factor: f64,
    /// When `true`, automatically switches to the Empirical prediction basis after `auto_baseline_eval_min_tasks` Tier 1 oracle tasks complete.
    pub auto_baseline_eval: bool,
    /// Minimum Tier 1 oracle task count before automatic baseline evaluation triggers.
    pub auto_baseline_eval_min_tasks: u32,
    /// When `false` (default), calibration aborts if all non-Mock adapters share the same provider family; set `true` to allow single-family pools with a warning.
    pub allow_single_family: bool,
    /// Fraction of proposals that must survive verification for a run to be considered non-wasteful; below this threshold the self-optimizer suggests reducing the `verify_threshold`.
    pub optimizer_waste_threshold: f64,
    /// Agent dispatch scheduling policy.
    pub scheduler_policy: SchedulerPolicy,
    /// Queue depth per cost tier at which `CostAwareSpillover` routes to the next tier.
    pub scheduler_spillover_threshold: usize,
    /// Byte length above which system_context is offloaded to the payload store. Default 524288 (512 KB) — half of NATS 1 MB default limit.
    pub payload_offload_threshold_bytes: usize,
    /// Events published per task before a state snapshot is written to NATS KV. Reduces crash-recovery replay time. Default 50. 0 disables snapshotting.
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
