pub mod prompts;

use h2ai_types::config::AdapterKind;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Configuration for Phase 1.5 task complexity assessment and quadrant routing.
///
/// All defaults are set in `reference.toml` under `[task_complexity]`.
/// Shadow mode (default: `true`) lets you collect routing data before the
/// GAP-A1 experiment validates the thresholds — ParetoRouter is unchanged until
/// shadow_mode is set to `false`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskComplexityConfig {
    /// When `true` (default): Phase 1.5 emits `TaskComplexityAssessedEvent` but
    /// `TopologyPlanner` ignores the quadrant — all tasks route as Coverage.
    /// Disable only after the GAP-A1 experiment validates threshold calibration.
    pub shadow_mode: bool,
    /// TCC below this threshold classifies the task as Precision (Self-MoA).
    pub tcc_precision_threshold: f64,
    /// TCC above this threshold classifies the task as Coverage (cross-family).
    pub tcc_coverage_threshold: f64,
    /// Soft-constraint weight coefficient in TCC_structural formula.
    pub k_soft: f64,
    /// Type-diversity coefficient in TCC_structural formula.
    pub k_type: f64,
    /// Interaction-term (soft_fraction × type_diversity) coefficient in TCC_structural.
    pub k_cross: f64,
    /// Heavy-fraction amplification: when `static_coverage < min_static_coverage_for_probe`,
    /// TCC_effective = tcc_structural × (1 + k_heavy × heavy_fraction).
    pub k_heavy: f64,
    /// Minimum static coverage fraction required to run the N-probe sampling.
    /// Corpora with `static_coverage < this` are treated as heavy-dominant (probe skipped).
    pub min_static_coverage_for_probe: f64,
    /// Number of mini-probe calls used to estimate TCC_empirical (ambiguous band only).
    /// Probe is skipped on unambiguous Precision/Coverage paths and heavy-dominant corpora.
    pub n_probe: usize,
    /// Pool N_eff threshold below which Coverage → Complex escalation occurs.
    pub n_eff_complex_threshold: f64,
    /// Max tokens per probe completion. Probe outputs are structure assessments, not
    /// full answers — 512 tokens is sufficient for static constraint evaluation.
    pub probe_max_tokens: u64,
    /// Temperature for probe completions. Mid-range τ produces varied but coherent
    /// outputs needed to generate an informative satisfaction matrix.
    pub probe_tau: f64,
    /// Minimum number of informative static constraints (≥1 pass AND ≥1 fail across
    /// probes) needed to compute TCC_empirical. Below this, eigendecomposition is
    /// degenerate; fall back to TCC_structural with heavy amplification.
    pub tcc_min_informative_constraints: usize,
    /// Penalty added to TCC_effective when TCC_structural > TCC_empirical + 1.0.
    /// Signals that the corpus is more complex than static probes detected — typically
    /// because Heavy-tier constraints dominate actual complexity. Routes toward Coverage.
    pub tcc_mismatch_penalty: f64,
    /// Probe n_eff threshold (as fraction of n_probe) for Coverage vs Complex routing.
    /// Tasks with probe n_eff below `neff_probe_min_fraction × n_probe` escalate to Complex.
    pub neff_probe_min_fraction: f64,
    /// Probe n_eff threshold (as fraction of n_probe) for Precision vs Degenerate routing.
    /// Tasks with probe n_eff below `neff_probe_warning_fraction × n_probe` → Degenerate.
    pub neff_probe_warning_fraction: f64,
}

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
/// `H2AI_<FIELD>` environment variables (highest priority wins).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct H2AIConfig {
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
    /// Minimum mean pairwise Hamming distance (on constraint-satisfaction fingerprints) required for the swarm to be considered diverse. Below this threshold the swarm is flagged as collectively hallucinated and a MAPE-K retry is triggered. Range [0, 1]. 0.0 disables the gate; recommended production value 0.15.
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
    /// CG collapse threshold: when CG_embed drops below this value the planner forces N_max = 1. Default `0.10` — below 10 % agent outputs are so divergent that coherence drag is unbounded.
    pub cg_collapse_threshold: f64,
    /// Cosine similarity threshold for counting two adapter outputs as "in agreement" when computing CG via embedding cosine (future; currently CG uses constraint-profile Hamming).
    pub cg_agreement_threshold: f64,
    /// Embedding model used for CG cosine agreement measurement; requires the `fastembed-embed` Cargo feature.
    pub embedding_model_name: EmbeddingModelName,
    /// Minimum N_eff increment required to include the next adapter in `EigenCalibration::n_pruned`.
    /// Adapter k is kept when adding it raises N_eff by ≥ this delta. Default 0.05.
    /// Increase toward 0.1–0.2 for calibrations with few adapters (N ≤ 4).
    pub eigen_n_eff_delta: f64,
    /// Minimum number of TAO loop samples before `TaoMultiplierEstimator` state is persisted
    /// to NATS. The EMA estimate is unreliable below this count. Default 20.
    /// Raise to 50–100 for high-variance task distributions.
    pub tao_estimator_warmup: usize,
    /// Initial N_max used to seed the Thompson Sampling bandit warm prior at first startup
    /// before any calibration result is available. Clamped to [1, 6] by the bandit. Default 4.
    pub bandit_n_max_initial: u32,
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
    /// Enable the synthesis phase. When false, the engine uses the selection chain exclusively.
    /// Default: true. Set false to reproduce pre-synthesis behavior for benchmarking.
    pub synthesis_enabled: bool,
    /// Minimum number of verified proposals required to attempt synthesis.
    /// Default: 2. Raising to 3+ reserves synthesis for richer ensembles.
    pub synthesis_min_proposals: usize,
    /// τ (temperature) for critique and synthesis calls. Lower than explorer τ
    /// encourages deterministic, structured critique output. Default: 0.2.
    pub synthesis_tau: f64,
    /// Max tokens for the critique call. Default: 1024.
    pub synthesis_critique_max_tokens: u64,
    /// Max tokens for the synthesis call. Default: 2048.
    pub synthesis_max_tokens: u64,
    /// Commands permitted in Normal-mode waves. Empty = unrestricted (unsafe in production).
    pub shell_allowlist: Vec<String>,
    /// Commands permitted in Hardened-mode waves. Should be a subset of `shell_allowlist`.
    pub shell_hardened_allowlist: Vec<String>,
    /// Maximum seconds a shell tool invocation may run before it is killed. Default: 5.
    pub shell_timeout_secs: u64,
    /// Maximum number of TAO loop tool-call iterations an edge agent may execute per task.
    /// After this limit the agent returns whatever output the LLM produced last. Default: 5.
    /// Valid range: 1–255. A value of 0 is rejected by the TaoAgent and treated as 1.
    pub agent_max_tool_iterations: u8,
    /// Google Custom Search configuration. Absent = WebSearch executor disabled.
    #[serde(default)]
    pub web_search: Option<WebSearchConfig>,
    /// MCP filesystem subprocess configuration. Absent = MCP executor disabled.
    #[serde(default)]
    pub mcp_filesystem: Option<McpFilesystemConfig>,
    /// WASM interpreter executor configuration. Absent = WASM executor disabled.
    #[serde(default)]
    pub wasm_executor: Option<WasmExecutorConfig>,
    /// Phase 1.5 task complexity assessment and quadrant routing configuration.
    pub task_complexity: TaskComplexityConfig,
    /// Human-in-the-loop approval gate configuration.
    pub hitl: HitlConfig,
    /// Constraint wiki configuration — tag routing, NATS KV storage.
    pub constraint_wiki: ConstraintWikiConfig,
}

/// Configuration for the WebSearch executor (Google Custom Search API).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSearchConfig {
    /// Name of the environment variable holding the Google Custom Search API key.
    pub api_key_env: String,
    /// Name of the environment variable holding the Google Custom Search Engine ID.
    pub cx_env: String,
    /// Maximum number of search result snippets returned to the LLM. Default: 3.
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

fn default_max_results() -> usize {
    3
}

/// Configuration for the MCP filesystem executor (stdio subprocess transport).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpFilesystemConfig {
    /// Binary to spawn for the MCP server (e.g. "npx").
    pub command: String,
    /// Arguments passed to the binary (e.g. ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]).
    pub args: Vec<String>,
    /// Seconds before the subprocess is killed via the process group reaper.
    pub timeout_secs: u64,
}

/// Configuration for the WASM executor (QuickJS interpreter sandbox).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WasmExecutorConfig {
    /// Path to the pre-compiled trusted interpreter WASM binary (e.g. "assets/quickjs.wasm").
    pub interpreter_wasm_path: String,
    /// Computational fuel budget per script execution; traps safely when exhausted.
    pub fuel_budget: u64,
}

/// Configuration for the human-in-the-loop approval gate.
///
/// Controls when task outputs are held for human review before delivery to the client.
/// All defaults are set in `reference.toml` under `[hitl]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HitlConfig {
    /// When `false`, the HITL gate is completely bypassed (development mode).
    pub enabled: bool,
    /// q_confidence below this threshold routes the task to human review.
    pub confidence_threshold: f64,
    /// Maximum milliseconds a task may wait for human approval before auto-rejection.
    pub timeout_ms: u64,
}

/// Configuration for the Constraint Wiki — tag-routed, NATS-backed constraint resolution.
///
/// When `enabled = false` (default), the system falls back to `corpus_path` flat-directory
/// behavior — identical to pre-wiki operation. No migration required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintWikiConfig {
    /// When true, use NATS KV wiki index for tag-based constraint resolution.
    /// When false, fall back to `corpus_path` flat-directory load (backward compat).
    pub enabled: bool,
    /// Filesystem path for flat-directory fallback.
    pub corpus_path: Option<String>,
    /// Maximum number of constraints resolved per task via tag lookup.
    /// Explicit manifest.constraints IDs are always included regardless of this limit.
    pub resolve_k: usize,
}

#[cfg(test)]
mod synthesis_config_tests {
    use super::*;

    #[test]
    fn synthesis_defaults_load_from_reference_toml() {
        let cfg = H2AIConfig::default();
        assert!(cfg.synthesis_enabled);
        assert_eq!(cfg.synthesis_min_proposals, 2);
        assert!((cfg.synthesis_tau - 0.2).abs() < 1e-9);
        assert_eq!(cfg.synthesis_critique_max_tokens, 1024);
        assert_eq!(cfg.synthesis_max_tokens, 2048);
    }

    #[test]
    fn subset_validation_does_not_panic_on_contradiction() {
        let cfg = H2AIConfig {
            shell_allowlist: vec!["git".into(), "ls".into()],
            shell_hardened_allowlist: vec!["ls".into(), "rm".into()],
            ..H2AIConfig::default()
        };
        cfg.validate_shell_allowlist_subset();
    }

    #[test]
    fn subset_validation_skipped_when_normal_allowlist_empty() {
        let cfg = H2AIConfig {
            shell_allowlist: vec![],
            shell_hardened_allowlist: vec!["rm".into()],
            ..H2AIConfig::default()
        };
        cfg.validate_shell_allowlist_subset();
    }
}

#[cfg(test)]
mod agent_config_tests {
    use super::*;

    #[test]
    fn agent_max_tool_iterations_default_is_five() {
        let cfg = H2AIConfig::default();
        assert_eq!(cfg.agent_max_tool_iterations, 5);
    }
}

#[cfg(test)]
mod a2a_adapter_tests {
    use super::*;

    #[test]
    fn a2a_adapter_kind_deserializes_from_toml() {
        use config::{Config, File, FileFormat};

        let toml_str = r#"
[adapter]
A2a = { endpoint = "https://example.com", auth_scheme = "bearer", auth_token_env = "TOKEN_ENV", timeout_minutes = 10, poll_interval_ms = 2000, max_poll_interval_ms = 30000, agent_card_cache_ttl_s = 3600 }
        "#;

        #[derive(serde::Deserialize)]
        struct Wrapper {
            adapter: AdapterKind,
        }

        let cfg = Config::builder()
            .add_source(File::from_str(toml_str, FileFormat::Toml))
            .build()
            .expect("config builder failed");

        let w: Wrapper = cfg
            .try_deserialize()
            .expect("should parse A2a AdapterKind from TOML");
        assert!(matches!(w.adapter, AdapterKind::A2a { .. }));

        // Verify field values
        if let AdapterKind::A2a {
            endpoint,
            auth_scheme,
            auth_token_env,
            timeout_minutes,
            poll_interval_ms,
            max_poll_interval_ms,
            agent_card_cache_ttl_s,
        } = w.adapter
        {
            assert_eq!(endpoint, "https://example.com");
            assert_eq!(auth_scheme, "bearer");
            assert_eq!(auth_token_env, "TOKEN_ENV");
            assert_eq!(timeout_minutes, 10);
            assert_eq!(poll_interval_ms, 2000);
            assert_eq!(max_poll_interval_ms, 30000);
            assert_eq!(agent_card_cache_ttl_s, 3600);
        } else {
            panic!("Expected A2a variant");
        }
    }
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
    /// Emits tracing::warn! for every command in shell_hardened_allowlist that is
    /// absent from shell_allowlist (when shell_allowlist is non-empty).
    /// Does NOT abort — the process boots with the contradiction in place.
    pub fn validate_shell_allowlist_subset(&self) {
        if self.shell_allowlist.is_empty() {
            return;
        }
        let normal: std::collections::HashSet<&str> =
            self.shell_allowlist.iter().map(String::as_str).collect();
        for cmd in &self.shell_hardened_allowlist {
            if !normal.contains(cmd.as_str()) {
                tracing::warn!(
                    cmd = cmd.as_str(),
                    "security contradiction: command is in shell_hardened_allowlist \
                     but absent from shell_allowlist — hardened mode grants MORE \
                     capability than normal mode"
                );
            }
        }
    }

    /// Load configuration using the three-layer stack (later layers win):
    ///
    /// 1. Embedded `reference.toml` — all defaults, always present
    /// 2. `override_path` file — operator-provided TOML with only changed fields
    /// 3. `H2AI_<FIELD>` env vars — highest priority, per-field overrides
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
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        );

        let cfg: Self = builder.build()?.try_deserialize()?;
        cfg.validate_shell_allowlist_subset();
        Ok(cfg)
    }

    /// Load configuration from a complete JSON file.
    ///
    /// Unlike `load_layered`, this does NOT merge with `reference.toml` — the JSON
    /// must contain all required fields. Partial JSON will fail deserialization.
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigLoadError> {
        let contents = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&contents)?;
        cfg.validate_shell_allowlist_subset();
        Ok(cfg)
    }
}
