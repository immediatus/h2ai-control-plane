pub mod prompts;

use h2ai_knowledge::factory::KnowledgeConfig;
use h2ai_types::config::AdapterKind;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Clarification template for oracle gate failures.
///
/// When the oracle gate fails with a reason, pattern matching against templates
/// generates follow-up questions for the user to clarify intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClarificationTemplate {
    /// Regex pattern to match oracle failure reason.
    pub pattern: String,
    /// Template with {placeholder} variables for generating clarification questions.
    pub question_template: String,
}

/// Configuration for the oracle gate feature.
///
/// When the engine finishes Phase 3, it calls an oracle service to verify
/// the proposed solution before proceeding to merge. All defaults are set in
/// `reference.toml` under `[oracle_gate]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleGateConfig {
    /// Enable the oracle gate. Default: false.
    #[serde(default = "default_oracle_gate_enabled")]
    pub enabled: bool,
    /// NATS subject to send gate requests to. Default: "h2ai.oracle.gate".
    #[serde(default = "default_oracle_gate_subject")]
    pub subject: String,
    /// Timeout in seconds for oracle gate requests. Default: 30.
    #[serde(default = "default_oracle_gate_timeout_secs")]
    pub timeout_secs: u64,
    /// Timeout behavior: "pass", "fail", or "clarify". Default: "pass".
    #[serde(default = "default_oracle_gate_on_timeout")]
    pub on_timeout: String,
    /// Gate passes if oracle confidence >= this. Default: 0.7.
    #[serde(default = "default_oracle_gate_min_confidence")]
    pub min_confidence: f64,
    /// Clarification templates for oracle failure reasons.
    #[serde(default)]
    pub clarification_templates: Vec<ClarificationTemplate>,
}

const fn default_oracle_gate_enabled() -> bool {
    false
}
fn default_oracle_gate_subject() -> String {
    "h2ai.oracle.gate".to_string()
}
const fn default_oracle_gate_timeout_secs() -> u64 {
    30u64
}
fn default_oracle_gate_on_timeout() -> String {
    "pass".to_string()
}
const fn default_oracle_gate_min_confidence() -> f64 {
    0.7f64
}

impl Default for OracleGateConfig {
    fn default() -> Self {
        Self {
            enabled: default_oracle_gate_enabled(),
            subject: default_oracle_gate_subject(),
            timeout_secs: default_oracle_gate_timeout_secs(),
            on_timeout: default_oracle_gate_on_timeout(),
            min_confidence: default_oracle_gate_min_confidence(),
            clarification_templates: Vec::new(),
        }
    }
}

/// Configuration for Phase 1.5 task complexity assessment and quadrant routing.
///
/// All defaults are set in `reference.toml` under `[task_complexity]`.
/// Shadow mode (default: `true`) lets you collect routing data before the
/// GAP-A1 experiment validates the thresholds — `ParetoRouter` is unchanged until
/// `shadow_mode` is set to `false`.
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
    /// Soft-constraint weight coefficient in `TCC_structural` formula.
    pub k_soft: f64,
    /// Type-diversity coefficient in `TCC_structural` formula.
    pub k_type: f64,
    /// Interaction-term (`soft_fraction` × `type_diversity`) coefficient in `TCC_structural`.
    pub k_cross: f64,
    /// Heavy-fraction amplification: when `static_coverage < min_static_coverage_for_probe`,
    /// `TCC_effective` = `tcc_structural` × (1 + `k_heavy` × `heavy_fraction`).
    pub k_heavy: f64,
    /// Minimum static coverage fraction required to run the N-probe sampling.
    /// Corpora with `static_coverage < this` are treated as heavy-dominant (probe skipped).
    pub min_static_coverage_for_probe: f64,
    /// Number of mini-probe calls used to estimate `TCC_empirical` (ambiguous band only).
    /// Probe is skipped on unambiguous Precision/Coverage paths and heavy-dominant corpora.
    pub n_probe: usize,
    /// Pool `N_eff` threshold below which Coverage → Complex escalation occurs.
    pub n_eff_complex_threshold: f64,
    /// Max tokens per probe completion. Probe outputs are structure assessments, not
    /// full answers — 512 tokens is sufficient for static constraint evaluation.
    pub probe_max_tokens: u64,
    /// Temperature for probe completions. Mid-range τ produces varied but coherent
    /// outputs needed to generate an informative satisfaction matrix.
    pub probe_tau: f64,
    /// Minimum number of informative static constraints (≥1 pass AND ≥1 fail across
    /// probes) needed to compute `TCC_empirical`. Below this, eigendecomposition is
    /// degenerate; fall back to `TCC_structural` with heavy amplification.
    pub tcc_min_informative_constraints: usize,
    /// Penalty added to `TCC_effective` when `TCC_structural` > `TCC_empirical` + 1.0.
    /// Signals that the corpus is more complex than static probes detected — typically
    /// because Heavy-tier constraints dominate actual complexity. Routes toward Coverage.
    pub tcc_mismatch_penalty: f64,
    /// Probe `n_eff` threshold (as fraction of `n_probe`) for Coverage vs Complex routing.
    /// Tasks with probe `n_eff` below `neff_probe_min_fraction × n_probe` escalate to Complex.
    pub neff_probe_min_fraction: f64,
    /// Probe `n_eff` threshold (as fraction of `n_probe`) for Precision vs Degenerate routing.
    /// Tasks with probe `n_eff` below `neff_probe_warning_fraction × n_probe` → Degenerate.
    pub neff_probe_warning_fraction: f64,
}

/// Model capability tier for Bayesian priors and OPRO triggering.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProfileTier {
    Fast,
    #[default]
    Standard,
    Capable,
}

/// Named adapter configuration entry used for `TaskProfile` routing.
///
/// Operators populate `H2AIConfig::adapter_profiles` with these entries so that
/// model backends are configured once and referenced by name throughout the
/// application, avoiding scattered `AdapterKind` values at startup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterProfile {
    /// Unique human-readable identifier for this profile (e.g. `"claude-sonnet"`).
    pub name: String,
    /// Backend kind and its connection parameters (API key env var, model string, etc.).
    pub kind: AdapterKind,
    /// Model capability tier for Bayesian priors and OPRO triggering.
    #[serde(default)]
    pub tier: ProfileTier,
    /// Set to `true` for models with built-in chain-of-thought (o1, o3, o4-mini, `DeepSeek` R1).
    /// Bypasses the TAO retry loop — these models' internal reasoning is the retry mechanism;
    /// injecting TAO memory over their own trace causes an α-spike that collapses USL `N_max`.
    #[serde(default)]
    pub is_reasoning_model: bool,
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

/// Configuration for the Phase 4 shadow auditor (GAP-C2).
///
/// When `enabled = true` and a shadow adapter is configured at startup
/// (via `H2AI_SHADOW_AUDITOR_PROVIDER`), every Phase 4 audit is accompanied
/// by a concurrent shadow call. Disagreements are counted per task domain.
/// When a domain's rolling disagreement rate exceeds `promotion_threshold`
/// over `promotion_window` observations, majority-vote (AND) enforcement
/// activates for that domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowAuditorConfig {
    /// Enable shadow auditor. Requires `H2AI_SHADOW_AUDITOR_PROVIDER` to be set.
    pub enabled: bool,
    /// Rolling-window disagreement rate above which majority vote activates.
    /// Default 0.05 (5%). Range (0, 1).
    pub promotion_threshold: f64,
    /// Observations per domain required before promotion is considered. Default 30.
    pub promotion_window: usize,
    /// Automatically remove a domain from majority-vote set when its disagreement
    /// rate drops below `promotion_threshold / 2` over `2 * promotion_window` obs.
    pub auto_demotion: bool,
    /// When true, shadow vote is binding (AND with primary) on every proposal regardless
    /// of domain promotion history. Set to true by Production and Strict safety profiles.
    #[serde(default)]
    pub strict: bool,
}

impl Default for ShadowAuditorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            promotion_threshold: 0.05,
            promotion_window: 30,
            auto_demotion: true,
            strict: false,
        }
    }
}

/// Named safety tier. When `profile != Custom`, `apply_safety_profile()` overwrites
/// all `SafetyConfig` fields from the profile definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SafetyProfile {
    #[default]
    Development,
    Production,
    Strict,
    Custom,
}

impl SafetyProfile {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Development => "development",
            Self::Production => "production",
            Self::Strict => "strict",
            Self::Custom => "custom",
        }
    }
}

/// Three-way policy for multi-family adapter pool enforcement.
/// Replaces `allow_single_family: bool`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FamilyConstraint {
    #[default]
    SingleFamilyOk,
    RequireDiverse,
    Disabled,
}

/// All safety-relevant config grouped under `[safety]` in TOML.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyConfig {
    pub profile: SafetyProfile,
    pub krum_fault_tolerance: usize,
    pub krum_threshold: f64,
    pub diversity_threshold: f64,
    pub family_constraint: FamilyConstraint,
    pub require_bivariate_cg: bool,
    pub shadow_auditor: ShadowAuditorConfig,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            profile: SafetyProfile::Development,
            krum_fault_tolerance: 0,
            krum_threshold: 0.30,
            diversity_threshold: 0.0,
            family_constraint: FamilyConstraint::SingleFamilyOk,
            require_bivariate_cg: false,
            shadow_auditor: ShadowAuditorConfig::default(),
        }
    }
}

/// Configuration for SRANI — Specification-Relative Architectural Noun Intersection.
/// Detects shared ungrounded architectural entities across proposals (GAP-C1 extension).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SraniConfig {
    /// When false, SRANI check is skipped entirely. Default true.
    #[serde(default = "srani_default_enabled")]
    pub enabled: bool,
    /// When true (default), use sigmoid gate with EMA midpoint.
    /// When false, use static `warn_threshold` / `inject_threshold`.
    #[serde(default = "srani_default_adaptive")]
    pub adaptive: bool,
    /// EMA smoothing factor for the adaptive midpoint. Range (0, 1].
    /// Lower = slower adaptation (longer memory). Default 0.20 ≈ 5-task horizon.
    #[serde(default = "srani_default_ema_alpha")]
    pub ema_alpha: f64,
    /// Sigmoid temperature: controls curve sharpness. Lower = sharper cliff. Default 0.15.
    #[serde(default = "srani_default_temperature")]
    pub temperature: f64,
    /// Injection pressure above which the grounding hint is injected. Default 0.50.
    #[serde(default = "srani_default_gate_threshold")]
    pub gate_threshold: f64,
    /// CFI above this threshold emits `CorrelatedFabricationEvent` (adaptive=false only).
    /// Also used for `cold_start_midpoint()` when adaptive=true. Default 0.3.
    #[serde(default = "srani_default_warn_threshold")]
    pub warn_threshold: f64,
    /// CFI above this threshold injects a grounding hint (adaptive=false only).
    /// Also used for `cold_start_midpoint()` when adaptive=true. Default 0.6.
    #[serde(default = "srani_default_inject_threshold")]
    pub inject_threshold: f64,
    /// When true (default) and a researcher adapter is available, distill raw
    /// web-search results with the LLM before injecting them as grounding context.
    #[serde(default = "srani_default_grounding_distill")]
    pub grounding_distill: bool,
    /// Minimum character count that triggers LLM distillation.
    /// Results shorter than this are already compact and skip compression.
    /// Default 800.
    #[serde(default = "srani_default_grounding_compress_threshold")]
    pub grounding_compress_threshold: usize,
}

const fn srani_default_enabled() -> bool {
    true
}
const fn srani_default_adaptive() -> bool {
    true
}
const fn srani_default_ema_alpha() -> f64 {
    0.20
}
const fn srani_default_temperature() -> f64 {
    0.15
}
const fn srani_default_gate_threshold() -> f64 {
    0.50
}
const fn srani_default_warn_threshold() -> f64 {
    0.3
}
const fn srani_default_inject_threshold() -> f64 {
    0.6
}
const fn srani_default_grounding_distill() -> bool {
    true
}
const fn srani_default_grounding_compress_threshold() -> usize {
    800
}

/// Configuration for OPRO (Optimization by Prompt Retrieval).
/// Controls trigger thresholds and promotion criteria for prompt variant optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OproConfig {
    /// Enable OPRO variant selection and promotion. Default: false.
    #[serde(default = "default_opro_enabled")]
    pub enabled: bool,
    /// Trigger OPRO when `j_eff` EMA falls below this threshold. Default: 0.6.
    #[serde(default = "default_opro_trigger_j_eff_threshold")]
    pub trigger_j_eff_threshold: f64,
    /// Minimum tasks completed before OPRO can be triggered. Default: 10.
    #[serde(default = "default_opro_min_tasks_before_trigger")]
    pub min_tasks_before_trigger: u32,
    /// Tasks to suppress re-trigger after each OPRO run. Default: 5.
    #[serde(default = "default_opro_suppress_n_tasks")]
    pub suppress_n_tasks: u32,
    /// Total tasks required before a variant can be promoted to primary. Default: 20.
    #[serde(default = "default_opro_graduation_tasks")]
    pub graduation_tasks: u32,
    /// Performance margin a variant must beat current by to promote. Default: 0.05.
    #[serde(default = "default_opro_promotion_margin")]
    pub promotion_margin: f64,
    /// EMA window size for `j_eff` smoothing. Default: 10.
    #[serde(default = "default_opro_ema_window")]
    pub ema_window: u32,
}

const fn default_opro_enabled() -> bool {
    false
}
const fn default_opro_trigger_j_eff_threshold() -> f64 {
    0.6f64
}
const fn default_opro_min_tasks_before_trigger() -> u32 {
    10u32
}
const fn default_opro_suppress_n_tasks() -> u32 {
    5u32
}
const fn default_opro_graduation_tasks() -> u32 {
    20u32
}
const fn default_opro_promotion_margin() -> f64 {
    0.05f64
}
const fn default_opro_ema_window() -> u32 {
    10u32
}

impl Default for OproConfig {
    fn default() -> Self {
        Self {
            enabled: default_opro_enabled(),
            trigger_j_eff_threshold: default_opro_trigger_j_eff_threshold(),
            min_tasks_before_trigger: default_opro_min_tasks_before_trigger(),
            suppress_n_tasks: default_opro_suppress_n_tasks(),
            graduation_tasks: default_opro_graduation_tasks(),
            promotion_margin: default_opro_promotion_margin(),
            ema_window: default_opro_ema_window(),
        }
    }
}

/// Configuration for calibration bootstrap parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationBootstrapConfig {
    /// Number of synthetic observations for the bootstrap prior. Default: 5.
    #[serde(default = "default_bootstrap_prior_weight")]
    pub prior_weight: u32,
}

const fn default_bootstrap_prior_weight() -> u32 {
    5u32
}

impl Default for CalibrationBootstrapConfig {
    fn default() -> Self {
        Self {
            prior_weight: default_bootstrap_prior_weight(),
        }
    }
}

/// Modifier injected into the system context for the epistemic probe phase.
/// Invalid TOML values cause a serde deserialization error at startup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SystemModifier {
    /// Injects: "Provide a maximally compressed architectural skeleton. Do not write
    /// implementation details. Prove constraint compliance immediately."
    #[default]
    CompressReasoning,
    /// Sends the full task prompt without modification.
    FullTask,
    /// Compiled from constraint YAML fields: criteria.pass + predicates → canonical task.
    ConstraintSkeleton,
}

/// Whether the epistemic probe uses the production prompt or a synthetic task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProbeTaskSource {
    /// Use the production prompt, truncated to probe `max_tokens` budget.
    #[default]
    Same,
    /// Compile from constraint YAML: criteria.pass + predicates → canonical task.
    /// Enables stationary k-regression independent of user payload entropy.
    Synthetic,
}

/// Configuration for the epistemic probe phase (Phase 2 of two-phase calibration).
///
/// The probe runs `agents` LLM instances on a short task, embeds outputs via cosine kernel,
/// computes `N_eff`, and derives β₀ = `f(N_eff^adj`, CG, k).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationProbeConfig {
    /// Number of agents in the epistemic probe. Minimum 3 for a valid β₀ solve.
    pub agents: usize,
    /// Max tokens for probe completions.
    pub max_tokens: u64,
    /// System context modifier injected during the probe.
    pub system_modifier: SystemModifier,
    /// Whether to use the production prompt or a synthetic constraint-derived task.
    pub probe_task_source: ProbeTaskSource,
    /// Exponent k in `N_eff^adj` = `clamp(N_eff` × CG^k, 1, `N_cal`).
    /// k=2 is a quadratic trapdoor: CG=0.65 → `N_eff^adj` collapses to ~1.
    pub neff_cg_exponent: f64,
}

impl Default for CalibrationProbeConfig {
    fn default() -> Self {
        Self {
            agents: 3,
            max_tokens: 512,
            system_modifier: SystemModifier::CompressReasoning,
            probe_task_source: ProbeTaskSource::Same,
            neff_cg_exponent: 2.0,
        }
    }
}

/// Configuration for AIMD slow start and congestion recovery.
///
/// On first registration, alpha starts at `seed_alpha`.
/// After each verified wave: `alpha = max(alpha × decay_rate, alpha_measured)`.
/// On yield < `reset_threshold`: `alpha = min(alpha × reset_multiplier, seed_alpha)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSlowStartConfig {
    /// Initial alpha for new adapter registrations. Conservative — suppresses `N_max` cold-start.
    pub seed_alpha: f64,
    /// Multiplicative decay per successful verification wave.
    pub decay_rate: f64,
    /// Multiplier applied to alpha on AIMD reset (yield < `reset_threshold`).
    pub reset_multiplier: f64,
    /// Yield floor (`N_useful/N_max`) below which AIMD reset fires.
    pub reset_threshold: f64,
}

impl Default for CalibrationSlowStartConfig {
    fn default() -> Self {
        Self {
            seed_alpha: 0.15,
            decay_rate: 0.95,
            reset_multiplier: 3.0,
            reset_threshold: 0.4,
        }
    }
}

impl Default for SraniConfig {
    fn default() -> Self {
        Self {
            enabled: srani_default_enabled(),
            adaptive: srani_default_adaptive(),
            ema_alpha: srani_default_ema_alpha(),
            temperature: srani_default_temperature(),
            gate_threshold: srani_default_gate_threshold(),
            warn_threshold: srani_default_warn_threshold(),
            inject_threshold: srani_default_inject_threshold(),
            grounding_distill: srani_default_grounding_distill(),
            grounding_compress_threshold: srani_default_grounding_compress_threshold(),
        }
    }
}

impl SraniConfig {
    /// Returns the static midpoint used during cold start (count < 5) and when adaptive=false.
    /// Derived from existing thresholds — no new config required.
    #[must_use]
    pub const fn cold_start_midpoint(&self) -> f64 {
        f64::midpoint(self.warn_threshold, self.inject_threshold)
    }
}

/// Adapter profile name mappings for the three thinking loop model tiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ThinkingModelTiers {
    /// Adapter profile name for Fast tier. Empty string = use first available explorer adapter.
    #[serde(default)]
    pub fast: String,
    /// Adapter profile name for Standard tier. Empty string = use first available explorer adapter.
    #[serde(default)]
    pub standard: String,
    /// Adapter profile name for Capable tier. Empty string = use first available explorer adapter.
    #[serde(default)]
    pub capable: String,
}

/// Configuration for the Persistent Reasoning Memory system (Phase 1–4).
///
/// Controls whether reasoning checkpoints are written, per-tenant KV bucket
/// TTLs, and induction cycle scheduling parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningMemoryConfig {
    /// Enable reasoning checkpoint writes and `TaskMetaState` projection.
    /// When `false`, all checkpoint writes are skipped (no NATS calls).
    #[serde(default)]
    pub enabled: bool,
    /// Minimum resolved-task count before triggering an induction cycle. Default: 10.
    #[serde(default = "default_induction_batch_size")]
    pub induction_batch_size: usize,
    /// Maximum seconds between induction cycles regardless of task count. Default: 86400 (24h).
    #[serde(default = "default_induction_max_interval_secs")]
    pub induction_max_interval_secs: u64,
    /// Maximum `TaskMetaState` records loaded per induction run. Default: 50.
    #[serde(default = "default_induction_max_tasks_per_run")]
    pub induction_max_tasks_per_run: usize,
    /// Jaccard tag-gate threshold for retrieval in Layer 3. Default: 0.2.
    #[serde(default = "default_tag_gate_threshold")]
    pub tag_gate_threshold: f64,
    /// Maximum archetype confidence boost from memory priors. Default: 0.15.
    #[serde(default = "default_max_archetype_boost")]
    pub max_archetype_boost: f64,
    /// Maximum archetype confidence penalty from `avoid_for_tags`. Default: 0.20.
    #[serde(default = "default_max_archetype_penalty")]
    pub max_archetype_penalty: f64,
    /// When `true`, reasoning checkpoint write failures abort the task with a hard error
    /// instead of logging a warning and continuing. Enable for compliance workloads where
    /// an un-auditable task execution is worse than a failed one. Default: `false`.
    #[serde(default)]
    pub strict_audit_checkpoint: bool,
}

const fn default_induction_batch_size() -> usize {
    10
}
const fn default_induction_max_interval_secs() -> u64 {
    86_400
}
const fn default_induction_max_tasks_per_run() -> usize {
    50
}
const fn default_tag_gate_threshold() -> f64 {
    0.2
}
const fn default_max_archetype_boost() -> f64 {
    0.15
}
const fn default_max_archetype_penalty() -> f64 {
    0.20
}

impl Default for ReasoningMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            induction_batch_size: default_induction_batch_size(),
            induction_max_interval_secs: default_induction_max_interval_secs(),
            induction_max_tasks_per_run: default_induction_max_tasks_per_run(),
            tag_gate_threshold: default_tag_gate_threshold(),
            max_archetype_boost: default_max_archetype_boost(),
            max_archetype_penalty: default_max_archetype_penalty(),
            strict_audit_checkpoint: false,
        }
    }
}

/// Configuration for the conflict-rate β signal (GAP-D1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConflictBetaConfig {
    #[serde(default = "default_conflict_beta_enabled")]
    pub enabled: bool,
    #[serde(default = "default_conflict_beta_max_samples")]
    pub max_samples: usize,
    #[serde(default = "default_conflict_beta_halflife_secs")]
    pub halflife_secs: u64,
    #[serde(default = "default_conflict_beta_min_samples_for_override")]
    pub min_samples_for_override: usize,
}

const fn default_conflict_beta_enabled() -> bool {
    true
}
const fn default_conflict_beta_max_samples() -> usize {
    100
}
const fn default_conflict_beta_halflife_secs() -> u64 {
    604_800
}
const fn default_conflict_beta_min_samples_for_override() -> usize {
    5
}

impl Default for ConflictBetaConfig {
    fn default() -> Self {
        Self {
            enabled: default_conflict_beta_enabled(),
            max_samples: default_conflict_beta_max_samples(),
            halflife_secs: default_conflict_beta_halflife_secs(),
            min_samples_for_override: default_conflict_beta_min_samples_for_override(),
        }
    }
}

/// Configuration for the multi-variant judge panel (GAP-A7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgePanelConfig {
    /// Supermajority fraction required for a confident `CrossFamily` verdict. Default: 0.67.
    #[serde(default = "default_judge_panel_quorum_fraction")]
    pub quorum_fraction: f64,
    /// Compliance score multiplier applied to uncertain-constraint contributions. Default: 0.7.
    #[serde(default = "default_judge_panel_uncertainty_weight")]
    pub uncertainty_weight: f64,
    /// Per-persona temperatures for `PersonaOnly` panels: [Literal, Contextual, Skeptical]. Default: [0.0, 0.2, 0.4].
    #[serde(default = "default_judge_panel_persona_temperatures")]
    pub persona_temperatures: [f32; 3],
    /// Minimum uncertain-vote count on one constraint per wave before emitting `ConstraintAmbiguityEvent`. Default: 2.
    #[serde(default = "default_judge_panel_ambiguity_threshold")]
    pub ambiguity_threshold: usize,
}

const fn default_judge_panel_quorum_fraction() -> f64 {
    0.67
}
const fn default_judge_panel_uncertainty_weight() -> f64 {
    0.7
}
const fn default_judge_panel_persona_temperatures() -> [f32; 3] {
    [0.0, 0.2, 0.4]
}
const fn default_judge_panel_ambiguity_threshold() -> usize {
    2
}

impl Default for JudgePanelConfig {
    fn default() -> Self {
        Self {
            quorum_fraction: default_judge_panel_quorum_fraction(),
            uncertainty_weight: default_judge_panel_uncertainty_weight(),
            persona_temperatures: default_judge_panel_persona_temperatures(),
            ambiguity_threshold: default_judge_panel_ambiguity_threshold(),
        }
    }
}

/// Configuration for the pre-execution thinking loop (spec: 2026-05-13-thinking-loop-design.md).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingLoopConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_thinking_max_iterations")]
    pub max_iterations: u32,
    #[serde(default = "default_thinking_max_archetypes")]
    pub max_archetypes: usize,
    #[serde(default = "default_thinking_coverage_threshold")]
    pub coverage_threshold: f64,
    #[serde(default = "default_thinking_convergence_threshold")]
    pub convergence_threshold: f64,
    /// Starting temperature for archetype brainstorm (iter 0). Default: 0.85
    #[serde(default = "default_tl_tau_max")]
    pub tau_max: f64,
    /// Minimum temperature for final iteration. Default: 0.20
    #[serde(default = "default_tl_tau_min")]
    pub tau_min: f64,
    /// Min verification pass rate required before expanding archetype count in iter N+1.
    /// Guards against quality degradation from uncontrolled diversity. Default: 0.30
    #[serde(default = "default_expansion_quality_floor")]
    pub expansion_quality_floor: f64,
    #[serde(default)]
    pub model_tiers: ThinkingModelTiers,
    /// Timeout for inline oracle check per archetype. Default: 20 seconds.
    #[serde(default = "default_oracle_timeout_secs")]
    pub oracle_timeout_secs: u64,
    /// `j_eff` boost when oracle passes. Default: 0.1
    #[serde(default = "default_oracle_confidence_bonus")]
    pub oracle_confidence_bonus: f64,
}

const fn default_oracle_timeout_secs() -> u64 {
    20u64
}
const fn default_oracle_confidence_bonus() -> f64 {
    0.1f64
}
const fn default_thinking_max_iterations() -> u32 {
    5
}
const fn default_thinking_max_archetypes() -> usize {
    4
}
const fn default_thinking_coverage_threshold() -> f64 {
    0.75
}
const fn default_thinking_convergence_threshold() -> f64 {
    0.90
}
const fn default_tl_tau_max() -> f64 {
    0.85
}
const fn default_tl_tau_min() -> f64 {
    0.20
}
const fn default_expansion_quality_floor() -> f64 {
    0.30
}

impl Default for ThinkingLoopConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_iterations: default_thinking_max_iterations(),
            max_archetypes: default_thinking_max_archetypes(),
            coverage_threshold: default_thinking_coverage_threshold(),
            convergence_threshold: default_thinking_convergence_threshold(),
            tau_max: default_tl_tau_max(),
            tau_min: default_tl_tau_min(),
            expansion_quality_floor: default_expansion_quality_floor(),
            model_tiers: ThinkingModelTiers::default(),
            oracle_timeout_secs: default_oracle_timeout_secs(),
            oracle_confidence_bonus: default_oracle_confidence_bonus(),
        }
    }
}

/// Configuration for GAP-I1: knowledge-gap detection and domain synthesis.
///
/// When `enabled = true`, the MAPE-K loop fires a researcher adapter on checks
/// whose historical pass rate is at or below `cold_check_threshold`. Synthesised
/// domain knowledge is accepted only when the LlmJudge scores it above
/// `synthesis_min_confidence`. At most `max_gap_records_per_wave` researcher
/// calls are issued per MAPE-K wave to bound latency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GapI1Config {
    /// Enable the GAP-I1 researcher loop. Default: false.
    pub enabled: bool,
    /// Only fire researcher on checks with pass rate ≤ this threshold. Default: 0.0.
    pub cold_check_threshold: f64,
    /// LlmJudge confidence score required to accept a `DomainSynthesis`. Default: 0.7.
    pub synthesis_min_confidence: f64,
    /// Maximum researcher calls issued per MAPE-K wave. Default: 3.
    pub max_gap_records_per_wave: usize,
    /// Seconds budget for web search + distillation per researcher call. Default: 90.
    pub researcher_timeout_secs: u64,
}

impl Default for GapI1Config {
    fn default() -> Self {
        Self {
            enabled: false,
            cold_check_threshold: 0.0,
            synthesis_min_confidence: 0.7,
            max_gap_records_per_wave: 3,
            researcher_timeout_secs: 90,
        }
    }
}

/// Configuration for GAP-K1 constraint coherence — pre-flight coherence probe and
/// runtime instability circuit breaker with automated spec repair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GapK1Config {
    /// Enable the pre-flight coherence probe and runtime instability detection. Default: false.
    pub enabled: bool,
    /// Enable automated spec repair via `SpecRepairAdvisor`. Default: false.
    pub auto_repair_enabled: bool,
    /// Minimum LlmJudge pass rate on the constraint's `pass` rubric to consider a check coherent. Default: 0.80.
    pub coherence_threshold: f64,
    /// Max Jaccard similarity between rejection reasons across waves before instability fires. Default: 0.10.
    pub instability_threshold: f64,
    /// Minimum self-consistency after repair to accept the rewrite. Default: 0.90.
    pub repair_acceptance_threshold: f64,
    /// Number of LlmJudge calls per check during coherence probe. Default: 5.
    pub probe_runs: usize,
    /// Number of candidate rewrites to generate per ambiguous check. Default: 3.
    pub repair_candidates: usize,
    /// TTL in seconds for coherence probe cache entries. Default: 86400 (24 h).
    pub probe_cache_ttl_secs: u64,
}

impl Default for GapK1Config {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_repair_enabled: false,
            coherence_threshold: 0.80,
            instability_threshold: 0.10,
            repair_acceptance_threshold: 0.90,
            probe_runs: 5,
            repair_candidates: 3,
            probe_cache_ttl_secs: 86400,
        }
    }
}

/// Configuration for CSPR-v2 patch-based constraint repair.
///
/// When enabled, `RetryWithHints` uses the best prior proposal as an anchor
/// and injects targeted per-violated-constraint repair instructions instead of
/// regenerating from scratch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CsprConfig {
    /// Enable CSPR-v2 patch repair. Default: false.
    pub enabled: bool,
}

/// Configuration for delta checkpoint encoding parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateDeltaConfig {
    #[serde(default = "default_delta_enabled")]
    pub enabled: bool,
    #[serde(default = "default_base_interval")]
    pub base_interval: u32,
    #[serde(default = "default_cache_ttl_secs")]
    pub cache_ttl_secs: u64,
    #[serde(default = "default_cache_max_entries")]
    pub cache_max_entries: usize,
}

const fn default_delta_enabled() -> bool {
    true
}
const fn default_base_interval() -> u32 {
    10
}
const fn default_cache_ttl_secs() -> u64 {
    60
}
const fn default_cache_max_entries() -> usize {
    200
}

impl Default for StateDeltaConfig {
    fn default() -> Self {
        Self {
            enabled: default_delta_enabled(),
            base_interval: default_base_interval(),
            cache_ttl_secs: default_cache_ttl_secs(),
            cache_max_entries: default_cache_max_entries(),
        }
    }
}

/// Configuration for NATS bucket names, stream names, and delta checkpoint encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateConfig {
    #[serde(default = "default_snapshots_bucket")]
    pub snapshots_bucket: String,
    #[serde(default = "default_task_checkpoints_bucket")]
    pub task_checkpoints_bucket: String,
    #[serde(default = "default_checkpoint_payloads_bucket")]
    pub checkpoint_payloads_bucket: String,
    #[serde(default = "default_oracle_calibration_bucket")]
    pub oracle_calibration_bucket: String,
    #[serde(default = "default_estimator_bucket")]
    pub estimator_bucket: String,
    #[serde(default = "default_calibration_bucket")]
    pub calibration_bucket: String,
    /// KV bucket for per-adapter-profile `CalibrationRecord` telemetry. Default: "`H2AI_CALIBRATION_RECORDS`".
    #[serde(default = "default_calibration_records_bucket")]
    pub calibration_records_bucket: String,
    /// KV bucket for per-adapter-profile `AuditorHealth` circuit-breaker state. Default: "`H2AI_AUDITOR_HEALTH`".
    #[serde(default = "default_auditor_health_bucket")]
    pub auditor_health_bucket: String,
    /// KV bucket for probe lease CAS tokens (`HalfOpen` circuit-breaker). Default: "`H2AI_PROBE_LEASE`".
    #[serde(default = "default_probe_lease_bucket")]
    pub probe_lease_bucket: String,
    #[serde(default = "default_sessions_bucket")]
    pub sessions_bucket: String,
    #[serde(default = "default_audit_shadow_bucket")]
    pub audit_shadow_bucket: String,
    #[serde(default = "default_prompt_variants_bucket")]
    pub prompt_variants_bucket: String,
    #[serde(default = "default_approvals_bucket")]
    pub approvals_bucket: String,
    /// NATS KV bucket name prefix for per-tenant reasoning checkpoints.
    /// Actual bucket: `{prefix}_{tenant_bucket_safe}`. Default: "`H2AI_CHECKPOINT`".
    #[serde(default = "default_reasoning_checkpoint_bucket_prefix")]
    pub reasoning_checkpoint_bucket_prefix: String,
    /// NATS KV bucket name prefix for per-tenant task meta-state records.
    /// Actual bucket: `{prefix}_{tenant_bucket_safe}`. Default: "`H2AI_META`".
    #[serde(default = "default_task_meta_state_bucket_prefix")]
    pub task_meta_state_bucket_prefix: String,
    /// NATS KV bucket name prefix for per-tenant distilled memory store (Phase 2).
    /// Actual bucket: `{prefix}_{tenant_bucket_safe}`. Default: "`H2AI_MEMORY`".
    #[serde(default = "default_tenant_memory_bucket_prefix")]
    pub tenant_memory_bucket_prefix: String,
    /// NATS KV bucket name prefix for per-tenant conflict-rate accumulators (GAP-D1).
    /// Actual bucket: `{prefix}_{tenant_bucket_safe}`. Default: "`H2AI_CONFLICT`".
    #[serde(default = "default_conflict_beta_bucket_prefix")]
    pub conflict_beta_bucket_prefix: String,
    // JetStream stream names
    #[serde(default = "default_tasks_stream")]
    pub tasks_stream: String,
    #[serde(default = "default_telemetry_stream")]
    pub telemetry_stream: String,
    #[serde(default = "default_results_stream")]
    pub results_stream: String,
    #[serde(default = "default_signals_stream")]
    pub signals_stream: String,
    /// NATS subject prefix for the signals stream. Default: "h2ai.signals".
    /// Override in tests to isolate from a concurrently-running server.
    #[serde(default = "default_signals_subject_prefix")]
    pub signals_subject_prefix: String,
    #[serde(default)]
    pub delta: StateDeltaConfig,
}

fn default_snapshots_bucket() -> String {
    "H2AI_SNAPSHOTS".to_string()
}
fn default_task_checkpoints_bucket() -> String {
    "H2AI_TASK_CHECKPOINTS".to_string()
}
fn default_checkpoint_payloads_bucket() -> String {
    "H2AI_CHECKPOINT_PAYLOADS".to_string()
}
fn default_oracle_calibration_bucket() -> String {
    "H2AI_ORACLE_CALIBRATION".to_string()
}
fn default_estimator_bucket() -> String {
    "H2AI_ESTIMATOR".to_string()
}
fn default_calibration_bucket() -> String {
    "H2AI_CALIBRATION".to_string()
}
fn default_calibration_records_bucket() -> String {
    "H2AI_CALIBRATION_RECORDS".to_string()
}
fn default_auditor_health_bucket() -> String {
    "H2AI_AUDITOR_HEALTH".to_string()
}
fn default_probe_lease_bucket() -> String {
    "H2AI_PROBE_LEASE".to_string()
}
fn default_sessions_bucket() -> String {
    "H2AI_SESSIONS".to_string()
}
fn default_audit_shadow_bucket() -> String {
    "H2AI_AUDIT_SHADOW".to_string()
}
fn default_prompt_variants_bucket() -> String {
    "H2AI_PROMPT_VARIANTS".to_string()
}
fn default_approvals_bucket() -> String {
    "H2AI_APPROVALS".to_string()
}
fn default_reasoning_checkpoint_bucket_prefix() -> String {
    "H2AI_CHECKPOINT".to_string()
}
fn default_task_meta_state_bucket_prefix() -> String {
    "H2AI_META".to_string()
}
fn default_tenant_memory_bucket_prefix() -> String {
    "H2AI_MEMORY".to_string()
}
fn default_conflict_beta_bucket_prefix() -> String {
    "H2AI_CONFLICT".to_string()
}
fn default_tasks_stream() -> String {
    "H2AI_TASKS".to_string()
}
fn default_telemetry_stream() -> String {
    "H2AI_TELEMETRY".to_string()
}
fn default_results_stream() -> String {
    "H2AI_RESULTS".to_string()
}
fn default_signals_stream() -> String {
    "H2AI_SIGNALS".to_string()
}
fn default_signals_subject_prefix() -> String {
    "h2ai.signals".to_string()
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            snapshots_bucket: default_snapshots_bucket(),
            task_checkpoints_bucket: default_task_checkpoints_bucket(),
            checkpoint_payloads_bucket: default_checkpoint_payloads_bucket(),
            oracle_calibration_bucket: default_oracle_calibration_bucket(),
            estimator_bucket: default_estimator_bucket(),
            calibration_bucket: default_calibration_bucket(),
            calibration_records_bucket: default_calibration_records_bucket(),
            auditor_health_bucket: default_auditor_health_bucket(),
            probe_lease_bucket: default_probe_lease_bucket(),
            sessions_bucket: default_sessions_bucket(),
            audit_shadow_bucket: default_audit_shadow_bucket(),
            prompt_variants_bucket: default_prompt_variants_bucket(),
            approvals_bucket: default_approvals_bucket(),
            reasoning_checkpoint_bucket_prefix: default_reasoning_checkpoint_bucket_prefix(),
            task_meta_state_bucket_prefix: default_task_meta_state_bucket_prefix(),
            tenant_memory_bucket_prefix: default_tenant_memory_bucket_prefix(),
            conflict_beta_bucket_prefix: default_conflict_beta_bucket_prefix(),
            tasks_stream: default_tasks_stream(),
            telemetry_stream: default_telemetry_stream(),
            results_stream: default_results_stream(),
            signals_stream: default_signals_stream(),
            signals_subject_prefix: default_signals_subject_prefix(),
            delta: StateDeltaConfig::default(),
        }
    }
}

/// Single configuration authority for the H2AI Control Plane runtime.
///
/// All fields are populated by `load_layered()`, which merges the embedded
/// `reference.toml` defaults, an optional operator override file, and
/// `H2AI_<FIELD>` environment variables (highest priority wins).
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct H2AIConfig {
    /// BFT consensus threshold — fraction of agents that must agree for a result to be accepted. Range [0, 1].
    pub bft_threshold: f64,
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
    /// Single model token budget — canonical reference for all full-generation LLM calls.
    /// Intentional exceptions: `leader_diagnosis_max_tokens` (128) and `calibration_probe.max_tokens` (512).
    pub model_max_tokens: u64,
    /// Maximum tokens per explorer generation call.
    pub explorer_max_tokens: u64,
    /// Maximum tokens per calibration probe call.
    pub calibration_max_tokens: u64,
    /// Maximum tokens for decomposition steps 1 and 2 (analysis + role design).
    pub decomposition_step_max_tokens: u64,
    /// Maximum tokens for decomposition step 3 (JSON formatting). Thinking models need extra
    /// budget here because they consume tokens on <think> chains before emitting JSON.
    pub decomposition_json_max_tokens: u64,
    /// When `false`, passes `chat_template_kwargs: {"enable_thinking": false}` in every
    /// adapter request body. Disables the extended thinking chain on Qwen3-style models
    /// served by llama.cpp, making multi-step pipelines practical. Cloud APIs (`OpenAI`,
    /// Anthropic) silently ignore unknown body fields, so this is safe globally.
    pub adapter_enable_thinking: bool,
    /// Temperature used for all calibration adapter probes. Range [0, 1].
    pub calibration_tau: f64,
    /// Initial verification pass threshold applied to each proposal's aggregate compliance score. Range [0, 1].
    pub verify_threshold: f64,
    /// Step size for `verify_threshold` reduction suggestions from the self-optimizer. Range (0, 1).
    pub optimizer_threshold_step: f64,
    /// Floor for `verify_threshold` reductions; the threshold never drops below this value.
    pub optimizer_threshold_floor: f64,
    /// Maximum autonomic MAPE-K retry iterations before a task is declared failed.
    pub max_autonomic_retries: u32,
    /// Enable Raft-style cross-wave epistemic leader election. Default: false.
    pub leader_enabled: bool,
    /// Minimum `q_confidence` improvement per wave to avoid stagnation. Range [0,1].
    pub leader_stagnation_threshold: f64,
    /// Consecutive stagnant waves before leadership rotates to runner-up.
    pub leader_stagnation_waves: u32,
    /// Max tokens for the leader's diagnosis re-prompt.
    pub leader_diagnosis_max_tokens: u64,
    /// Temperature for leader diagnosis re-prompt. Range [0,1].
    pub leader_diagnosis_tau: f64,
    /// Number of EIG candidate questions generated per diagnosis round.
    pub leader_eig_candidates: u32,
    /// Credibility decay/recovery rate per wave. Range [0.0, 1.0].
    pub leader_credibility_decay_rate: f64,
    /// Credibility threshold below which follower context is marked low-confidence.
    pub leader_credibility_warn_threshold: f64,
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
    /// Hard deadline in seconds for a single task end-to-end. `None` means no deadline; omit from the override file to leave unlimited.
    #[serde(default)]
    pub task_deadline_secs: Option<u64>,
    /// Maximum number of concurrent task executions; requests beyond this limit receive HTTP 503.
    pub max_concurrent_tasks: usize,
    /// Named adapter profiles available for `TaskProfile` routing.
    pub adapter_profiles: Vec<AdapterProfile>,
    /// Context pressure sensitivity γ: scales how much a full context window raises β. `0` disables the effect; `0.5` doubles β at 100% context fill. Range [0, 1].
    pub context_pressure_gamma: f64,
    /// Per-adapter baseline accuracy proxy. `0.0` uses the CG-mean proxy (`0.5 + CG_mean / 2`); set to an empirically measured value via `compare.py` (benchmark tool — see `docs/architecture/reference.md`).
    pub baseline_accuracy_proxy: f64,
    /// Number of adapter instances spawned during calibration. Minimum 3 for a valid USL two-point fit; fewer falls back to `alpha_contention` and `beta_base_default`.
    pub calibration_adapter_count: usize,
    /// τ spread `[min, max]` for calibration instances; instances are spaced linearly across this range. The spread may expand up to `tau_spread_max_factor` when Talagrand detects over-confidence.
    pub calibration_tau_spread: [f64; 2],
    /// CG collapse threshold: when `CG_embed` drops below this value the planner forces `N_max` = 1. Default `0.10` — below 10 % agent outputs are so divergent that coherence drag is unbounded.
    pub cg_collapse_threshold: f64,
    /// Cosine similarity threshold for counting two adapter outputs as "in agreement" when computing CG via embedding cosine (future; currently CG uses constraint-profile Hamming).
    pub cg_agreement_threshold: f64,
    /// Embedding model used for CG cosine agreement measurement; requires the `fastembed-embed` Cargo feature.
    pub embedding_model_name: EmbeddingModelName,
    /// Minimum `N_eff` increment required to include the next adapter in `EigenCalibration::n_pruned`.
    /// Adapter k is kept when adding it raises `N_eff` by ≥ this delta. Default 0.05.
    /// Increase toward 0.1–0.2 for calibrations with few adapters (N ≤ 4).
    pub eigen_n_eff_delta: f64,
    /// Minimum number of TAO loop samples before `TaoMultiplierEstimator` state is persisted
    /// to NATS. The EMA estimate is unreliable below this count. Default 20.
    /// Raise to 50–100 for high-variance task distributions.
    pub tao_estimator_warmup: usize,
    /// Initial `N_max` used to seed the Thompson Sampling bandit warm prior at first startup
    /// before any calibration result is available. Clamped to [1, `bandit_n_max_arms`] by the bandit. Default 4.
    pub bandit_n_max_initial: u32,
    /// Tasks completed before activating the bandit (Phase 0 — pure exploration); during Phase 0 N = `N_max_USL` unconditionally.
    pub bandit_phase0_k: u32,
    /// Tasks completed before switching from ε-greedy to pure Thompson Sampling (Phase 1).
    pub bandit_phase1_k: u32,
    /// ε for Phase 1 ε-greedy: probability of selecting a random arm each task. Range [0, 1].
    pub bandit_epsilon: f64,
    /// Soft-reset decay factor applied to the learned posterior when the adapter version hash changes. `0.3` blends 30 % toward the initial prior.
    pub bandit_soft_reset_decay: f64,
    /// Maximum ensemble size explored by `Condorcet/n_it_optimal` search during calibration.
    /// Bounds `EnsembleCalibration::from_cg_mean` and `from_measured_p`. Default 9.
    /// Lower on resource-constrained nodes; raise for very large adapter pools.
    #[serde(default = "default_calibration_max_ensemble_size")]
    pub calibration_max_ensemble_size: usize,
    /// Maximum number of arms (N values) the Thompson Sampling bandit explores.
    /// The bandit considers N ∈ [1, `bandit_n_max_arms`]. Default 6.
    /// Must be ≤ `calibration_max_ensemble_size` for coherent physics.
    #[serde(default = "default_bandit_n_max_arms")]
    pub bandit_n_max_arms: u32,
    /// σ for the Gaussian warm prior centred on `N_max_USL` in the bandit.
    /// σ=2 means arms within 2 of `N_max` get meaningful weight; σ²=4 appears in the exponent.
    /// Default 2.0. Decrease for tighter priors (faster convergence, less exploration).
    #[serde(default = "default_bandit_prior_sigma")]
    pub bandit_prior_sigma: f64,
    /// Prior strength (pseudo-observation count) for the bandit warm prior. Default 5.0.
    /// Higher = slower to learn from real task feedback; good for fresh deployments.
    #[serde(default = "default_bandit_prior_strength")]
    pub bandit_prior_strength: f64,
    /// Maximum slot count for Precision-quadrant tasks (Self-MoA budget). Default 3.
    /// Lower bound is 2 (synthesis requires ≥2 proposals). Raise for larger single-family pools.
    #[serde(default = "default_precision_mode_max_slots")]
    pub precision_mode_max_slots: usize,
    /// Rolling window size for oracle calibration observations. Default 200.
    /// Older entries are dropped (FIFO) when the window exceeds this size.
    #[serde(default = "default_oracle_window_size")]
    pub oracle_window_size: usize,
    /// ECE (Expected Calibration Error) threshold above which `CalibrationDriftWarning`
    /// fires. Requires ≥ 30 oracle observations. Default 0.15.
    #[serde(default = "default_oracle_ece_alert_threshold")]
    pub oracle_ece_alert_threshold: f64,
    /// Oracle pass-rate floor below which `OracleSuspectEvent` fires.
    /// Requires ≥ 30 oracle observations. Default 0.30.
    #[serde(default = "default_oracle_pass_rate_floor")]
    pub oracle_pass_rate_floor: f64,
    /// Maximum τ-spread expansion factor when Talagrand detects over-confidence. `2.0` means the spread can at most double.
    pub tau_spread_max_factor: f64,
    /// When `true`, automatically switches to the Empirical prediction basis after `auto_baseline_eval_min_tasks` Tier 1 oracle tasks complete.
    pub auto_baseline_eval: bool,
    /// Minimum Tier 1 oracle task count before automatic baseline evaluation triggers.
    pub auto_baseline_eval_min_tasks: u32,
    /// Fraction of proposals that must survive verification for a run to be considered non-wasteful; below this threshold the self-optimizer suggests reducing the `verify_threshold`.
    pub optimizer_waste_threshold: f64,
    /// Agent dispatch scheduling policy.
    pub scheduler_policy: SchedulerPolicy,
    /// Queue depth per cost tier at which `CostAwareSpillover` routes to the next tier.
    pub scheduler_spillover_threshold: usize,
    /// Byte length above which `system_context` is offloaded to the payload store. Default 524288 (512 KB) — half of NATS 1 MB default limit.
    pub payload_offload_threshold_bytes: usize,
    /// Events published per task before a state snapshot is written to NATS KV. Reduces crash-recovery replay time. Default 50. 0 disables snapshotting.
    pub snapshot_interval_events: usize,
    /// URL path prefix for all versioned API routes (tasks, calibrate).
    /// Health/ready/metrics are always mounted at root. Example: `"v1"` → `/v1/tasks`.
    pub api_version: String,
    /// HTTP listen address for the API server. Default: "0.0.0.0:8080".
    pub listen_addr: String,
    /// NATS server URL used by the API server, agent binary, and integration tests.
    pub nats_url: String,
    /// Enable NATS agent dispatch mode. When true, explorer slots are dispatched to `TaoAgent` processes via NATS. Default: false.
    pub nats_dispatch_enabled: bool,
    /// TTL in seconds for NATS-dispatched agent task slots. Default: 30.
    pub nats_agent_ttl_secs: u64,
    /// Model name reported in `AgentDescriptor` for NATS-dispatched agent tasks.
    pub nats_agent_model: String,
    /// Timeout in seconds for a single NATS-dispatched agent task. Default: 120.
    pub nats_agent_timeout_secs: u64,
    /// Enable the synthesis phase. When false, the engine uses the selection chain exclusively.
    /// Default: true. Set false to reproduce pre-synthesis behavior for benchmarking.
    pub synthesis_enabled: bool,
    /// Enable the GAP-F1 synthesis wave — one terminal LLM call after all retries exhaust.
    /// Fires only when partial-pass proposals exist. Default: true.
    pub synthesis_wave_enabled: bool,
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
    /// Overhead factor used in the partial-pass truncation budget formula:
    /// `chars_per_partial = model_max_tokens * 4 / (max_k + partial_pass_overhead_factor)`.
    /// Represents non-partial content (system prompt, task description, repair context) in
    /// units of partial-slot equivalents. Default: 5.0.
    pub partial_pass_overhead_factor: f64,
    /// Enable sequential constraint grafting in the synthesis wave (GAP-H1).
    /// When true, the engine iteratively integrates one constraint cluster per LLM call
    /// starting from the highest-scoring partial as the seed. Each round includes
    /// intermediate verification with rollback — destructive grafts are discarded.
    /// When false (default), the existing single-shot `build_synthesis_context` path is used.
    pub sequential_grafting_enabled: bool,
    /// Maximum graft rounds before stopping the loop.
    /// Each round is one focused LLM call + one verification pass.
    /// Default: 4. Setting to 1 returns only the seed partial without grafting.
    pub sequential_grafting_max_rounds: usize,
    /// DDM sliding window size in tasks. Default: 20.
    pub drift_ddm_window: usize,
    /// DDM detection threshold in standard deviations. Default: 2.5.
    pub drift_ddm_k: f64,
    /// BOCPD geometric hazard rate: P(changepoint now) per observation. Default: 0.01.
    pub drift_bocpd_hazard_rate: f64,
    /// BOCPD posterior mass threshold on run-lengths {0..=4} to fire CalibrationChangepoint. Default: 0.90.
    pub drift_bocpd_changepoint_threshold: f64,
    /// When true, POST /calibrate is triggered automatically on CalibrationChangepoint.
    /// Default: false — recalibration costs LLM calls; require explicit operator opt-in.
    pub auto_recalibrate_on_drift: bool,
    /// Max seconds before a stale-calibration tracing::warn is emitted. Default: 3600.
    pub drift_staleness_ttl_secs: u64,
    /// ORCA conformal margin subtracted from VerificationConfig::threshold during active drift.
    /// Widens the gate (more proposals pass) as a conservative coverage guarantee. Default: 0.05.
    pub drift_conformal_margin: f64,
    /// Commands permitted in Normal-mode waves. Empty = unrestricted (unsafe in production).
    pub shell_allowlist: Vec<String>,
    /// Commands permitted in Hardened-mode waves. Should be a subset of `shell_allowlist`.
    pub shell_hardened_allowlist: Vec<String>,
    /// Maximum seconds a shell tool invocation may run before it is killed. Default: 5.
    pub shell_timeout_secs: u64,
    /// Maximum number of TAO loop tool-call iterations an edge agent may execute per task.
    /// After this limit the agent returns whatever output the LLM produced last. Default: 5.
    /// Valid range: 1–255. A value of 0 is rejected by the `TaoAgent` and treated as 1.
    pub agent_max_tool_iterations: u8,
    /// Maximum UTF-8 byte length of a single tool observation appended to the agent
    /// context. Observations longer than this are truncated with a suffix noting the
    /// original and capped lengths. Default: 8192. Set to 0 to disable truncation.
    pub agent_max_observation_chars: usize,
    /// Google Custom Search configuration. Absent = `WebSearch` executor disabled.
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
    /// Constraint resolution backend (fs / disabled).
    #[serde(default)]
    pub constraint_wiki: ConstraintWikiConfig,
    /// Token budget for per-slot context assembly. `None` disables compression.
    /// When set, rule-based dedup + importance trimming run on every wave.
    /// LLM summarization triggers only when rule-based pass still exceeds budget.
    #[serde(default)]
    pub context_budget_tokens: Option<usize>,
    /// Minimum compression ratio before the quality guard stops further compression.
    /// 0.4 means stop if more than 60 % of the original context would be removed.
    /// `None` uses the default of 0.4.
    #[serde(default)]
    pub context_quality_guard_ratio: Option<f32>,
    /// Adapter profile name to use for the LLM summarization pass.
    /// When absent, the first explorer adapter is used.
    #[serde(default)]
    pub compression_adapter: Option<String>,
    /// Safety profile — groups all safety-relevant fields.
    /// When `profile != custom`, `apply_safety_profile()` in `load_layered()` overwrites
    /// all safety field values from the profile definition.
    #[serde(default)]
    pub safety: SafetyConfig,
    /// CV threshold below which correlated hallucination detection fires (GAP-C1).
    /// CV = `std_dev(pairwise_jaccard_distances)` / `mean(pairwise_jaccard_distances)`.
    /// Low CV = proposals are semantically clustered → correlated assumption risk.
    /// Default 0.30. Set to 0.0 to disable C1 detection entirely.
    #[serde(default = "default_correlated_hallucination_cv_threshold")]
    pub correlated_hallucination_cv_threshold: f64,
    /// Minimum mean pairwise Jaccard distance required for C1 to fire.
    /// Guards against spurious C1 retries when proposals are already diverse by absolute measure
    /// (uniform-but-diverse ensemble: low CV but high mean distance). Default 0.50.
    /// C1 fires only when BOTH cv < `cv_threshold` AND `mean_jaccard` < this floor.
    #[serde(default = "default_correlated_hallucination_min_jaccard_floor")]
    pub correlated_hallucination_min_jaccard_floor: f64,
    /// Minimum fraction of constraint corpus domains that slot assignments must cover (GAP-C3).
    /// `coverage_score` = |`covered_domains` ∩ `corpus_domains`| / |`corpus_domains`|.
    /// Fires `DiversityGuardDegradedEvent` when below this threshold. Default 0.40.
    #[serde(default = "default_domain_coverage_threshold")]
    pub domain_coverage_threshold: f64,
    /// SRANI correlated fabrication detection configuration (GAP-C1 extension).
    #[serde(default)]
    pub srani: SraniConfig,
    /// Optional path to an NDJSON debug log file. When set, every completed task
    /// appends one JSON line containing the full spec, all proposals with scores,
    /// SRANI events, EMA before/after, and the merged output — no truncation.
    /// File is opened in append mode; directory must already exist.
    /// Example: `/tmp/h2ai-debug.ndjson`
    #[serde(default)]
    pub debug_log_path: Option<String>,
    /// LLM judge passes per Hard constraint evaluation (averaged). 1 = current single-pass behavior.
    /// ≥2 reduces false-positives by requiring multiple independent passes to agree.
    #[serde(default = "default_verifier_consensus_passes")]
    pub verifier_consensus_passes: u8,
    /// Pre-execution thinking loop configuration.
    /// When enabled, runs archetype selection + parallel brainstorm + synthesis before
    /// the 3-step decomposition. Disabled by default — opt-in per scenario.
    #[serde(default)]
    pub thinking_loop: ThinkingLoopConfig,
    /// CSPR-v2 patch-based constraint repair configuration.
    #[serde(default)]
    pub cspr: CsprConfig,
    /// Oracle gate configuration.
    /// Controls Phase 3 → Phase 4 verification via oracle service.
    #[serde(default)]
    pub oracle_gate: OracleGateConfig,
    /// NATS bucket names and delta checkpoint encoding parameters.
    #[serde(default)]
    pub state: StateConfig,
    /// OPRO (Optimization by Prompt Retrieval) configuration.
    /// Controls variant selection, promotion criteria, and trigger thresholds.
    #[serde(default)]
    pub opro: OproConfig,
    /// Calibration bootstrap configuration.
    /// Controls synthetic prior weight for calibration initialization.
    #[serde(default)]
    pub calibration_bootstrap: CalibrationBootstrapConfig,
    /// Persistent Reasoning Memory configuration — checkpoint writes, induction cycles, retrieval.
    #[serde(default)]
    pub reasoning_memory: ReasoningMemoryConfig,
    /// Conflict-rate β accumulator configuration (GAP-D1).
    #[serde(default)]
    pub conflict_beta: ConflictBetaConfig,
    /// Judge panel configuration for Phase 3.5 multi-variant evaluation (GAP-A7).
    #[serde(default)]
    pub judge_panel: JudgePanelConfig,
    /// Knowledge provider configuration. When absent, uses `PassthroughProvider`
    /// (delegates to existing `ConstraintResolver` — zero behaviour change).
    #[serde(default)]
    pub knowledge: Option<KnowledgeConfig>,
    /// ms to wait at each `WaveCompleted` boundary for a `WaveContinue` signal. 0 = disabled.
    #[serde(default)]
    pub signal_wave_window_ms: u64,
    /// Minimum `timeout_ms` a caller may request via POST /signal.
    #[serde(default = "default_signal_min_timeout_ms")]
    pub signal_min_timeout_ms: u64,
    /// Maximum `timeout_ms` a caller may request via POST /signal.
    #[serde(default = "default_signal_max_timeout_ms")]
    pub signal_max_timeout_ms: u64,
    /// Epistemic probe phase configuration (two-phase calibration, spec section 2-3).
    /// Absent from config file → uses defaults (agents=3, `max_tokens=512`, k=2.0).
    #[serde(default)]
    pub calibration_probe: CalibrationProbeConfig,
    /// AIMD slow start and congestion recovery configuration (spec section 4).
    /// Absent from config file → uses defaults (`seed_alpha=0.15`, `decay_rate=0.95`).
    #[serde(default)]
    pub calibration_slow_start: CalibrationSlowStartConfig,
    /// Optional OSP configuration. When None, merger uses legacy strategy dispatch.
    #[serde(default)]
    pub osp: Option<h2ai_types::sizing::OspConfig>,
    /// NATS KV bucket name for human oracle pending requests (per-tenant prefix).
    /// Default: "`H2AI_ORACLE_HUMAN`".
    #[serde(default = "default_oracle_human_bucket")]
    pub oracle_human_bucket: String,
    /// GAP-I1 knowledge-gap detection and domain synthesis configuration.
    #[serde(default)]
    pub gap_i1: GapI1Config,
    /// GAP-K1 constraint coherence configuration.
    #[serde(default)]
    pub gap_k1: GapK1Config,
}

fn default_oracle_human_bucket() -> String {
    "H2AI_ORACLE_HUMAN".to_string()
}

const fn default_correlated_hallucination_cv_threshold() -> f64 {
    0.30
}
const fn default_correlated_hallucination_min_jaccard_floor() -> f64 {
    0.50
}
const fn default_domain_coverage_threshold() -> f64 {
    0.40
}
const fn default_calibration_max_ensemble_size() -> usize {
    9
}
const fn default_bandit_n_max_arms() -> u32 {
    6
}
const fn default_bandit_prior_sigma() -> f64 {
    2.0
}
const fn default_bandit_prior_strength() -> f64 {
    5.0
}
const fn default_precision_mode_max_slots() -> usize {
    3
}
const fn default_oracle_window_size() -> usize {
    200
}
const fn default_oracle_ece_alert_threshold() -> f64 {
    0.15
}
const fn default_oracle_pass_rate_floor() -> f64 {
    0.30
}
const fn default_verifier_consensus_passes() -> u8 {
    1
}
const fn default_signal_min_timeout_ms() -> u64 {
    60_000
}
const fn default_signal_max_timeout_ms() -> u64 {
    86_400_000
}

/// Configuration for the `WebSearch` executor (Google Custom Search API).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchConfig {
    /// Name of the environment variable holding the Google Custom Search API key.
    pub api_key_env: String,
    /// Name of the environment variable holding the Google Custom Search Engine ID.
    pub cx_env: String,
    /// Maximum number of search result snippets returned to the LLM. Default: 3.
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

const fn default_max_results() -> usize {
    3
}

/// Configuration for the MCP filesystem executor (stdio subprocess transport).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpFilesystemConfig {
    /// Binary to spawn for the MCP server (e.g. "npx").
    pub command: String,
    /// Arguments passed to the binary (e.g. `["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]`).
    pub args: Vec<String>,
    /// Seconds before the subprocess is killed via the process group reaper.
    pub timeout_secs: u64,
}

/// Configuration for the WASM executor (`QuickJS` interpreter sandbox).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// `q_confidence` below this threshold routes the task to human review.
    pub confidence_threshold: f64,
    /// Maximum milliseconds a task may wait for human approval before auto-rejection.
    pub timeout_ms: u64,
    /// Multiplier applied per consecutive non-response: effective = base × decay^n.
    /// Must be in (0.0, 1.0). Default: 0.5 (halve per miss).
    pub timeout_decay: f64,
    /// Minimum effective timeout in ms regardless of decay. Default: `300_000` (5 min).
    pub timeout_floor_ms: u64,
}

const fn default_resolve_k() -> usize {
    50
}

/// Constraint resolution backend. Exactly one variant is active — the
/// previously-possible `enabled=true + corpus_path` ambiguity cannot be expressed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ConstraintWikiConfig {
    /// No constraint resolution (default).
    #[default]
    Disabled,
    /// FS-backed: load from local YAML constraint files at `corpus_path`.
    Fs {
        corpus_path: String,
        #[serde(default = "default_resolve_k")]
        resolve_k: usize,
    },
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
    /// Better MTEB STS scores than `AllMiniLmL6V2`; recommended for production deployments.
    BgeSmallEnV1_5,
}

impl Default for H2AIConfig {
    fn default() -> Self {
        Self::load_layered(None).expect("embedded reference.toml is always valid")
    }
}

impl H2AIConfig {
    /// Emits `tracing::warn`! for every command in `shell_hardened_allowlist` that is
    /// absent from `shell_allowlist` (when `shell_allowlist` is non-empty).
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
    /// # Errors
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

        let mut cfg: Self = builder.build()?.try_deserialize()?;
        apply_safety_profile(&mut cfg);
        cfg.validate_shell_allowlist_subset();
        Ok(cfg)
    }

    /// Load configuration from a complete JSON file.
    ///
    /// Unlike `load_layered`, this does NOT merge with `reference.toml` — the JSON
    /// must contain all required fields. Partial JSON will fail deserialization.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigLoadError`] if the file cannot be read or if deserialization fails.
    pub fn load_from_file(path: &Path) -> Result<Self, ConfigLoadError> {
        let contents = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&contents)?;
        cfg.validate_shell_allowlist_subset();
        Ok(cfg)
    }
}

/// Emit one structured tracing log block per nested config section at startup.
///
/// Logs at WARN when a primary safety/activation flag is in a degraded or disabled state;
/// logs at INFO otherwise. Call this once after `load_layered()` completes.
#[allow(clippy::too_many_lines)]
pub fn log_startup_config_report(cfg: &H2AIConfig) {
    // [safety] block — WARN if development profile
    if cfg.safety.profile == SafetyProfile::Development {
        tracing::warn!(
            profile = %cfg.safety.profile.as_str(),
            krum_fault_tolerance = cfg.safety.krum_fault_tolerance,
            krum_threshold = cfg.safety.krum_threshold,
            diversity_threshold = cfg.safety.diversity_threshold,
            family_constraint = %format!("{:?}", cfg.safety.family_constraint),
            require_bivariate_cg = cfg.safety.require_bivariate_cg,
            shadow_auditor_enabled = cfg.safety.shadow_auditor.enabled,
            "[safety] profile={} ({})",
            cfg.safety.profile.as_str(),
            "non-production defaults active"
        );
    } else {
        tracing::info!(
            profile = %cfg.safety.profile.as_str(),
            krum_fault_tolerance = cfg.safety.krum_fault_tolerance,
            krum_threshold = cfg.safety.krum_threshold,
            diversity_threshold = cfg.safety.diversity_threshold,
            family_constraint = %format!("{:?}", cfg.safety.family_constraint),
            require_bivariate_cg = cfg.safety.require_bivariate_cg,
            shadow_auditor_enabled = cfg.safety.shadow_auditor.enabled,
            "[safety] profile={} ({})",
            cfg.safety.profile.as_str(),
            "production-grade safety active"
        );
    }

    // [task_complexity] block — WARN if shadow_mode=true
    if cfg.task_complexity.shadow_mode {
        tracing::warn!(
            shadow_mode = cfg.task_complexity.shadow_mode,
            tcc_precision = cfg.task_complexity.tcc_precision_threshold,
            tcc_coverage = cfg.task_complexity.tcc_coverage_threshold,
            k_soft = cfg.task_complexity.k_soft,
            k_type = cfg.task_complexity.k_type,
            k_cross = cfg.task_complexity.k_cross,
            k_heavy = cfg.task_complexity.k_heavy,
            n_probe = cfg.task_complexity.n_probe,
            n_eff_complex_threshold = cfg.task_complexity.n_eff_complex_threshold,
            "[task_complexity] shadow_mode={} ({})",
            cfg.task_complexity.shadow_mode,
            "quadrant routing disabled"
        );
    } else {
        tracing::info!(
            shadow_mode = cfg.task_complexity.shadow_mode,
            tcc_precision = cfg.task_complexity.tcc_precision_threshold,
            tcc_coverage = cfg.task_complexity.tcc_coverage_threshold,
            k_soft = cfg.task_complexity.k_soft,
            k_type = cfg.task_complexity.k_type,
            k_cross = cfg.task_complexity.k_cross,
            k_heavy = cfg.task_complexity.k_heavy,
            n_probe = cfg.task_complexity.n_probe,
            n_eff_complex_threshold = cfg.task_complexity.n_eff_complex_threshold,
            "[task_complexity] shadow_mode={} ({})",
            cfg.task_complexity.shadow_mode,
            "quadrant routing active"
        );
    }

    // [srani] block — WARN if !enabled
    if cfg.srani.enabled {
        tracing::info!(
            enabled = cfg.srani.enabled,
            adaptive = cfg.srani.adaptive,
            ema_alpha = cfg.srani.ema_alpha,
            temperature = cfg.srani.temperature,
            gate_threshold = cfg.srani.gate_threshold,
            warn_threshold = cfg.srani.warn_threshold,
            inject_threshold = cfg.srani.inject_threshold,
            "[srani] enabled={} ({})",
            cfg.srani.enabled,
            "hint injection active"
        );
    } else {
        tracing::warn!(
            enabled = cfg.srani.enabled,
            adaptive = cfg.srani.adaptive,
            ema_alpha = cfg.srani.ema_alpha,
            temperature = cfg.srani.temperature,
            gate_threshold = cfg.srani.gate_threshold,
            warn_threshold = cfg.srani.warn_threshold,
            inject_threshold = cfg.srani.inject_threshold,
            "[srani] enabled={} ({})",
            cfg.srani.enabled,
            "hint injection disabled"
        );
    }

    // [hitl] block — WARN if !enabled
    if cfg.hitl.enabled {
        tracing::info!(
            enabled = cfg.hitl.enabled,
            confidence_threshold = cfg.hitl.confidence_threshold,
            timeout_ms = cfg.hitl.timeout_ms,
            "[hitl] enabled={} ({})",
            cfg.hitl.enabled,
            "human approval gate active"
        );
    } else {
        tracing::warn!(
            enabled = cfg.hitl.enabled,
            confidence_threshold = cfg.hitl.confidence_threshold,
            timeout_ms = cfg.hitl.timeout_ms,
            "[hitl] enabled={} ({})",
            cfg.hitl.enabled,
            "human approval gate bypassed"
        );
    }
}

/// Post-deserialization step called inside `load_layered()`.
///
/// For non-Custom profiles, overwrites all safety fields from the canonical profile.
/// Only `shadow_auditor` tuning sub-fields are preserved from operator override.
pub const fn apply_safety_profile(cfg: &mut H2AIConfig) {
    match cfg.safety.profile {
        SafetyProfile::Development => {
            cfg.safety.krum_fault_tolerance = 0;
            cfg.safety.krum_threshold = 0.30;
            cfg.safety.diversity_threshold = 0.0;
            cfg.safety.family_constraint = FamilyConstraint::SingleFamilyOk;
            cfg.safety.require_bivariate_cg = false;
            cfg.safety.shadow_auditor.enabled = false;
            cfg.safety.shadow_auditor.strict = false;
        }
        SafetyProfile::Production => {
            cfg.safety.krum_fault_tolerance = 1;
            cfg.safety.krum_threshold = 0.30;
            cfg.safety.diversity_threshold = 0.15;
            cfg.safety.family_constraint = FamilyConstraint::RequireDiverse;
            cfg.safety.require_bivariate_cg = false;
            cfg.safety.shadow_auditor.enabled = true;
            cfg.safety.shadow_auditor.strict = true;
        }
        SafetyProfile::Strict => {
            cfg.safety.krum_fault_tolerance = 2;
            cfg.safety.krum_threshold = 0.20;
            cfg.safety.diversity_threshold = 0.20;
            cfg.safety.family_constraint = FamilyConstraint::RequireDiverse;
            cfg.safety.require_bivariate_cg = true;
            cfg.safety.shadow_auditor.enabled = true;
            cfg.safety.shadow_auditor.strict = true;
        }
        SafetyProfile::Custom => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_i1_config_defaults() {
        let cfg = GapI1Config::default();
        assert!(!cfg.enabled, "I1 must be off by default");
        assert_eq!(cfg.cold_check_threshold, 0.0);
        assert!(cfg.synthesis_min_confidence > 0.5);
        assert!(cfg.max_gap_records_per_wave >= 1);
        assert!(cfg.researcher_timeout_secs > 0);
    }

    #[test]
    fn h2ai_config_has_gap_i1_field() {
        let cfg = H2AIConfig::default();
        let _ = cfg.gap_i1; // field must exist
    }

    #[test]
    fn gap_k1_config_defaults() {
        let cfg = GapK1Config::default();
        assert!(!cfg.enabled);
        assert!(!cfg.auto_repair_enabled);
        assert!((cfg.coherence_threshold - 0.80).abs() < 1e-9);
        assert!((cfg.instability_threshold - 0.10).abs() < 1e-9);
        assert!((cfg.repair_acceptance_threshold - 0.90).abs() < 1e-9);
        assert_eq!(cfg.probe_runs, 5);
        assert_eq!(cfg.repair_candidates, 3);
        assert_eq!(cfg.probe_cache_ttl_secs, 86400);
    }

    #[test]
    fn h2ai_config_has_gap_k1_field() {
        let cfg = H2AIConfig::default();
        let _ = cfg.gap_k1;
    }

    #[test]
    fn shadow_auditor_config_strict_default_is_false() {
        let cfg = ShadowAuditorConfig::default();
        assert!(!cfg.strict, "strict must be false by default");
    }

    #[test]
    fn apply_safety_profile_sets_strict_for_production() {
        let mut cfg = H2AIConfig::default();
        cfg.safety.profile = SafetyProfile::Production;
        apply_safety_profile(&mut cfg);
        assert!(cfg.safety.shadow_auditor.strict);
    }

    #[test]
    fn apply_safety_profile_sets_strict_for_strict_profile() {
        let mut cfg = H2AIConfig::default();
        cfg.safety.profile = SafetyProfile::Strict;
        apply_safety_profile(&mut cfg);
        assert!(cfg.safety.shadow_auditor.strict);
    }

    #[test]
    fn apply_safety_profile_keeps_strict_false_for_development() {
        let mut cfg = H2AIConfig::default();
        cfg.safety.profile = SafetyProfile::Development;
        apply_safety_profile(&mut cfg);
        assert!(!cfg.safety.shadow_auditor.strict);
    }
}
