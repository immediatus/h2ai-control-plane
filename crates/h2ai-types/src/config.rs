use crate::identity::ExplorerId;
use crate::sizing::TauValue;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("ParetoWeights must sum to 1.0 (got {0:.4})")]
    InvalidWeightSum(f64),
    #[error("All ParetoWeights must be non-negative")]
    NegativeWeight,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoWeights {
    pub throughput: f64,
    pub containment: f64,
    pub diversity: f64,
}

impl ParetoWeights {
    /// Construct a validated set of Pareto weights.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::NegativeWeight`] when any component is negative,
    /// or [`ConfigError::InvalidWeightSum`] when the components do not sum to
    /// `1.0` within `1e-6`.
    pub fn new(throughput: f64, containment: f64, diversity: f64) -> Result<Self, ConfigError> {
        if throughput < 0.0 || containment < 0.0 || diversity < 0.0 {
            return Err(ConfigError::NegativeWeight);
        }
        let sum = throughput + containment + diversity;
        if (sum - 1.0).abs() > 1e-6 {
            return Err(ConfigError::InvalidWeightSum(sum));
        }
        Ok(Self {
            throughput,
            containment,
            diversity,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum AgentRole {
    Coordinator,
    #[default]
    Executor,
    Evaluator,
    Synthesizer,
    Custom {
        name: String,
        tau: TauValue,
        role_error_cost: f64,
    },
}

impl AgentRole {
    #[must_use]
    pub const fn default_tau(&self) -> f64 {
        match self {
            Self::Coordinator => 0.05,
            Self::Executor => 0.40,
            Self::Evaluator => 0.10,
            Self::Synthesizer => 0.80,
            Self::Custom { tau, .. } => tau.value(),
        }
    }

    #[must_use]
    pub const fn default_role_error_cost(&self) -> f64 {
        match self {
            Self::Coordinator | Self::Synthesizer => 0.1,
            Self::Executor => 0.5,
            Self::Evaluator => 0.9,
            Self::Custom {
                role_error_cost, ..
            } => *role_error_cost,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleSpec {
    pub agent_id: String,
    pub role: AgentRole,
    pub tau: Option<TauValue>,
    pub role_error_cost: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewGate {
    pub reviewer: String,
    pub blocks: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyKind {
    Ensemble,
    HierarchicalTree { branching_factor: Option<u8> },
    TeamSwarmHybrid,
}

/// Coarse model-family grouping for judge-panel bias-diversity detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdapterFamily {
    Anthropic,
    OpenAI,
    /// `LocalLlamaCpp` and Ollama — locally-served models.
    Local,
    /// `CloudGeneric` and `A2a` — endpoint-based adapters without vendor-specific family.
    Cloud,
}

impl AdapterFamily {
    #[must_use]
    pub const fn from_kind(kind: &AdapterKind) -> Self {
        match kind {
            AdapterKind::Anthropic { .. } => Self::Anthropic,
            AdapterKind::OpenAI { .. } => Self::OpenAI,
            AdapterKind::LocalLlamaCpp { .. } | AdapterKind::Ollama { .. } => Self::Local,
            AdapterKind::CloudGeneric { .. } | AdapterKind::A2a { .. } => Self::Cloud,
        }
    }
}

impl std::fmt::Display for AdapterFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAI => write!(f, "openai"),
            Self::Local => write!(f, "local"),
            Self::Cloud => write!(f, "cloud"),
        }
    }
}

/// Which wire-protocol/thinking-control dialect a `CloudGeneric` adapter speaks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CloudProvider {
    /// Plain OpenAI-compatible endpoint — no extra thinking fields sent.
    #[default]
    Generic,
    /// llama.cpp server — uses `chat_template_kwargs: {"enable_thinking": <bool>}`.
    LlamaCpp,
    /// Google Gemini — uses `thinking_config: {"thinking_budget": <-1|0>}`.
    Gemini,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterKind {
    LocalLlamaCpp {
        model_path: PathBuf,
        n_threads: usize,
    },
    CloudGeneric {
        endpoint: String,
        api_key_env: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default)]
        provider: CloudProvider,
    },
    OpenAI {
        api_key_env: String,
        model: String,
    },
    Anthropic {
        api_key_env: String,
        model: String,
    },
    Ollama {
        endpoint: String,
        model: String,
    },
    A2a {
        endpoint: String,
        auth_scheme: String,    // "bearer", "api_key", "none"
        auth_token_env: String, // env var name; empty string when auth_scheme = "none"
        timeout_minutes: u64,
        poll_interval_ms: u64,
        max_poll_interval_ms: u64,
        agent_card_cache_ttl_s: u64,
    },
}

impl AdapterKind {
    #[must_use]
    pub const fn family(&self) -> AdapterFamily {
        AdapterFamily::from_kind(self)
    }

    #[must_use]
    pub fn model_lineage_key(&self) -> String {
        match self {
            Self::CloudGeneric {
                endpoint,
                model,
                provider,
                ..
            } => {
                let provider_str = format!("{provider:?}").to_lowercase();
                let model_str = model.as_deref().unwrap_or("unknown");
                format!("cloud::{provider_str}::{endpoint}::{model_str}")
            }
            Self::OpenAI { model, .. } => format!("openai::{model}"),
            Self::Anthropic { model, .. } => format!("anthropic::{model}"),
            Self::Ollama { endpoint, model } => format!("ollama::{endpoint}::{model}"),
            Self::LocalLlamaCpp { model_path, .. } => {
                format!("local::{}", model_path.display())
            }
            Self::A2a { endpoint, .. } => format!("a2a::{endpoint}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerConfig {
    pub explorer_id: ExplorerId,
    pub tau: TauValue,
    pub adapter: AdapterKind,
    pub role: Option<AgentRole>,
    /// When `true`, the TAO retry loop is bypassed and the adapter is called exactly once.
    /// Set this for models with built-in chain-of-thought (`DeepSeek` R1, o1, o3, o4-mini)
    /// to avoid α-spike from injecting the model's own reasoning trace back as TAO memory.
    #[serde(default)]
    pub is_reasoning_model: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditorConfig {
    pub adapter: AdapterKind,
    pub tau: TauValue,
    pub max_tokens: u64,
    #[serde(default = "crate::prompts::auditor_system_prompt_default")]
    pub system_prompt: String,
    pub prompt_template: String,
}

impl Default for AuditorConfig {
    fn default() -> Self {
        Self {
            adapter: AdapterKind::CloudGeneric {
                endpoint: String::new(),
                api_key_env: String::new(),
                model: None,
                provider: CloudProvider::default(),
            },
            tau: TauValue::new(0.1).unwrap(),
            max_tokens: 4096,
            system_prompt: crate::prompts::AUDITOR_SYSTEM_PROMPT.into(),
            prompt_template: crate::prompts::AUDITOR_PROMPT_TEMPLATE.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaoConfig {
    pub max_turns: u8,
    pub verify_pattern: Option<String>,
    pub observation_pass: String,
    pub observation_fail_pattern: String,
    pub observation_fail_schema: String,
    pub retry_instruction: String,
    /// Token-overlap similarity threshold above which successive TAO turn outputs are
    /// considered a stuck repetition loop. When exceeded on a failed turn, the
    /// loop returns Err immediately. Set to a value > 1.0 to disable.
    pub repetition_threshold: f64,
    /// Per-turn adapter call timeout in seconds. Increase for slow local models.
    pub per_turn_timeout_secs: u64,
    /// When `true`, retry once on turn-1 (or bypass) timeout with a reduced
    /// `max_tokens` cap to get a faster response from slow local LLMs.
    #[serde(default = "default_retry_on_timeout")]
    pub retry_on_timeout: bool,
    /// `max_tokens` cap used on the single retry request after a timeout.
    /// Smaller values elicit faster responses from slow local models.
    #[serde(default = "default_timeout_retry_max_tokens")]
    pub timeout_retry_max_tokens: u32,
}

fn default_retry_on_timeout() -> bool {
    true
}

fn default_timeout_retry_max_tokens() -> u32 {
    512
}

impl Default for TaoConfig {
    fn default() -> Self {
        Self {
            max_turns: 3,
            verify_pattern: None,
            observation_pass: crate::prompts::TAO_OBSERVATION_PASS.into(),
            observation_fail_pattern: crate::prompts::TAO_OBSERVATION_FAIL_PATTERN.into(),
            observation_fail_schema: crate::prompts::TAO_OBSERVATION_FAIL_SCHEMA.into(),
            retry_instruction: crate::prompts::TAO_RETRY_INSTRUCTION.into(),
            repetition_threshold: 0.92,
            per_turn_timeout_secs: 600,
            retry_on_timeout: default_retry_on_timeout(),
            timeout_retry_max_tokens: default_timeout_retry_max_tokens(),
        }
    }
}

/// Optional JSON schema config for validating TAO loop output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputSchemaConfig {
    /// JSON Schema string (Draft 7 / 2019-09 / 2020-12 — any jsonschema-supported format).
    pub schema_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    pub threshold: f64,
    pub rubric: String,
    pub evaluator_system_prompt: String,
    pub evaluator_tau: TauValue,
    pub evaluator_max_tokens: u64,
    /// Per-LlmJudge call timeout in seconds. Defaults to 600s (10 min).
    /// Set from `oracle_timeout_secs` in the main config.
    #[serde(default = "default_evaluator_timeout_secs")]
    pub evaluator_timeout_secs: u64,
    /// When true, run both standard and adversarial verifiers and emit `VerifierComparisonEvent`.
    /// Does not affect pruning decisions. Off by default; enable only for measurement runs.
    #[serde(default)]
    pub record_adversarial_comparison: bool,
    /// Multiplicative scale applied to Hard constraint thresholds. Default 1.0 (no relaxation).
    /// Set to `0.9^retry_count` on retry waves to relax thresholds by 10% per wave.
    #[serde(default = "default_constraint_threshold_scale")]
    pub constraint_threshold_scale: f64,
}

const fn default_evaluator_timeout_secs() -> u64 {
    600
}

fn default_constraint_threshold_scale() -> f64 {
    1.0
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.45,
            rubric: crate::prompts::COT_RUBRIC.into(),
            evaluator_system_prompt: crate::prompts::EVALUATOR_SYSTEM_PROMPT.into(),
            evaluator_tau: TauValue::new(0.1).unwrap(),
            evaluator_max_tokens: 32768,
            evaluator_timeout_secs: default_evaluator_timeout_secs(),
            record_adversarial_comparison: false,
            constraint_threshold_scale: 1.0,
        }
    }
}
