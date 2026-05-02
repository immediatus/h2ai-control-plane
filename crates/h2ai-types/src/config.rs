use crate::identity::ExplorerId;
use crate::physics::TauValue;
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentRole {
    Coordinator,
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
    pub fn default_tau(&self) -> f64 {
        match self {
            Self::Coordinator => 0.05,
            Self::Executor => 0.40,
            Self::Evaluator => 0.10,
            Self::Synthesizer => 0.80,
            Self::Custom { tau, .. } => tau.value(),
        }
    }

    pub fn default_role_error_cost(&self) -> f64 {
        match self {
            Self::Coordinator => 0.1,
            Self::Executor => 0.5,
            Self::Evaluator => 0.9,
            Self::Synthesizer => 0.1,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterKind {
    LocalLlamaCpp {
        model_path: PathBuf,
        n_threads: usize,
    },
    CloudGeneric {
        endpoint: String,
        api_key_env: String,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerConfig {
    pub explorer_id: ExplorerId,
    pub tau: TauValue,
    pub adapter: AdapterKind,
    pub role: Option<AgentRole>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditorConfig {
    pub adapter: AdapterKind,
    pub tau: TauValue,
    pub max_tokens: u64,
    pub prompt_template: String,
}

impl Default for AuditorConfig {
    fn default() -> Self {
        Self {
            adapter: AdapterKind::CloudGeneric {
                endpoint: String::new(),
                api_key_env: String::new(),
            },
            tau: TauValue::new(0.1).unwrap(),
            max_tokens: 256,
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
    /// Jaccard similarity threshold above which successive TAO turn outputs are
    /// considered a stuck repetition loop. When exceeded on a failed turn, the
    /// loop returns Err immediately. Set to a value > 1.0 to disable.
    pub repetition_threshold: f64,
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
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.45,
            rubric: crate::prompts::COT_RUBRIC.into(),
            evaluator_system_prompt: crate::prompts::EVALUATOR_SYSTEM_PROMPT.into(),
            evaluator_tau: TauValue::new(0.1).unwrap(),
            evaluator_max_tokens: 128,
        }
    }
}
