use crate::identity::ExplorerId;
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
        tau: f64,
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
            Self::Custom { tau, .. } => *tau,
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
    pub tau: Option<f64>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerConfig {
    pub explorer_id: ExplorerId,
    pub tau: f64,
    pub adapter: AdapterKind,
    pub role: Option<AgentRole>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditorConfig {
    pub adapter: AdapterKind,
}

impl AuditorConfig {
    pub fn tau(&self) -> f64 {
        0.0
    }
}
