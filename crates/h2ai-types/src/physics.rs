use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PhysicsError {
    #[error("alpha must be in [0, 1), got {0}")]
    InvalidAlpha(f64),
    #[error("c_i must be in [0, 1], got {0}")]
    InvalidErrorCost(f64),
    #[error("J_eff must be in [0, 1], got {0}")]
    InvalidJeff(f64),
    #[error("cg_samples must not be empty")]
    EmptyCgSamples,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherencyCoefficients {
    pub alpha: f64,
    pub kappa_base: f64,
    pub cg_samples: Vec<f64>,
}

impl CoherencyCoefficients {
    pub fn new(alpha: f64, kappa_base: f64, cg_samples: Vec<f64>) -> Result<Self, PhysicsError> {
        if !(0.0..1.0).contains(&alpha) {
            return Err(PhysicsError::InvalidAlpha(alpha));
        }
        if cg_samples.is_empty() {
            return Err(PhysicsError::EmptyCgSamples);
        }
        Ok(Self {
            alpha,
            kappa_base,
            cg_samples,
        })
    }

    pub fn kappa_eff(&self) -> f64 {
        let mean_cg = self.cg_mean();
        self.kappa_base / mean_cg
    }

    pub fn n_max(&self) -> f64 {
        ((1.0 - self.alpha) / self.kappa_eff()).sqrt()
    }

    pub fn cg_mean(&self) -> f64 {
        self.cg_samples.iter().sum::<f64>() / self.cg_samples.len() as f64
    }

    pub fn cg_std_dev(&self) -> f64 {
        let mean = self.cg_mean();
        let variance = self
            .cg_samples
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / self.cg_samples.len() as f64;
        variance.sqrt()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordinationThreshold(f64);

impl CoordinationThreshold {
    pub fn from_calibration(cc: &CoherencyCoefficients) -> Self {
        let spread = cc.cg_mean() - cc.cg_std_dev();
        Self(spread.clamp(0.0, 0.3_f64))
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleErrorCost(f64);

impl RoleErrorCost {
    pub fn new(value: f64) -> Result<Self, PhysicsError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(PhysicsError::InvalidErrorCost(value));
        }
        Ok(Self(value))
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

const BFT_THRESHOLD: f64 = 0.85;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    CrdtSemilattice,
    BftConsensus,
}

impl MergeStrategy {
    pub fn from_role_costs(costs: &[RoleErrorCost]) -> Self {
        let max_ci = costs
            .iter()
            .map(|c| c.value())
            .fold(f64::NEG_INFINITY, f64::max);
        if max_ci > BFT_THRESHOLD {
            MergeStrategy::BftConsensus
        } else {
            MergeStrategy::CrdtSemilattice
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JeffectiveGap(f64);

impl JeffectiveGap {
    pub fn new(value: f64) -> Result<Self, PhysicsError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(PhysicsError::InvalidJeff(value));
        }
        Ok(Self(value))
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    pub fn is_below_threshold(&self, threshold: f64) -> bool {
        self.0 < threshold
    }
}

pub struct MultiplicationCondition;

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum MultiplicationConditionFailure {
    #[error("baseline competence {actual:.2} < required {required:.2}")]
    InsufficientCompetence { actual: f64, required: f64 },
    #[error("error correlation ρ={actual:.2} ≥ threshold {threshold:.2} — explorers too similar")]
    InsufficientDecorrelation { actual: f64, threshold: f64 },
    #[error("CG_mean {cg_mean:.2} < θ_coord {theta:.2} — common ground below floor")]
    CommonGroundBelowFloor { cg_mean: f64, theta: f64 },
}

impl MultiplicationCondition {
    pub fn evaluate(
        baseline_competence: f64,
        error_correlation: f64,
        cg_mean: f64,
        theta_coord: f64,
    ) -> Result<(), MultiplicationConditionFailure> {
        if baseline_competence <= 0.5 {
            return Err(MultiplicationConditionFailure::InsufficientCompetence {
                actual: baseline_competence,
                required: 0.5,
            });
        }
        if error_correlation >= 0.9 {
            return Err(MultiplicationConditionFailure::InsufficientDecorrelation {
                actual: error_correlation,
                threshold: 0.9,
            });
        }
        if cg_mean < theta_coord {
            return Err(MultiplicationConditionFailure::CommonGroundBelowFloor {
                cg_mean,
                theta: theta_coord,
            });
        }
        Ok(())
    }
}
