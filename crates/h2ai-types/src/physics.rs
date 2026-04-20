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
    #[error("tau must be in [0, 1], got {0}")]
    InvalidTau(f64),
    #[error("cg_samples must not be empty")]
    EmptyCgSamples,
    #[error("beta_base must be ≥ 0, got {0}")]
    InvalidBetaBase(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TauValue(f64);

impl TauValue {
    pub fn new(v: f64) -> Result<Self, PhysicsError> {
        if (0.0..=1.0).contains(&v) {
            Ok(Self(v))
        } else {
            Err(PhysicsError::InvalidTau(v))
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Calibrated coherency parameters for a set of compute adapters.
///
/// `alpha` is the contention (serial-fraction) coefficient from USL calibration.
/// `beta_base` (β₀) is the base coherency cost per agent pair measured from calibration timing.
/// `beta_eff` = β₀ / CG_mean couples coordination cost with how much common ground agents share.
/// `n_max` = round(√((1−α)/β_eff)) is derived from USL Proposition 1 by setting dX/dN = 0
/// in X(N) = N / (1 + α(N−1) + β·N(N−1)).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherencyCoefficients {
    pub alpha: f64,
    /// Base coherency cost per agent pair (β₀), measured from calibration timing.
    /// Lower CG_mean raises β_eff = β₀/CG_mean, reducing N_max.
    #[serde(alias = "kappa_base")]
    pub beta_base: f64,
    pub cg_samples: Vec<f64>,
}

impl CoherencyCoefficients {
    pub fn new(alpha: f64, beta_base: f64, cg_samples: Vec<f64>) -> Result<Self, PhysicsError> {
        if !(0.0..1.0).contains(&alpha) {
            return Err(PhysicsError::InvalidAlpha(alpha));
        }
        if beta_base < 0.0 {
            return Err(PhysicsError::InvalidBetaBase(beta_base));
        }
        if cg_samples.is_empty() {
            return Err(PhysicsError::EmptyCgSamples);
        }
        Ok(Self {
            alpha,
            beta_base,
            cg_samples,
        })
    }

    /// Effective coordination cost per agent pair: β_eff = β₀ / CG_mean.
    ///
    /// Higher common ground (CG_mean → 1) reduces effective coordination cost.
    /// Lower common ground amplifies it. Used in `n_max()` via USL Proposition 1.
    pub fn beta_eff(&self) -> f64 {
        self.beta_base / self.cg_mean().max(f64::EPSILON)
    }

    /// Maximum useful ensemble size from USL Proposition 1: round(√((1−α)/β_eff)).
    ///
    /// Derived by setting dX/dN = 0 in X(N) = N / (1 + α(N−1) + β·N(N−1)).
    /// Acts as a safety ceiling; `EnsembleCalibration::n_optimal` (Condorcet-based)
    /// is the primary ensemble-size target.
    pub fn n_max(&self) -> f64 {
        let beta_eff = self.beta_eff().max(f64::EPSILON);
        ((1.0 - self.alpha).max(0.0) / beta_eff).sqrt().round()
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
    pub fn from_calibration(cc: &CoherencyCoefficients, max: f64) -> Self {
        let spread = cc.cg_mean() - cc.cg_std_dev();
        Self(spread.clamp(0.0, max))
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Low–medium error cost (max c_i ≤ bft_threshold): pick highest-scored surviving proposal.
    ScoreOrdered,
    /// Medium-high error cost (bft_threshold < max c_i ≤ krum_threshold): Condorcet selection.
    /// Picks the proposal with highest mean Jaccard similarity to the rest of the ensemble.
    /// NOTE: not Byzantine-resistant. Vulnerable to coordinated Byzantine proposals at f ≥ n/2.
    ConsensusMedian,
    /// High error cost (max c_i > krum_threshold) with explicit f > 0: Krum single-selection.
    /// Byzantine-resistant for n ≥ 2f+3. Selects the proposal minimising sum of distances
    /// to its n-f-2 nearest neighbours in Jaccard-distance space.
    Krum { f: usize },
    /// Multi-Krum: iteratively select m Byzantine-resistant survivors, then take the
    /// highest verification-scored one. Requires n ≥ 2f+3.
    MultiKrum { f: usize, m: usize },
}

impl MergeStrategy {
    /// Select merge strategy based on role error costs.
    ///
    /// Three-tier selection:
    /// 1. `krum_f > 0` AND `max_ci > krum_threshold` → `Krum { f: krum_f }`
    /// 2. `max_ci > bft_threshold` → `ConsensusMedian`
    /// 3. Otherwise → `ScoreOrdered`
    pub fn from_role_costs(
        costs: &[RoleErrorCost],
        bft_threshold: f64,
        krum_threshold: f64,
        krum_f: usize,
    ) -> Self {
        let max_ci = costs
            .iter()
            .map(|c| c.value())
            .fold(f64::NEG_INFINITY, f64::max);
        if krum_f > 0 && max_ci > krum_threshold {
            MergeStrategy::Krum { f: krum_f }
        } else if max_ci > bft_threshold {
            MergeStrategy::ConsensusMedian
        } else {
            MergeStrategy::ScoreOrdered
        }
    }

    /// Minimum number of proposals needed to safely run Krum/MultiKrum with fault bound f.
    /// Derived from n ≥ 2f + 3 (Blanchard et al. 2017, Theorem 2).
    pub const fn min_krum_quorum(f: usize) -> usize {
        2 * f + 3
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

/// Exponential decay alignment between two τ (creativity temperature) values.
///
/// `tau_alignment(a, b) = exp(-3 × |a − b|)`.
/// Same τ → 1.0. Difference of 1.0 → exp(−3) ≈ 0.05.
pub fn tau_alignment(a: TauValue, b: TauValue) -> f64 {
    (-3.0 * (a.value() - b.value()).abs()).exp()
}

/// Condorcet Jury Theorem ensemble accuracy with error correlation.
///
/// Returns the probability that a majority vote among `n_agents` agents, each correct
/// with probability `p`, is correct — adjusted for pairwise error correlation `rho`.
///
/// Q(N,p,ρ) = p + (Q_independent(N,p) − p) × (1 − ρ).
/// Boundary: N=1 → Q=p; ρ=1 → Q=p.
pub fn condorcet_quality(n_agents: usize, p: f64, rho: f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    let rho = rho.clamp(0.0, 1.0);
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    if n_agents == 0 {
        return 0.0;
    }
    if n_agents == 1 {
        return p;
    }
    let n = n_agents;
    let q_ind = {
        let majority = n / 2 + 1; // strict majority: > N/2 votes needed
        let mut sum = 0.0f64;
        for k in majority..=n {
            let log_term = log_binom_coeff(n, k)
                + k as f64 * p.ln()
                + (n - k) as f64 * (1.0 - p).ln();
            sum += log_term.exp();
        }
        // For even N, exact tie → 0.5 probability of being correct
        if n.is_multiple_of(2) {
            let k = n / 2;
            let log_term = log_binom_coeff(n, k)
                + k as f64 * p.ln()
                + k as f64 * (1.0 - p).ln();
            sum += 0.5 * log_term.exp();
        }
        sum.clamp(0.0, 1.0)
    };
    (p + (q_ind - p) * (1.0 - rho)).clamp(0.0, 1.0)
}

/// Log of binomial coefficient C(n, k) via log-gamma.
fn log_binom_coeff(n: usize, k: usize) -> f64 {
    if k == 0 || k == n {
        return 0.0;
    }
    log_gamma(n + 1) - log_gamma(k + 1) - log_gamma(n - k + 1)
}

/// Computes `ln((n-1)!)` = `ln(Γ(n))`.
/// To get `ln(n!)`, call as `log_gamma(n + 1)`.
fn log_gamma(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    (1..n).map(|i| (i as f64).ln()).sum()
}

/// Condorcet-based calibration result for an ensemble of compute adapters.
///
/// Produced alongside `CoherencyCoefficients` by `CalibrationHarness`.
/// Provides the theoretically optimal ensemble size and the expected quality
/// gain at that size, derived from the Condorcet Jury Theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleCalibration {
    /// Mean per-adapter estimated accuracy proxy: `0.5 + CG_mean / 2`.
    pub p_mean: f64,
    /// Mean pairwise error correlation proxy: `1.0 − CG_mean`.
    pub rho_mean: f64,
    /// Ensemble size that maximises Q(N,p,ρ)/(N+1), capped at 9.
    pub n_optimal: usize,
    /// Expected ensemble quality Q(n_optimal, p_mean, rho_mean).
    pub q_optimal: f64,
}

impl EnsembleCalibration {
    /// Derive calibration from CG_mean using proxy formulas:
    ///   p_mean   = 0.5 + CG_mean / 2   (accuracy proxy)
    ///   rho_mean = 1.0 − CG_mean        (correlation proxy)
    ///
    /// N_optimal = argmax_N (Q(N,p,ρ) − p) / N for N in 1..=max_n.
    /// This is marginal Condorcet gain per agent above the single-agent baseline.
    /// N=1 scores 0; when ρ=1 (no ensemble benefit) all scores are 0 and N=1 is returned.
    pub fn from_cg_mean(cg_mean: f64, max_n: usize) -> Self {
        let cg = cg_mean.clamp(f64::EPSILON, 1.0);
        let rho_mean = (1.0 - cg).clamp(0.0, 1.0);
        let p_mean = (0.5 + cg / 2.0).clamp(0.5, 1.0);
        let max_n = max_n.max(1);
        let (n_optimal, _) = (1..=max_n)
            .map(|n| {
                let q = condorcet_quality(n, p_mean, rho_mean);
                // Marginal gain per agent over single-agent baseline.
                // N=1 scores 0. For ρ=1 all score 0 and max_by returns N=1 (first element).
                let score = (q - p_mean) / n as f64;
                (n, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((1, 0.0));
        let q_optimal = condorcet_quality(n_optimal, p_mean, rho_mean);
        Self { p_mean, rho_mean, n_optimal, q_optimal }
    }

    /// Construct with a directly measured accuracy value, overriding the CG-mean proxy.
    /// Use when `baseline_accuracy_proxy` is set in config from `scripts/baseline_eval.py`.
    /// Uses the same marginal-gain-per-agent scoring as `from_cg_mean`.
    pub fn from_measured_p(p_mean: f64, cg_mean: f64, max_n: usize) -> Self {
        let p = p_mean.clamp(0.5, 1.0);
        let cg = cg_mean.clamp(f64::EPSILON, 1.0);
        let rho_mean = (1.0 - cg).clamp(0.0, 1.0);
        let max_n = max_n.max(1);
        let (n_optimal, _) = (1..=max_n)
            .map(|n| {
                let q = condorcet_quality(n, p, rho_mean);
                let score = (q - p) / n as f64;
                (n, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((1, 0.0));
        let q_optimal = condorcet_quality(n_optimal, p, rho_mean);
        Self { p_mean: p, rho_mean, n_optimal, q_optimal }
    }

    /// Expected quality at a given ensemble size.
    pub fn quality_at_n(&self, n: usize) -> f64 {
        condorcet_quality(n, self.p_mean, self.rho_mean)
    }

    /// Condorcet gain over single-agent baseline: Q(n_optimal) − p_mean.
    pub fn topology_gain(&self) -> f64 {
        (self.q_optimal - self.p_mean).max(0.0)
    }
}

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
        min_competence: f64,
        max_correlation: f64,
    ) -> Result<(), MultiplicationConditionFailure> {
        if baseline_competence <= min_competence {
            return Err(MultiplicationConditionFailure::InsufficientCompetence {
                actual: baseline_competence,
                required: min_competence,
            });
        }
        if error_correlation >= max_correlation {
            return Err(MultiplicationConditionFailure::InsufficientDecorrelation {
                actual: error_correlation,
                threshold: max_correlation,
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

#[cfg(test)]
mod condorcet_tests {
    use super::*;

    #[test]
    fn condorcet_quality_n0_returns_zero() {
        assert_eq!(condorcet_quality(0, 0.7, 0.2), 0.0);
    }

    #[test]
    fn condorcet_quality_p_zero_returns_zero() {
        assert_eq!(condorcet_quality(3, 0.0, 0.2), 0.0);
    }

    #[test]
    fn condorcet_quality_p_one_returns_one() {
        assert_eq!(condorcet_quality(3, 1.0, 0.2), 1.0);
    }

    #[test]
    fn tau_alignment_same_tau_is_one() {
        let a = TauValue::new(0.5).unwrap();
        let b = TauValue::new(0.5).unwrap();
        let result = tau_alignment(a, b);
        assert!((result - 1.0).abs() < 1e-10, "same τ → alignment 1.0, got {result}");
    }

    #[test]
    fn tau_alignment_far_apart_is_small() {
        let a = TauValue::new(0.0).unwrap();
        let b = TauValue::new(1.0).unwrap();
        let result = tau_alignment(a, b);
        // exp(-3 * 1.0) ≈ 0.0498
        assert!(result < 0.06, "τ distance 1.0 → small alignment, got {result}");
        assert!(result > 0.04, "τ distance 1.0 → ~0.05, got {result}");
    }

    #[test]
    fn condorcet_quality_n1_equals_p() {
        for p in [0.3, 0.5, 0.7, 0.9] {
            let q = condorcet_quality(1, p, 0.3);
            assert!((q - p).abs() < 1e-10, "N=1 → Q=p for p={p}, got {q}");
        }
    }

    #[test]
    fn condorcet_quality_full_correlation_equals_p() {
        for n in [3usize, 5, 7] {
            let q = condorcet_quality(n, 0.7, 1.0);
            assert!((q - 0.7).abs() < 1e-10, "ρ=1 → Q=p for N={n}, got {q}");
        }
    }

    #[test]
    fn condorcet_quality_increases_with_n_for_p_above_half() {
        let qs: Vec<f64> = [1usize, 3, 5, 7, 9]
            .iter()
            .map(|&n| condorcet_quality(n, 0.7, 0.2))
            .collect();
        for i in 0..qs.len() - 1 {
            assert!(
                qs[i + 1] >= qs[i],
                "Q should be non-decreasing in N for p=0.7, rho=0.2: {:?}",
                qs
            );
        }
    }

    #[test]
    fn condorcet_quality_bounded_01() {
        for n in [1usize, 3, 5, 7, 9] {
            for p_int in [30i32, 50, 70, 90] {
                let p = p_int as f64 / 100.0;
                for rho_int in [0i32, 20, 50, 80, 100] {
                    let rho = rho_int as f64 / 100.0;
                    let q = condorcet_quality(n, p, rho);
                    assert!(
                        q >= 0.0 && q <= 1.0,
                        "Q out of [0,1]: N={n} p={p} rho={rho} → {q}"
                    );
                }
            }
        }
    }

    #[test]
    fn ensemble_calibration_from_cg_mean_n_optimal_at_least_1() {
        for cg_int in [20i32, 40, 60, 80] {
            let cg = cg_int as f64 / 100.0;
            let ec = EnsembleCalibration::from_cg_mean(cg, 9);
            assert!(ec.n_optimal >= 1, "n_optimal >= 1 for cg={cg}");
            assert!(ec.q_optimal >= ec.p_mean, "q_optimal >= p_mean for cg={cg}");
        }
    }

    #[test]
    fn ensemble_calibration_n_optimal_greater_than_1_for_typical_cg() {
        // For typical CG values (rho < 1), ensemble of >1 agent is cost-optimal
        let ec = EnsembleCalibration::from_cg_mean(0.7, 9);
        assert!(ec.n_optimal > 1, "n_optimal should be >1 for cg=0.7, got {}", ec.n_optimal);
    }

    #[test]
    fn ensemble_calibration_low_cg_still_recommends_small_ensemble() {
        // Even at very high correlation (low CG_mean), any rho < 1 gives a tiny positive
        // Condorcet gain, so n_optimal is still > 1 (but small — N=3 typically).
        let ec = EnsembleCalibration::from_cg_mean(0.001, 9);
        assert!(ec.n_optimal >= 1, "n_optimal must be >= 1, got {}", ec.n_optimal);
        assert!(ec.n_optimal <= 5, "very high correlation should give small n_optimal, got {}", ec.n_optimal);
    }

    #[test]
    fn ensemble_calibration_quality_at_n1_equals_p() {
        let ec = EnsembleCalibration::from_cg_mean(0.7, 9);
        let q = ec.quality_at_n(1);
        assert!((q - ec.p_mean).abs() < 1e-10, "quality_at_n(1) == p_mean, got {q} vs {}", ec.p_mean);
    }

    #[test]
    fn ensemble_calibration_topology_gain_non_negative() {
        for cg_int in [20i32, 50, 80] {
            let cg = cg_int as f64 / 100.0;
            let ec = EnsembleCalibration::from_cg_mean(cg, 9);
            assert!(ec.topology_gain() >= 0.0, "topology_gain >= 0 for cg={cg}");
        }
    }

    #[test]
    fn ensemble_calibration_from_measured_p_uses_given_p() {
        let ec = EnsembleCalibration::from_measured_p(0.9, 0.7, 9);
        assert!((ec.p_mean - 0.9).abs() < 1e-10, "p_mean should be 0.9, got {}", ec.p_mean);
    }
}
