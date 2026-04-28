use nalgebra::DMatrix;
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
/// `beta_eff` = β₀ × (1 − CG_mean) couples coordination cost with how divergent adapter outputs are.
/// Higher CG_mean → lower β_eff → higher N_max. Bounded at β₀ when CG_mean = 0.
/// `n_max` = round(√((1−α)/β_eff)) is derived from USL Proposition 1 by setting dX/dN = 0
/// in X(N) = N / (1 + α(N−1) + β·N(N−1)).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherencyCoefficients {
    pub alpha: f64,
    /// Base coherency cost per agent pair (β₀), measured from calibration timing.
    /// β_eff = β₀ × (1 − CG_mean); bounded at β₀ when CG_mean = 0.
    #[serde(alias = "kappa_base")]
    pub beta_base: f64,
    pub cg_samples: Vec<f64>,
    /// Unix timestamps (seconds) for each entry in `cg_samples`.
    ///
    /// When present (same length as `cg_samples`), `beta_eff_temporal` applies
    /// Ebbinghaus decay so stale CG measurements contribute less weight, causing
    /// β_eff to drift toward the conservative ceiling (β₀) without re-calibrating.
    /// Empty when constructed via `new()` — `beta_eff_temporal` then falls back to
    /// the unweighted `beta_eff()`.
    #[serde(default)]
    pub sample_timestamps: Vec<u64>,
}

/// Time constant τ for CG sample decay under exponential temporal weighting.
/// 7 days: a sample one week old contributes at e^−1 ≈ 37% weight (`exp(-t/τ)` at t=τ).
pub const CG_HALFLIFE_SECS: u64 = 604_800;

impl CoherencyCoefficients {
    pub fn new(alpha: f64, beta_base: f64, cg_samples: Vec<f64>) -> Result<Self, PhysicsError> {
        Self::new_with_timestamps(alpha, beta_base, cg_samples, vec![])
    }

    /// Construct with per-sample Unix timestamps (seconds) for temporal decay.
    ///
    /// `sample_timestamps` must be the same length as `cg_samples`; pass an
    /// empty vec to skip temporal weighting (equivalent to `new`).
    pub fn new_with_timestamps(
        alpha: f64,
        beta_base: f64,
        cg_samples: Vec<f64>,
        sample_timestamps: Vec<u64>,
    ) -> Result<Self, PhysicsError> {
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
            sample_timestamps,
        })
    }

    /// Effective coordination cost per agent pair: `β_eff = β₀ × (1 − CG_mean)`.
    ///
    /// - At CG_mean = 0 (no overlap): β_eff = β₀ (maximum cost, bounded).
    /// - At CG_mean = 1 (full overlap): β_eff ≈ 0 (coordination-free).
    /// - Previous formula β₀/CG_mean diverged at CG→0; this form is bounded everywhere.
    pub fn beta_eff(&self) -> f64 {
        let cg = self.cg_mean().clamp(0.0, 1.0);
        (self.beta_base * (1.0 - cg)).max(1e-6)
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

    /// N_max adjusted for context-window pressure.
    ///
    /// As N agents each contribute `proposal_tokens` to a context of `max_tokens`,
    /// fill fraction `f(N) = min(1, N × proposal_tokens / max_tokens)` rises.
    /// Context pressure amplifies β: `β_ctx(N) = β_eff × (1 + γ × f(N))`.
    ///
    /// Solves `N = √((1−α) / β_ctx(N))` iteratively (converges in ≤ 5 steps).
    ///
    /// Returns `n_max()` when `proposal_tokens` or `max_tokens` is < 1.0.
    /// Result is always ≥ 1.0.
    pub fn n_max_context_aware(&self, proposal_tokens: f64, max_tokens: f64, gamma: f64) -> f64 {
        if proposal_tokens < 1.0 || max_tokens < 1.0 {
            return self.n_max();
        }
        let beta_eff = self.beta_eff().max(f64::EPSILON);
        let alpha = self.alpha;
        let mut n = self.n_max();
        for _ in 0..5 {
            let fill = (n * proposal_tokens / max_tokens).min(1.0);
            let beta_ctx = beta_eff * (1.0 + gamma * fill);
            let n_new = ((1.0 - alpha).max(0.0) / beta_ctx.max(f64::EPSILON))
                .sqrt()
                .round();
            if (n_new - n).abs() < 0.5 {
                break;
            }
            n = n_new;
        }
        n.max(1.0)
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

    /// Effective coordination cost with Ebbinghaus temporal decay.
    ///
    /// Each CG sample is weighted by `e^(-(now_secs − t) / CG_HALFLIFE_SECS)` using
    /// `self.sample_timestamps`. As samples age their contribution fades, causing β_eff
    /// to drift toward the conservative ceiling (β₀) without explicit re-calibration.
    ///
    /// Falls back to `beta_eff()` (unweighted) when `sample_timestamps` is empty or
    /// its length does not match `cg_samples` — no timing information is available.
    ///
    /// Timestamps after `now_secs` are treated as age zero (weight 1.0) via saturating
    /// subtraction — a future timestamp is never penalised.
    pub fn beta_eff_temporal(&self, now_secs: u64) -> f64 {
        let ts = &self.sample_timestamps;
        if ts.len() != self.cg_samples.len() || ts.is_empty() {
            return self.beta_eff();
        }
        let halflife = CG_HALFLIFE_SECS as f64;
        let (weighted_cg, total_weight) = self
            .cg_samples
            .iter()
            .zip(ts)
            .fold((0.0f64, 0.0f64), |(wsum, wt), (cg, &t)| {
                let w = (-(now_secs.saturating_sub(t) as f64) / halflife).exp();
                (wsum + cg * w, wt + w)
            });
        // 1e-15: fires only when every sample has decayed beyond ~35 half-lives
        // (~245 years at the 7-day halflife). Return beta_base — most conservative.
        if total_weight < 1e-15 {
            return self.beta_base.max(1e-6);
        }
        let cg_eff = (weighted_cg / total_weight).clamp(0.0, 1.0);
        (self.beta_base * (1.0 - cg_eff)).max(1e-6)
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

    /// Information-theoretic optimal ensemble size from `rho_mean`.
    ///
    /// See [`n_it_optimal`] for the derivation.
    pub fn n_it_optimal(&self) -> usize {
        n_it_optimal(self.rho_mean)
    }
}

/// Information-theoretic optimal ensemble size.
///
/// Returns the smallest N where the marginal information gain drops below half
/// of the per-adapter entropy:
///
/// ```text
/// I_marginal(N) = H(X) × (1 − ρ)^(N-1) < 0.5 × H(X)
/// ⟹  (1 − ρ)^(N-1) < 0.5
/// ⟹  N > 1 + log(0.5) / log(1 − ρ)
/// N_it_optimal = ceil(1 + log(0.5) / log(1 − ρ))
/// ```
///
/// Matches Condorcet `n_optimal` within ±1 for ρ ∈ [0.3, 0.95] (typical LLM ensembles).
/// At ρ → 0 (independent sources) returns 1; at ρ → 1 returns 9 (capped).
pub fn n_it_optimal(rho: f64) -> usize {
    if rho <= 1e-10 {
        return 1;
    }
    if rho >= 1.0 - 1e-10 {
        return 9;
    }
    let n = 1.0 + 0.5_f64.ln() / (1.0 - rho).ln();
    (n.ceil() as usize).clamp(1, 9)
}

/// Eigenvalue-based ensemble calibration from the pairwise CG similarity matrix.
///
/// Implements the portfolio theory "participation ratio" (Choueifaty & Coignard 2008):
///   N_eff = (Σ λᵢ)² / Σ λᵢ²
///
/// At full independence (Σ = I), N_eff = N. At full correlation (Σ = 𝟏𝟏ᵀ), N_eff = 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EigenCalibration {
    /// Effective number of independent adapters: (Σλ)²/Σλ².
    pub n_effective: f64,
    /// Normalized Shannon entropy of eigenvalue distribution ∈ [0, 1].
    /// 1.0 = fully decorrelated; 0.0 = one adapter dominates.
    pub h_diversity: f64,
    /// Eigenvalues of the CG similarity matrix, sorted descending.
    pub eigenvalues: Vec<f64>,
    /// Recommended adapter count: first N where adding another raises N_eff by < 0.05.
    pub n_pruned: usize,
}

impl EigenCalibration {
    /// Compute from an N×N symmetric positive-semidefinite CG similarity matrix.
    pub fn from_cg_matrix(sigma: &DMatrix<f64>) -> Self {
        let eig = sigma.clone().symmetric_eigen();
        let mut evs: Vec<f64> = eig.eigenvalues.iter().copied().map(|v| v.max(0.0)).collect();
        evs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = evs.iter().sum();
        let sum_sq: f64 = evs.iter().map(|l| l * l).sum();
        let n_eff = if sum_sq > 1e-12 { sum * sum / sum_sq } else { 1.0 };

        let h_div: f64 = evs
            .iter()
            .filter(|&&l| l > 1e-12)
            .map(|&l| { let p = l / sum; -p * p.ln() })
            .sum();
        let h_norm = if evs.len() > 1 {
            h_div / (evs.len() as f64).ln()
        } else {
            0.0
        };

        let n_pruned = {
            let mut prev = 0.0f64;
            let mut pruned = evs.len();
            for (i, _) in evs.iter().enumerate() {
                let partial_sum: f64 = evs[..=i].iter().sum();
                let partial_sum_sq: f64 = evs[..=i].iter().map(|l| l * l).sum();
                let current = if partial_sum_sq > 1e-12 {
                    partial_sum * partial_sum / partial_sum_sq
                } else {
                    1.0
                };
                if i > 0 && current - prev < 0.05 {
                    pruned = i;
                    break;
                }
                prev = current;
            }
            pruned.max(1)
        };

        Self {
            n_effective: n_eff,
            h_diversity: h_norm.clamp(0.0, 1.0),
            eigenvalues: evs,
            n_pruned,
        }
    }

    /// Derive effective correlation from N_eff: ρ_eff = 1 − N_eff/N.
    pub fn rho_eff(&self, n: usize) -> f64 {
        (1.0 - self.n_effective / n as f64).clamp(0.0, 1.0)
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
mod context_aware_tests {
    use super::*;

    #[test]
    fn n_max_context_aware_equals_n_max_when_no_pressure() {
        // Huge context budget → fill ≈ 0 → β_ctx ≈ β_eff → same N_max
        let cc = CoherencyCoefficients::new(0.15, 0.039, vec![0.4]).unwrap();
        let n_base = cc.n_max();
        let n_ctx = cc.n_max_context_aware(1024.0, 1_000_000.0, 0.5);
        assert!((n_ctx - n_base).abs() < 1.0, "no pressure: n_ctx={n_ctx} n_base={n_base}");
    }

    #[test]
    fn n_max_context_aware_reduces_n_when_context_full() {
        // Tiny context: fill reaches 1 well before N_max
        let cc = CoherencyCoefficients::new(0.15, 0.039, vec![0.4]).unwrap();
        let n_base = cc.n_max();
        // max_tokens = 512, proposal_tokens = 1024 → fill(N) = min(1, N*1024/512) ≥ 1 for any N≥1
        let n_ctx = cc.n_max_context_aware(1024.0, 512.0, 0.5);
        assert!(n_ctx <= n_base, "pressure must reduce N_max: n_ctx={n_ctx} n_base={n_base}");
        assert!(n_ctx >= 1.0, "must be at least 1 agent");
    }

    #[test]
    fn n_max_context_aware_clamps_at_one() {
        // Extreme beta: even N=1 is near the ceiling; pressure pushes to floor
        let cc = CoherencyCoefficients::new(0.15, 0.5, vec![0.0]).unwrap(); // β_eff = β₀(1-0)=0.5
        let n_ctx = cc.n_max_context_aware(512.0, 256.0, 1.0);
        assert!(n_ctx >= 1.0, "minimum 1 agent always");
    }

    #[test]
    fn n_max_context_aware_falls_back_when_tokens_zero() {
        let cc = CoherencyCoefficients::new(0.15, 0.039, vec![0.4]).unwrap();
        let n_base = cc.n_max();
        // proposal_tokens = 0 → fallback to n_max()
        let n_ctx = cc.n_max_context_aware(0.0, 1000.0, 0.5);
        assert!((n_ctx - n_base).abs() < 0.5, "zero proposal tokens must fall back to n_max()");
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

    #[test]
    fn beta_eff_temporal_fresh_sample_equals_beta_eff() {
        let now = 1_000_000u64;
        let cc = CoherencyCoefficients::new_with_timestamps(0.1, 0.02, vec![0.6], vec![now]).unwrap();
        let result = cc.beta_eff_temporal(now);
        let expected = cc.beta_eff();
        assert!((result - expected).abs() < 1e-9, "fresh sample: {result} vs {expected}");
    }

    #[test]
    fn beta_eff_temporal_stale_sample_approaches_beta_base() {
        let now = CG_HALFLIFE_SECS * 100;
        let cc = CoherencyCoefficients::new_with_timestamps(0.1, 0.05, vec![0.8], vec![0]).unwrap();
        let result = cc.beta_eff_temporal(now);
        assert!(
            (result - cc.beta_base).abs() < 0.001,
            "stale sample must approach beta_base={}, got {result}", cc.beta_base
        );
    }

    #[test]
    fn beta_eff_temporal_no_timestamps_falls_back_to_beta_eff() {
        // new() leaves sample_timestamps empty → beta_eff_temporal falls back to beta_eff()
        let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.6, 0.7]).unwrap();
        let result = cc.beta_eff_temporal(1_000_000);
        assert!((result - cc.beta_eff()).abs() < 1e-9);
    }

    #[test]
    fn beta_eff_temporal_empty_struct_timestamps_falls_back() {
        // new() without timestamps — same as above, explicit check for single-sample case
        let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.6]).unwrap();
        let result = cc.beta_eff_temporal(1_000_000);
        assert!((result - cc.beta_eff()).abs() < 1e-9);
    }

    #[test]
    fn beta_eff_temporal_recent_low_cg_dominates_old_high_cg() {
        let now = CG_HALFLIFE_SECS * 10;
        let cc = CoherencyCoefficients::new_with_timestamps(
            0.1, 0.05, vec![0.9, 0.2], vec![0u64, now],
        ).unwrap();
        let result = cc.beta_eff_temporal(now);
        let fresh_only_beta = cc.beta_base * (1.0 - 0.2);
        assert!(
            (result - fresh_only_beta).abs() < 0.005,
            "recent low-CG sample must dominate: expected ≈{fresh_only_beta:.4}, got {result:.4}"
        );
    }
}
