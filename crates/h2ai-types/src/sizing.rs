use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors produced when physics domain values are constructed with out-of-range inputs.
#[derive(Debug, Error)]
pub enum PhysicsError {
    /// α must be in [0, 1); value at or above 1.0 makes `N_max` undefined.
    #[error("alpha must be in [0, 1), got {0}")]
    InvalidAlpha(f64),
    /// Role error cost `c_i` must be in [0, 1]; values outside this range have no physical meaning.
    #[error("c_i must be in [0, 1], got {0}")]
    InvalidErrorCost(f64),
    /// `J_eff` (context fill fraction) must be in [0, 1].
    #[error("J_eff must be in [0, 1], got {0}")]
    InvalidJeff(f64),
    /// τ (creativity temperature) must be in [0, 1].
    #[error("tau must be in [0, 1], got {0}")]
    InvalidTau(f64),
    /// At least one CG sample is required to compute `β_eff` and `N_max`.
    #[error("cg_samples must not be empty")]
    EmptyCgSamples,
    /// β₀ (base coherency cost) must be non-negative; negative costs are unphysical.
    #[error("beta_base must be ≥ 0, got {0}")]
    InvalidBetaBase(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TauValue(f64);

impl TauValue {
    /// Construct a `TauValue`; returns `InvalidTau` if `v` is outside `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns `PhysicsError::InvalidTau` if `v` is not in `[0, 1]`.
    pub fn new(v: f64) -> Result<Self, PhysicsError> {
        if (0.0..=1.0).contains(&v) {
            Ok(Self(v))
        } else {
            Err(PhysicsError::InvalidTau(v))
        }
    }

    #[must_use]
    pub const fn value(&self) -> f64 {
        self.0
    }
}

/// Calibrated coherency parameters for a set of compute adapters.
///
/// `alpha` is the contention (serial-fraction) coefficient from USL calibration.
/// `beta_base` (β₀) is the base coherence-drag coefficient measured from calibration timing.
///
/// Coherence drag has two physical components in LLM ensembles:
///   1. **Conflict reconciliation** — merge step must resolve every contradictory agent-pair; O(N²).
///   2. **Context-attention degradation** — synthesis LLM's retrieval quality degrades for proposals
///      buried in a long context ("Lost in the Middle", Liu et al. 2023); super-linear in N.
///
/// `beta_eff` = β₀ × (1 − `CG_mean`) reduces the *conflict* component via Common Ground.
/// `n_max_context_aware()` further reduces the *positional* component via context-fill pressure.
/// Higher `CG_mean` → lower `β_eff` → higher `N_max`. Bounded at β₀ when `CG_mean` = 0.
/// `n_max` = round(√((1−α)/`β_eff`)) is derived from USL Proposition 1 by setting `dX/dN` = 0
/// in X(N) = N / (1 + α(N−1) + β·N(N−1)).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherencyCoefficients {
    pub alpha: f64,
    /// Base coherency cost per agent pair (β₀), measured from calibration timing.
    /// `β_eff` = β₀ × (1 − `CG_mean`); bounded at β₀ when `CG_mean` = 0.
    #[serde(alias = "kappa_base")]
    pub beta_base: f64,
    /// Conflict-rate-based β, derived from pairwise constraint disagreement.
    /// When present, replaces the `beta_base × (1 − CG_mean)` proxy in `beta_eff()`.
    /// `None` until conflict accumulator data is available for this tenant.
    #[serde(default)]
    pub beta_quality: Option<f64>,
    pub cg_samples: Vec<f64>,
    /// Unix timestamps (seconds) for each entry in `cg_samples`.
    ///
    /// When present (same length as `cg_samples`), `beta_eff_temporal` applies
    /// Ebbinghaus decay so stale CG measurements contribute less weight, causing
    /// `β_eff` to drift toward the conservative ceiling (β₀) without re-calibrating.
    /// Empty when constructed via `new()` — `beta_eff_temporal` then falls back to
    /// the unweighted `beta_eff()`.
    #[serde(default)]
    pub sample_timestamps: Vec<u64>,
}

/// Time constant τ for CG sample decay under exponential temporal weighting.
/// 7 days: a sample one week old contributes at e^−1 ≈ 37% weight (`exp(-t/τ)` at t=τ).
pub const CG_HALFLIFE_SECS: u64 = 604_800;

/// Maximum iterations for the context-aware `N_max` fixed-point solver.
///
/// Empirically converges in ≤3 steps; 5 provides a safe margin.
pub const USL_SOLVER_MAX_ITERS: usize = 5;

/// Convergence tolerance (in agents) for the context-aware `N_max` solver.
///
/// Loop exits when successive iterates differ by less than half an agent.
pub const USL_SOLVER_CONVERGENCE_TOL: f64 = 0.5;

/// Hard cap on recommended ensemble size across all calibration methods.
///
/// Derived from the information-theoretic `n_it_optimal` formula: at typical
/// LLM error-correlation values the marginal gain of a 10th adapter is negligible.
pub const N_MAX_ENSEMBLE_CAP: usize = 9;

/// Upper clamp on pairwise error correlation ρ to prevent Condorcet degeneracy.
/// At ρ = 1.0 the Condorcet gain denominator collapses; 0.99 keeps it numerically stable.
pub const RHO_UPPER_CLAMP: f64 = 0.99;

/// Signed exponent coefficient for τ (creativity temperature) alignment.
///
/// `tau_alignment(a,b) = exp(TAU_ALIGNMENT_DECAY_COEFF × |a−b|)`.
/// Negative value: at |a−b| = 1.0 gives exp(−3) ≈ 0.05 (5% residual alignment).
pub const TAU_ALIGNMENT_DECAY_COEFF: f64 = -3.0;

impl CoherencyCoefficients {
    /// Construct calibrated coherency coefficients without temporal decay information.
    ///
    /// # Errors
    ///
    /// Returns `PhysicsError::InvalidAlpha` if `alpha` is outside `[0, 1)`,
    /// `PhysicsError::InvalidBetaBase` if `beta_base` is negative, or
    /// `PhysicsError::EmptyCgSamples` if `cg_samples` is empty.
    pub fn new(alpha: f64, beta_base: f64, cg_samples: Vec<f64>) -> Result<Self, PhysicsError> {
        Self::new_with_timestamps(alpha, beta_base, cg_samples, vec![])
    }

    /// Construct with per-sample Unix timestamps (seconds) for temporal decay.
    ///
    /// `sample_timestamps` must be the same length as `cg_samples`; pass an
    /// empty vec to skip temporal weighting (equivalent to `new`).
    ///
    /// # Errors
    ///
    /// Returns `PhysicsError::InvalidAlpha` if `alpha` is outside `[0, 1)`,
    /// `PhysicsError::InvalidBetaBase` if `beta_base` is negative, or
    /// `PhysicsError::EmptyCgSamples` if `cg_samples` is empty.
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
            beta_quality: None,
            cg_samples,
            sample_timestamps,
        })
    }

    /// Effective coordination cost per agent pair.
    ///
    /// When `beta_quality` is `Some`, returns it directly (no CG adjustment).
    /// Otherwise, uses the proxy: `β_eff = β₀ × (1 − CG_mean)`.
    ///
    /// - At `CG_mean` = 0 (no overlap): `β_eff` = β₀ (maximum cost, bounded).
    /// - At `CG_mean` = 1 (full overlap): `β_eff` ≈ 0 (coordination-free).
    /// - Previous formula β₀/`CG_mean` diverged at CG→0; this form is bounded everywhere.
    #[must_use]
    pub fn beta_eff(&self) -> f64 {
        self.beta_quality.map_or_else(
            || {
                let cg = self.cg_mean().clamp(0.0, 1.0);
                (self.beta_base * (1.0 - cg)).max(1e-6)
            },
            |bq| bq.max(1e-6),
        )
    }

    /// Maximum useful ensemble size from USL Proposition 1: round(√((1−α)/`β_eff`)).
    ///
    /// Derived by setting `dX/dN` = 0 in X(N) = N / (1 + α(N−1) + β·N(N−1)).
    /// Acts as a safety ceiling; `EnsembleCalibration::n_optimal` (Condorcet-based)
    /// is the primary ensemble-size target.
    #[must_use]
    pub fn n_max(&self) -> f64 {
        let beta_eff = self.beta_eff().max(f64::EPSILON);
        ((1.0 - self.alpha).max(0.0) / beta_eff).sqrt().round()
    }

    /// One-σ confidence interval for `N_max`, derived by propagating CG uncertainty.
    ///
    /// Evaluates `n_max()` at `CG_mean ± cg_std_dev()`. The interval widens with
    /// more CG sample variance. Returns `(n_max(), n_max())` when only one sample
    /// exists (`std_dev` = 0).
    ///
    /// `n_max_lo` is the pessimistic bound (high CG → low `β_eff` → high `N_max` reversal);
    /// `n_max_hi` is the optimistic bound. The pair bounds the true `N_max` under measurement
    /// uncertainty.
    ///
    /// Both bounds are floored at **3.0** — the minimum quorum required for BFT/Krum/SRANI.
    /// When the unclamped value falls below 3, call `n_max_degraded()` to detect the
    /// degradation and trigger a circuit-breaker rather than silently using a 1–2 agent pool.
    #[must_use]
    pub fn n_max_ci(&self) -> (f64, f64) {
        let sigma = self.cg_std_dev();
        let cg_mean = self.cg_mean();
        let n_at_cg = |cg: f64| -> f64 {
            let b = (self.beta_base * (1.0 - cg.clamp(0.0, 1.0)))
                .max(1e-6)
                .max(f64::EPSILON);
            ((1.0 - self.alpha).max(0.0) / b).sqrt().round().max(1.0)
        };
        let n_hi = n_at_cg(cg_mean + sigma);
        let n_lo = n_at_cg(cg_mean - sigma);
        // Hard floor: BFT/Krum/SRANI require N ≥ 3. Unclamped values < 3 indicate
        // adapter degradation; callers should check n_max_degraded() and trip the
        // circuit breaker rather than proceeding with a sub-quorum pool.
        let lo = n_lo.min(n_hi).max(3.0);
        let hi = n_lo.max(n_hi).max(lo);
        (lo, hi)
    }

    /// Returns `true` when the **unclamped** `N_max` point estimate falls below 3.0,
    /// indicating the adapter is too degraded to maintain BFT/Krum/SRANI quorum.
    ///
    /// When this is `true`, callers in non-shadow-mode should fail fast with
    /// `MultiplicationConditionFailure::QuorumDegradedBelowMinimum` rather than
    /// proceeding with a clamped-but-meaningless pool of 1–2 agents.
    #[must_use]
    pub fn n_max_degraded(&self) -> bool {
        self.n_max() < 3.0
    }

    /// `N_max` adjusted for context-window pressure (attention-degradation model).
    ///
    /// Models the "Lost in the Middle" phenomenon (Liu et al. 2023): as N proposals
    /// fill the synthesis context, the synthesizer's retrieval quality degrades for
    /// middle-positioned proposals. This is the *positional* component of coherence drag,
    /// orthogonal to the *conflict* component reduced by CG.
    ///
    /// Fill fraction `f(N) = min(1, N × proposal_tokens / max_tokens)`.
    /// Positional drag amplifies β: `β_ctx(N) = β_eff × (1 + γ × f(N))`.
    /// `γ` (gamma) is the attention-sensitivity coefficient; larger γ = steeper degradation.
    ///
    /// Solves `N = √((1−α) / β_ctx(N))` iteratively (converges in ≤ 5 steps).
    ///
    /// Returns `n_max()` when `proposal_tokens` or `max_tokens` is < 1.0.
    /// Result is always ≥ 1.0.
    #[must_use]
    pub fn n_max_context_aware(&self, proposal_tokens: f64, max_tokens: f64, gamma: f64) -> f64 {
        if proposal_tokens < 1.0 || max_tokens < 1.0 {
            return self.n_max();
        }
        let beta_eff = self.beta_eff().max(f64::EPSILON);
        let alpha = self.alpha;
        let mut n = self.n_max();
        for _ in 0..USL_SOLVER_MAX_ITERS {
            let fill = (n * proposal_tokens / max_tokens).min(1.0);
            let beta_ctx = beta_eff * gamma.mul_add(fill, 1.0);
            let n_new = ((1.0 - alpha).max(0.0) / beta_ctx.max(f64::EPSILON))
                .sqrt()
                .round();
            if (n_new - n).abs() < USL_SOLVER_CONVERGENCE_TOL {
                break;
            }
            n = n_new;
        }
        n.max(1.0)
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cg_mean(&self) -> f64 {
        self.cg_samples.iter().sum::<f64>() / self.cg_samples.len() as f64
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cg_std_dev(&self) -> f64 {
        let n = self.cg_samples.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.cg_mean();
        let variance = self
            .cg_samples
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        variance.sqrt()
    }

    /// Effective coordination cost with Ebbinghaus temporal decay.
    ///
    /// Each CG sample is weighted by `e^(-(now_secs − t) / CG_HALFLIFE_SECS)` using
    /// `self.sample_timestamps`. As samples age their contribution fades, causing `β_eff`
    /// to drift toward the conservative ceiling (β₀) without explicit re-calibration.
    ///
    /// Falls back to `beta_eff()` (unweighted) when `sample_timestamps` is empty or
    /// its length does not match `cg_samples` — no timing information is available.
    ///
    /// Timestamps after `now_secs` are treated as age zero (weight 1.0) via saturating
    /// subtraction — a future timestamp is never penalised.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn beta_eff_temporal(&self, now_secs: u64) -> f64 {
        let ts = &self.sample_timestamps;
        if ts.len() != self.cg_samples.len() || ts.is_empty() {
            return self.beta_eff();
        }
        let halflife = CG_HALFLIFE_SECS as f64;
        let (weighted_cg, total_weight) =
            self.cg_samples
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

/// Minimum `CG_mean` threshold below which ensemble coherency is insufficient to proceed.
///
/// Derived from calibration as `cg_mean − cg_std_dev`, clamped to `[0, max]`.
/// Ensures the multiplication condition gate rejects topologies whose coordination
/// quality falls below the calibrated floor, preventing low-signal ensemble runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoordinationThreshold(f64);

impl CoordinationThreshold {
    #[must_use]
    pub fn from_calibration(cc: &CoherencyCoefficients, max: f64) -> Self {
        let spread = cc.cg_mean() - cc.cg_std_dev();
        Self(spread.clamp(0.0, max))
    }

    /// Raw threshold value in `[0, 1]`.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.0
    }
}

/// Per-role error cost `c_i` ∈ [0, 1] driving merge-strategy selection.
///
/// Higher `c_i` indicates a less trustworthy explorer role; the merge engine escalates
/// from `ScoreOrdered` → `ConsensusMedian` → `OutlierResistant` as the maximum `c_i`
/// across the ensemble crosses the configured `bft_threshold` and `krum_threshold` values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleErrorCost(f64);

impl RoleErrorCost {
    /// Construct a `RoleErrorCost`; returns `InvalidErrorCost` if `value` is outside `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns `PhysicsError::InvalidErrorCost` if `value` is not in `[0, 1]`.
    pub fn new(value: f64) -> Result<Self, PhysicsError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(PhysicsError::InvalidErrorCost(value));
        }
        Ok(Self(value))
    }

    /// Raw error cost in `[0, 1]`.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Low–medium error cost (max `c_i` ≤ `bft_threshold`): pick highest-scored surviving proposal.
    ScoreOrdered,
    /// Medium-high error cost (`bft_threshold` < max `c_i` ≤ `krum_threshold`): Condorcet selection.
    /// Picks the proposal with highest mean Jaccard similarity to the rest of the ensemble.
    /// NOTE: not Byzantine-resistant. Vulnerable to coordinated Byzantine proposals at f ≥ n/2.
    ConsensusMedian,
    /// High error cost (max `c_i` > `krum_threshold`) with explicit f > 0: outlier-resistant
    /// single-selection. Selects the proposal with smallest sum of distances to its
    /// n-f-2 nearest neighbours in Jaccard-distance space. Requires n ≥ 2f+3.
    OutlierResistant { f: usize },
    /// Multi-step outlier-resistant selection: iteratively select m survivors via
    /// `OutlierResistant` scoring, then take the highest verification-scored one. Requires n ≥ 2f+3.
    MultiOutlierResistant { f: usize, m: usize },
}

impl MergeStrategy {
    /// Select merge strategy based on role error costs.
    ///
    /// Three-tier selection:
    /// 1. `krum_f > 0` AND `max_ci > krum_threshold` → `OutlierResistant { f: krum_f }`
    /// 2. `max_ci > bft_threshold` → `ConsensusMedian`
    /// 3. Otherwise → `ScoreOrdered`
    #[must_use]
    pub fn from_role_costs(
        costs: &[RoleErrorCost],
        bft_threshold: f64,
        krum_threshold: f64,
        krum_f: usize,
    ) -> Self {
        let max_ci = costs
            .iter()
            .map(RoleErrorCost::value)
            .fold(f64::NEG_INFINITY, f64::max);
        if krum_f > 0 && max_ci > krum_threshold {
            Self::OutlierResistant { f: krum_f }
        } else if max_ci > bft_threshold {
            Self::ConsensusMedian
        } else {
            Self::ScoreOrdered
        }
    }

    /// Minimum number of proposals needed for `OutlierResistant`/`MultiOutlierResistant` with fault bound f.
    /// Derived from n ≥ 2f + 3 (Blanchard et al. 2017, Theorem 2).
    #[must_use]
    pub const fn min_krum_quorum(f: usize) -> usize {
        2 * f + 3
    }
}

/// Configuration for the Optimal Synthesis Policy (OSP).
///
/// All fields have calibrated production defaults. Pass `OspConfig::default()` unless
/// you have empirically measured verifier noise σ for a specific deployment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OspConfig {
    /// Verifier noise temperature `T_v` = σ²/2. Default 0.125 (σ ≈ 0.5).
    /// Regime boundary: Δ ≥ 2·`T_v` → `ClearLeader` (P(correct) ≥ 0.92).
    pub t_v: f64,
    /// False-positive rate α for adaptive concordance threshold τ(`N_f`).
    /// τ(`N_f`) = clamp(0.5 + √(−ln(α) / (2·`N_f`)), 0.5, 1.0). Default 0.1.
    pub concordance_alpha: f64,
    /// Maximum `N_v` for Zone 3 injection. Omit Zone 3 when `N_v` > this value.
    /// Gravity-well amplification scales linearly with `N_v`. Default 4.
    pub max_n_v_for_zone3: usize,
    /// Leaky accumulation decay λ. Default 0.7 (half-life ≈ 2 retries).
    pub accumulation_decay: f64,
}

impl Default for OspConfig {
    fn default() -> Self {
        Self {
            t_v: 0.125,
            concordance_alpha: 0.1,
            max_n_v_for_zone3: 4,
            accumulation_decay: 0.7,
        }
    }
}

/// Effective context-fill fraction `J_eff` ∈ [0, 1] for a compiled task context.
///
/// `J_eff` measures how much of the constraint corpus was successfully embedded into the
/// system context after compaction. Values below the configured threshold cause
/// `ContextUnderflow` — the context lacks enough relevant signal to run the ensemble.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JeffectiveGap(f64);

impl JeffectiveGap {
    /// Construct a `JeffectiveGap`; returns `InvalidJeff` if `value` is outside `[0, 1]`.
    ///
    /// # Errors
    /// Returns `PhysicsError::InvalidJeff` when `value` is not in `[0.0, 1.0]`.
    pub fn new(value: f64) -> Result<Self, PhysicsError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(PhysicsError::InvalidJeff(value));
        }
        Ok(Self(value))
    }

    /// Raw `J_eff` value in `[0, 1]`.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Returns `true` when `J_eff` is strictly below `threshold`, indicating underflow.
    #[must_use]
    pub fn is_below_threshold(&self, threshold: f64) -> bool {
        self.0 < threshold
    }
}

/// Gate that enforces the three conditions required for ensemble quality multiplication.
///
/// All three must hold for the planner to proceed: sufficient baseline competence (p > `min_competence`),
/// sufficient decorrelation (ρ < `max_correlation`), and sufficient common ground (`CG_mean` ≥ `θ_coord`).
pub struct MultiplicationCondition;

/// Exponential decay alignment between two τ (creativity temperature) values.
///
/// `tau_alignment(a, b) = exp(TAU_ALIGNMENT_DECAY_COEFF × |a − b|)`.
/// Same τ → 1.0. Difference of 1.0 → exp(−3) ≈ 0.05.
#[must_use]
pub fn tau_alignment(a: TauValue, b: TauValue) -> f64 {
    (TAU_ALIGNMENT_DECAY_COEFF * (a.value() - b.value()).abs()).exp()
}

/// Condorcet Jury Theorem ensemble accuracy with error correlation.
///
/// Returns the probability that a majority vote among `n_agents` agents, each correct
/// with probability `p`, is correct — adjusted for pairwise error correlation `rho`.
///
/// Q(N,p,ρ) = p + (`Q_independent(N,p)` − p) × (1 − ρ).
/// Boundary: N=1 → Q=p; ρ=1 → Q=p.
#[must_use]
#[allow(clippy::cast_precision_loss)]
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
            let log_term = ((n - k) as f64).mul_add(
                (1.0 - p).ln(),
                (k as f64).mul_add(p.ln(), log_binom_coeff(n, k)),
            );
            sum += log_term.exp();
        }
        // For even N, exact tie → 0.5 probability of being correct
        if n.is_multiple_of(2) {
            let k = n / 2;
            let log_term = (k as f64).mul_add(
                (1.0 - p).ln(),
                (k as f64).mul_add(p.ln(), log_binom_coeff(n, k)),
            );
            sum += 0.5 * log_term.exp();
        }
        sum.clamp(0.0, 1.0)
    };
    (q_ind - p).mul_add(1.0 - rho, p).clamp(0.0, 1.0)
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
#[allow(clippy::cast_precision_loss)]
fn log_gamma(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    (1..n).map(|i| (i as f64).ln()).sum()
}

/// Whether quality predictions use empirical measurement or the `CG_mean` proxy.
///
/// `Heuristic`: p and ρ are proxied from `CG_mean` — not empirically validated.
/// `Empirical`: p is from `baseline_accuracy_proxy` (measured on a held-out set).
/// The proxy underestimates `ρ_actual` by 0.2–0.3 on factual tasks (arxiv 2511.12309).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PredictionBasis {
    #[default]
    Heuristic,
    Empirical,
}

/// Condorcet-based calibration result for an ensemble of compute adapters.
///
/// Produced alongside `CoherencyCoefficients` by `CalibrationHarness`.
/// Provides the theoretically optimal ensemble size and the expected quality
/// gain at that size, derived from the Condorcet Jury Theorem.
///
/// `topology_gain` is derived from the CG embedding proxy for ρ. Empirical
/// validation via `compare.py` (benchmark tool — see `docs/architecture/reference.md`) is recommended for production
/// quality claims. See `prediction_basis` for the source of p and ρ estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleCalibration {
    /// Mean per-adapter estimated accuracy proxy: `0.5 + CG_mean / 2`.
    pub p_mean: f64,
    /// Mean pairwise error correlation proxy: `1.0 − CG_mean`.
    pub rho_mean: f64,
    /// Ensemble size that maximises Q(N,p,ρ)/(N+1), capped at 9.
    pub n_optimal: usize,
    /// Expected ensemble quality `Q(n_optimal`, `p_mean`, `rho_mean`).
    pub q_optimal: f64,
    /// Whether quality predictions are CG-proxy-based (Heuristic) or from
    /// measured baseline accuracy (Empirical).
    #[serde(default)]
    pub prediction_basis: PredictionBasis,
}

impl EnsembleCalibration {
    /// Derive calibration from `CG_mean` using proxy formulas:
    ///   `p_mean`   = 0.5 + `CG_mean` / 2   (accuracy proxy)
    ///   `rho_mean` = 1.0 − `CG_mean`        (correlation proxy)
    ///
    /// `N_optimal` = `argmax_N` (Q(N,p,ρ) − p) / N for N in `1..=max_n`.
    /// This is marginal Condorcet gain per agent above the single-agent baseline.
    /// N=1 scores 0; when ρ=1 (no ensemble benefit) all scores are 0 and N=1 is returned.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
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
        Self {
            p_mean,
            rho_mean,
            n_optimal,
            q_optimal,
            prediction_basis: PredictionBasis::Heuristic,
        }
    }

    /// Construct with a directly measured accuracy value, overriding the CG-mean proxy.
    /// Use when `baseline_accuracy_proxy` is set in config from `compare.py` (benchmark tool — see `docs/architecture/reference.md`).
    /// Uses the same marginal-gain-per-agent scoring as `from_cg_mean`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
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
        Self {
            p_mean: p,
            rho_mean,
            n_optimal,
            q_optimal,
            prediction_basis: PredictionBasis::Empirical,
        }
    }

    /// Construct with directly measured p and ρ, bypassing all proxy formulas.
    /// Used by the online ρ EMA updater (INNOVATION-3, GAP-A3).
    /// `rho` is clamped to [0.0, 0.99] to prevent degenerate Condorcet computation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_empirical(p_mean: f64, rho_empirical: f64, max_n: usize) -> Self {
        let p = p_mean.clamp(0.5, 1.0);
        let rho = rho_empirical.clamp(0.0, RHO_UPPER_CLAMP);
        let max_n = max_n.max(1);
        let (n_optimal, _) = (1..=max_n)
            .map(|n| {
                let q = condorcet_quality(n, p, rho);
                let score = (q - p) / n as f64;
                (n, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((1, 0.0));
        let q_optimal = condorcet_quality(n_optimal, p, rho);
        Self {
            p_mean: p,
            rho_mean: rho,
            n_optimal,
            q_optimal,
            prediction_basis: PredictionBasis::Empirical,
        }
    }

    /// Expected quality at a given ensemble size.
    #[must_use]
    pub fn quality_at_n(&self, n: usize) -> f64 {
        condorcet_quality(n, self.p_mean, self.rho_mean)
    }

    /// Condorcet gain over single-agent baseline: `Q(n_optimal)` − `p_mean`.
    #[must_use]
    pub fn topology_gain(&self) -> f64 {
        (self.q_optimal - self.p_mean).max(0.0)
    }

    /// Information-theoretic optimal ensemble size from `rho_mean`.
    ///
    /// See [`n_it_optimal`] for the derivation.
    #[must_use]
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
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn n_it_optimal(rho: f64) -> usize {
    if rho <= 1e-10 {
        return 1;
    }
    if rho >= 1.0 - 1e-10 {
        return N_MAX_ENSEMBLE_CAP;
    }
    let n = 1.0 + 0.5_f64.log(1.0 - rho);
    (n.ceil() as usize).clamp(1, N_MAX_ENSEMBLE_CAP)
}

/// Eigenvalue-based ensemble calibration from the pairwise CG similarity matrix.
///
/// Implements the portfolio theory "participation ratio" (Choueifaty & Coignard 2008):
///   `N_eff` = (Σ λᵢ)² / Σ λᵢ²
///
/// At full independence (Σ = I), `N_eff` = N. At full correlation (Σ = 𝟏𝟏ᵀ), `N_eff` = 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EigenCalibration {
    /// Effective number of independent adapters: (Σλ)²/Σλ².
    pub n_effective: f64,
    /// Normalized Shannon entropy of eigenvalue distribution ∈ [0, 1].
    /// 1.0 = fully decorrelated; 0.0 = one adapter dominates.
    pub h_diversity: f64,
    /// Eigenvalues of the CG similarity matrix, sorted descending.
    pub eigenvalues: Vec<f64>,
    /// Recommended adapter count: first N where adding another raises `N_eff` by < 0.05.
    pub n_pruned: usize,
}

impl EigenCalibration {
    /// Compute from an N×N symmetric positive-semidefinite CG similarity matrix.
    ///
    /// `delta`: minimum `N_eff` increment to include the next adapter in `n_pruned`.
    /// Use `H2AIConfig::eigen_n_eff_delta` (default 0.05) at production call sites.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_cg_matrix(sigma: &DMatrix<f64>, delta: f64) -> Self {
        let eig = sigma.clone().symmetric_eigen();
        let mut evs: Vec<f64> = eig
            .eigenvalues
            .iter()
            .copied()
            .map(|v| v.max(0.0))
            .collect();
        evs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = evs.iter().sum();
        let sum_sq: f64 = evs.iter().map(|l| l * l).sum();
        let n_eff = if sum_sq > 1e-12 {
            sum * sum / sum_sq
        } else {
            1.0
        };

        let h_div: f64 = evs
            .iter()
            .filter(|&&l| l > 1e-12)
            .map(|&l| {
                let p = l / sum;
                -p * p.ln()
            })
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
                if i > 0 && current - prev < delta {
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

    /// Compute from a pre-normalised N×N cosine kernel matrix K where trace(K) = 1.
    ///
    /// The raw cosine matrix C has C[i][i] = 1.0, so trace(C) = N. The caller must
    /// normalise: K = C / N so that eigenvalues sum to 1 and `N_eff` ∈ [1, N].
    /// Clamps negative eigenvalues to 0 (numerical noise from `symmetric_eigen`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_cosine_matrix(k: &DMatrix<f64>, delta: f64) -> Self {
        let eig = k.clone().symmetric_eigen();
        let mut evs: Vec<f64> = eig
            .eigenvalues
            .iter()
            .copied()
            .map(|v| v.max(0.0))
            .collect();
        evs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = evs.iter().sum();
        let sum_sq: f64 = evs.iter().map(|l| l * l).sum();
        let n_eff = if sum_sq > 1e-12 {
            sum * sum / sum_sq
        } else {
            1.0
        };

        let h_div: f64 = evs
            .iter()
            .filter(|&&l| l > 1e-12)
            .map(|&l| {
                let p = l / sum.max(1e-15);
                -p * p.ln()
            })
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
                if i > 0 && current - prev < delta {
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

    /// Derive effective correlation from `N_eff`: `ρ_eff` = 1 − `N_eff/N`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
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
    #[error("n_eff_cosine_prior {n_eff:.2} < 1 + diversity_threshold {threshold:.2} — pool cannot produce independent perspectives")]
    InsufficientPoolDiversity { n_eff: f64, threshold: f64 },
    #[error("verifier family '{verifier_family}' matches explorer family '{explorer_family}' — monoculture verification invalidates Condorcet independence")]
    VerifierExplorerFamilyConflict {
        explorer_family: String,
        verifier_family: String,
    },
    #[error("adapter N_max={unclamped_n_max:.1} < 3 — model too degraded to maintain BFT/Krum/SRANI quorum; adapter must be marked Offline")]
    QuorumDegradedBelowMinimum { unclamped_n_max: f64 },
}

impl MultiplicationCondition {
    /// Check the three multiplication conditions; returns the first failure encountered.
    ///
    /// Checks in order: competence, decorrelation, common ground.
    /// Returns `Ok(())` when all three pass and ensemble quality multiplication is expected.
    ///
    /// # Errors
    /// Returns the first `MultiplicationConditionFailure` encountered.
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

/// Routing quadrant for Phase 1.5 task complexity assessment.
///
/// Derived from `TCC_effective` (task dimensionality) and pool `N_eff` (adapter diversity).
/// Drives topology selection: Precision → Self-MoA, Coverage → cross-family committee,
/// Complex → forced `CoT` + synthesis, Degenerate → `MultiplicationConditionFailed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskQuadrant {
    /// Low TCC, normal `N_eff`: single feasible region; within-family τ-spread suffices.
    Precision,
    /// High TCC, normal `N_eff`: diverse solution space; cross-family committee needed.
    Coverage,
    /// High TCC, low `N_eff`: complex space but pool is under-diverse; force `CoT` + synthesis.
    Complex,
    /// Both low: pool cannot explore the space; routes to `MultiplicationConditionFailed`.
    Degenerate,
}

/// Reason the N-probe mini-generation step was skipped in Phase 1.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProbeSkipReason {
    /// Probe ran normally; `TCC_empirical` is present.
    #[default]
    None,
    /// `TCC_structural` clearly above `tcc_coverage_threshold`; probe cannot change routing.
    UnambiguousCoverage,
    /// `TCC_structural` clearly below `tcc_precision_threshold`; probe cannot change routing.
    UnambiguousPrecision,
    /// `static_coverage < min_static_coverage_for_probe`: satisfaction matrix would be
    /// near-empty; heavy amplification applied to `TCC_structural` instead.
    HeavyDominantCorpus,
    /// `CalibrationQuality::Bootstrap`: synthetic priors not suitable for probe comparison.
    BootstrapCalibration,
    /// `TCC_structural` is in the ambiguous band `(tcc_precision_threshold, tcc_coverage_threshold)`
    /// but the N-probe step is deferred pending GAP-A1 experiment threshold validation.
    /// Routes conservatively to Coverage. Remove once Path B probe is enabled in `shadow_mode=false`.
    AmbiguousBandProbeDeferred,
}

/// Structural family of an oracle type — used for FUSE worst-of-family reduction.
///
/// Oracles in the same family share a correlated failure mode (e.g., both JSON Schema
/// and Z3 Symbolic fail on malformed JSON). Grouping by family and taking the min score
/// within each family before averaging across families prevents a single syntactic defect
/// from registering as multiple independent failures in the MMSE calibration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleFamily {
    /// Structural/syntax-level checks (JSON Schema, AST parsing, Z3 pre-parse).
    Syntactic,
    /// Semantic evaluation (free-form LLM judge, multiple-choice, test suites).
    Semantic,
    /// Human reviewer gate — always independent from automated families.
    Human,
}

/// Geographic/semantic domain of an oracle evaluation.
///
/// Used to stratify calibration residuals per-domain when `n_domain` ≥ 15.
/// Falls back to pooled residuals with a 20% width penalty when sparse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleDomain {
    Code,
    Factual,
    Reasoning,
    Human,
    Unknown,
}

impl OracleDomain {
    /// Map this domain to its structural family for FUSE worst-of-family reduction.
    #[must_use]
    pub fn family(&self) -> OracleFamily {
        match self {
            OracleDomain::Code => OracleFamily::Syntactic,
            OracleDomain::Factual | OracleDomain::Reasoning | OracleDomain::Unknown => {
                OracleFamily::Semantic
            }
            OracleDomain::Human => OracleFamily::Human,
        }
    }
}

/// Configuration for Phase 6 async oracle evaluation.
///
/// Carried on `TaskManifest::oracle`. When `None`, Phase 6 is skipped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleSpec {
    /// HTTP URL of the external oracle service.
    /// Example: `http://oracle-service:9090/evaluate`
    pub runner_uri: String,
    /// Milliseconds before the HTTP call is abandoned.
    /// On timeout: `passed=false`, `score=0.0`.
    pub timeout_ms: u64,
    /// Domain tag forwarded to the oracle and stored in calibration observations.
    pub domain: OracleDomain,
}

/// A single oracle evaluation result stored in the rolling calibration window.
///
/// `residual = |q_confidence − y_oracle as f64|` — the nonconformity score for
/// split-conformal calibration on binary outcomes (Angelopoulos & Bates 2023).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleObservation {
    pub task_id: String,
    pub q_confidence: f64,
    pub y_oracle: bool,
    /// `|q_confidence − y_oracle as f64|` — nonconformity score.
    pub residual: f64,
    pub domain: OracleDomain,
    pub timestamp_ms: u64,
}

/// Opaque verdict returned by the external oracle service.
/// Stored for audit logging — not interpreted by the control plane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleVerdict {
    pub details: serde_json::Value,
}
