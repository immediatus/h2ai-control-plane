use crate::diagnostics::CalibrationState;
use h2ai_types::sizing::{condorcet_quality, EigenCalibration, PredictionBasis};
use rand::Rng;

/// Input parameters for computing harness attribution.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributionInput {
    /// Mean per-adapter estimated accuracy (from `EnsembleCalibration.p_mean`,
    /// or proxy `0.5 + CG_mean / 2` when `EnsembleCalibration` unavailable).
    pub p_mean: f64,
    /// Mean pairwise error correlation (from `EnsembleCalibration.rho_mean`,
    /// or proxy `1 - CG_mean`).
    pub rho_mean: f64,
    /// Number of explorer agents in the ensemble.
    pub n_agents: u32,
    /// Fraction of proposals that survived verification (1.0 = nothing filtered).
    pub verification_filter_ratio: f64,
    /// Mean number of TAO loop turns executed across accepted proposals.
    pub tao_turns_mean: f64,
    /// Multiplicative factor applied per additional TAO turn (from `H2AIConfig::tao_per_turn_factor`).
    pub tao_per_turn_factor: f64,
    /// Source of quality predictions: CG-proxy (Heuristic) or measured `baseline_accuracy_proxy` (Empirical).
    pub prediction_basis: PredictionBasis,
    /// Talagrand rank-histogram calibration state from the current task's verification scores.
    /// `Some(UnderDispersed)` means proposals scored too similarly — a Case B (correlated
    /// failure) flag. When firing AND `rho_mean < 0.5` (`CG_mean` > 0.5), ρ is corrected upward.
    pub talagrand_state: Option<CalibrationState>,
    /// Eigenvalue-based diversity metric from calibration. When `n_eff / n_agents < 0.4`,
    /// agents have low effective diversity — a second independent Case B signal.
    pub eigen_calibration: Option<EigenCalibration>,
}

/// Condorcet-grounded decomposition of output confidence into per-component contributions.
///
/// `q_confidence = 1 − (1 − Q(N, p, ρ_adj)) × verification_filter_ratio × tao_multiplier`
/// clamped to `[p_mean, 1.0]`, where `ρ_adj` may be conservatively adjusted upward when
/// Case B (correlated failure) signals fire.
///
/// `q_confidence` measures how confident the system is in its output, not whether the output
/// is correct. Oracle-grounded correctness requires external measurement (see GAP-B3, GAP-E1).
///
/// Ground truth for `ρ_actual` requires `baseline_eval.py` ICC measurement (see S5/S8).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HarnessAttribution {
    /// Single-agent expected accuracy: `p_mean`.
    pub baseline_quality: f64,
    /// Confidence improvement from N-agent ensemble via Condorcet Jury Theorem.
    /// Computed using `rho_adjusted`, not raw `rho_mean`.
    pub topology_gain: f64,
    /// Upper-bound estimate of the confidence contribution from the verification phase.
    /// Computed as `Q_ensemble × (1 − verification_filter_ratio)`. Informational only.
    pub verification_gain: f64,
    /// Upper-bound estimate of the confidence contribution from TAO loop iterations.
    /// Computed as `Q_ensemble × (1 − tao_multiplier)`. Informational only.
    pub tao_gain: f64,
    /// Total output confidence, clamped to `[p_mean, 1.0]`.
    /// This is a self-assessment: it predicts correctness but does not measure it.
    pub q_confidence: f64,
    /// Basis for quality predictions: CG-proxy (Heuristic) or measured baseline accuracy (Empirical).
    pub prediction_basis: PredictionBasis,
    /// Fraction of oracle (Tier 1) tests passed for this task.
    pub q_measured: Option<f64>,
    /// ρ after Talagrand and `N_eff` Case B corrections. Equals `rho_mean` when no
    /// correction was applied. Used in place of raw `rho_mean` for `condorcet_quality`.
    pub rho_adjusted: f64,
    /// True when at least one Case B signal fired (Talagrand `UnderDispersed` or `N_eff/N` < 0.4).
    /// Quality predictions are conservatively adjusted; `N_max` selection is unchanged.
    pub case_b_flag: bool,
    /// Quality improvement from the synthesis phase: Q(synthesis) − `max(Q(individual_proposals))`.
    /// Zero when synthesis was skipped, fell back to selection, or the engine did not run synthesis.
    pub synthesis_gain: f64,
}

impl HarnessAttribution {
    #[must_use]
    pub fn compute(input: &AttributionInput) -> Self {
        let p = input.p_mean.clamp(0.0, 1.0);
        let n = input.n_agents.max(1);

        // ── S7: ρ correction for Case B (correlated failure) ─────────────────
        // Talagrand UnderDispersed (Λ-curve) + rho_proxy < 0.5 (CG_mean > 0.5):
        // apply +0.30×(1−ρ) to conservatively close 59–89% of Q_predicted error.
        // Guard at rho ≥ 0.5 prevents overshoot at low CG where proxy ≈ actual.
        let rho_raw = input.rho_mean.clamp(0.0, 1.0);

        let talagrand_correction = match input.talagrand_state {
            Some(CalibrationState::UnderDispersed) if rho_raw < 0.5 => 0.30 * (1.0 - rho_raw),
            _ => 0.0,
        };

        // N_eff/N < 0.4 → low effective diversity → second Case B signal (max +15pp).
        let neff_correction = match &input.eigen_calibration {
            Some(eigen) if (eigen.n_effective / f64::from(n)) < 0.4 => 0.15 * (1.0 - rho_raw),
            _ => 0.0,
        };

        let case_b_flag = talagrand_correction > 0.0 || neff_correction > 0.0;
        let rho_adjusted = (rho_raw + talagrand_correction + neff_correction).clamp(0.0, 1.0);
        // ─────────────────────────────────────────────────────────────────────

        let baseline_quality = p;
        let q_ensemble = condorcet_quality(n as usize, p, rho_adjusted);
        let topology_gain = (q_ensemble - p).max(0.0);

        let tpf = input.tao_per_turn_factor.clamp(0.0, 1.0);
        let turns = input.tao_turns_mean.max(1.0);
        let tao_multiplier = tpf.powf(turns - 1.0);
        let tao_gain = (q_ensemble * (1.0 - tao_multiplier)).max(0.0);

        let fr = input.verification_filter_ratio.clamp(0.0, 1.0);
        let verification_gain = (q_ensemble * (1.0 - fr)).max(0.0);

        let error_remaining = (1.0 - q_ensemble) * fr * tao_multiplier;
        let total_quality = (1.0 - error_remaining).clamp(baseline_quality, 1.0);

        Self {
            baseline_quality,
            topology_gain,
            verification_gain,
            tao_gain,
            q_confidence: total_quality,
            prediction_basis: input.prediction_basis,
            q_measured: None,
            rho_adjusted,
            case_b_flag,
            synthesis_gain: 0.0,
        }
    }
}

// ── Attribution uncertainty quantification ────────────────────────────────────

/// How a `[q_total_lo, q_total_hi]` interval was derived.
#[derive(Debug, Clone, PartialEq)]
pub enum IntervalBasis {
    /// 90% bootstrap CI from n CG calibration samples (no ground truth needed).
    Bootstrap { n_cg_samples: usize },
    /// Coverage-guaranteed conformal interval from n Tier 1 calibration tasks.
    Conformal { n_calibration: usize, coverage: f64 },
    /// Fewer than 2 CG samples — no meaningful interval available.
    None,
}

/// Output confidence with a 90% uncertainty interval.
#[derive(Debug, Clone)]
pub struct AttributionInterval {
    /// Point estimate (same as `HarnessAttribution::q_confidence`).
    pub q_confidence: f64,
    /// 5th percentile of the bootstrap or conformal interval.
    pub q_confidence_lo: f64,
    /// 95th percentile of the bootstrap or conformal interval.
    pub q_confidence_hi: f64,
    /// Source of the interval.
    pub interval_basis: IntervalBasis,
}

/// Compute a 90% bootstrap confidence interval for `q_confidence` from CG sample variance.
///
/// Draws `n_bootstrap` samples with replacement from `cg_samples`, recomputes `q_confidence`
/// for each, and returns the (p5, p95) percentiles as `[q_confidence_lo, q_confidence_hi]`.
///
/// Returns `IntervalBasis::None` when `cg_samples.len() < 2`.
#[must_use]
pub fn bootstrap_interval(
    base_input: &AttributionInput,
    cg_samples: &[f64],
    n_bootstrap: usize,
) -> AttributionInterval {
    let q_confidence = HarnessAttribution::compute(base_input).q_confidence;

    if cg_samples.len() < 2 {
        return AttributionInterval {
            q_confidence,
            q_confidence_lo: q_confidence,
            q_confidence_hi: q_confidence,
            interval_basis: IntervalBasis::None,
        };
    }

    let n = cg_samples.len();
    let mut rng = rand::thread_rng();
    let mut boot_totals: Vec<f64> = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Draw n samples with replacement
        let cg_mean_boot: f64 =
            (0..n).map(|_| cg_samples[rng.gen_range(0..n)]).sum::<f64>() / n as f64;
        let cg_boot = cg_mean_boot.clamp(0.0, 1.0);

        // Recompute p/rho from bootstrap CG (same proxy as the heuristic path)
        let p_boot = match base_input.prediction_basis {
            PredictionBasis::Empirical => base_input.p_mean,
            PredictionBasis::Heuristic => (0.5 + cg_boot / 2.0).clamp(0.0, 1.0),
        };
        let rho_boot = match base_input.prediction_basis {
            PredictionBasis::Empirical => base_input.rho_mean,
            PredictionBasis::Heuristic => (1.0 - cg_boot).clamp(0.0, 1.0),
        };

        let boot_input = AttributionInput {
            p_mean: p_boot,
            rho_mean: rho_boot,
            ..base_input.clone()
        };
        boot_totals.push(HarnessAttribution::compute(&boot_input).q_confidence);
    }

    boot_totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p5_idx = ((n_bootstrap as f64 * 0.05) as usize).min(n_bootstrap - 1);
    let p95_idx = ((n_bootstrap as f64 * 0.95) as usize).min(n_bootstrap - 1);

    AttributionInterval {
        q_confidence,
        q_confidence_lo: boot_totals[p5_idx],
        q_confidence_hi: boot_totals[p95_idx],
        interval_basis: IntervalBasis::Bootstrap { n_cg_samples: n },
    }
}

/// Compute a coverage-guaranteed conformal prediction interval.
///
/// Uses split-conformal calibration: the threshold `q_hat` is the
/// `⌈(n+1)(1−α)⌉ / n` quantile of the sorted calibration residuals.
///
/// Returns `(q_predicted − q_hat, q_predicted + q_hat)` clamped to `[0, 1]`.
/// Returns `(0.0, 1.0)` when `calibration_residuals` is empty.
#[must_use]
pub fn conformal_interval(
    q_predicted: f64,
    calibration_residuals: &[f64],
    coverage: f64,
) -> (f64, f64) {
    if calibration_residuals.is_empty() {
        return (0.0_f64.max(q_predicted - 1.0), (q_predicted + 1.0).min(1.0));
    }
    let n = calibration_residuals.len();
    let idx = (((n + 1) as f64 * coverage).ceil() as usize).min(n);
    let mut sorted = calibration_residuals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q_hat = sorted.get(idx.saturating_sub(1)).copied().unwrap_or(1.0);
    (
        (q_predicted - q_hat).clamp(0.0, 1.0),
        (q_predicted + q_hat).clamp(0.0, 1.0),
    )
}
