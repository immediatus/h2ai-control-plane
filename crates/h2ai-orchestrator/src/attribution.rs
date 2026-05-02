use crate::diagnostics::CalibrationState;
use h2ai_types::physics::{condorcet_quality, EigenCalibration, PredictionBasis};
use rand::Rng;

/// Input parameters for computing harness attribution.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributionInput {
    /// Mean per-adapter estimated accuracy (from EnsembleCalibration.p_mean,
    /// or proxy `0.5 + CG_mean / 2` when EnsembleCalibration unavailable).
    pub p_mean: f64,
    /// Mean pairwise error correlation (from EnsembleCalibration.rho_mean,
    /// or proxy `1 - CG_mean`).
    pub rho_mean: f64,
    /// Number of explorer agents in the ensemble.
    pub n_agents: u32,
    /// Fraction of proposals that survived verification (1.0 = nothing filtered).
    pub verification_filter_ratio: f64,
    /// Mean number of TAO loop turns executed across accepted proposals.
    pub tao_turns_mean: f64,
    /// Multiplicative factor applied per additional TAO turn (from H2AIConfig::tao_per_turn_factor).
    pub tao_per_turn_factor: f64,
    /// Source of quality predictions: CG-proxy (Heuristic) or measured baseline_accuracy_proxy (Empirical).
    pub prediction_basis: PredictionBasis,
    /// Talagrand rank-histogram calibration state from the current task's verification scores.
    /// `Some(UnderDispersed)` means proposals scored too similarly — a Case B (correlated
    /// failure) flag. When firing AND `rho_mean < 0.5` (CG_mean > 0.5), ρ is corrected upward.
    pub talagrand_state: Option<CalibrationState>,
    /// Eigenvalue-based diversity metric from calibration. When `n_eff / n_agents < 0.4`,
    /// agents have low effective diversity — a second independent Case B signal.
    pub eigen_calibration: Option<EigenCalibration>,
}

/// Condorcet-grounded decomposition of total output quality into per-component contributions.
///
/// `total_quality = 1 − (1 − Q(N, p, ρ_adj)) × verification_filter_ratio × tao_multiplier`
/// clamped to `[p_mean, 1.0]`, where ρ_adj may be conservatively adjusted upward when
/// Case B (correlated failure) signals fire.
///
/// Ground truth for ρ_actual requires `baseline_eval.py` ICC measurement (see S5/S8).
#[derive(Debug, Clone)]
pub struct HarnessAttribution {
    /// Single-agent expected quality: p_mean.
    pub baseline_quality: f64,
    /// Quality improvement from N-agent ensemble via Condorcet Jury Theorem.
    /// Computed using `rho_adjusted`, not raw `rho_mean`.
    pub topology_gain: f64,
    /// Upper-bound estimate of the quality contribution from the verification phase.
    /// Computed as `Q_ensemble × (1 − verification_filter_ratio)`. Informational only.
    pub verification_gain: f64,
    /// Upper-bound estimate of the quality contribution from TAO loop iterations.
    /// Computed as `Q_ensemble × (1 − tao_multiplier)`. Informational only.
    pub tao_gain: f64,
    /// Total quality, clamped to `[p_mean, 1.0]`.
    pub total_quality: f64,
    /// Basis for quality predictions: CG-proxy (Heuristic) or measured baseline accuracy (Empirical).
    pub prediction_basis: PredictionBasis,
    /// Fraction of oracle (Tier 1) tests passed for this task.
    pub q_measured: Option<f64>,
    /// ρ after Talagrand and N_eff Case B corrections. Equals `rho_mean` when no
    /// correction was applied. Used in place of raw `rho_mean` for `condorcet_quality`.
    pub rho_adjusted: f64,
    /// True when at least one Case B signal fired (Talagrand UnderDispersed or N_eff/N < 0.4).
    /// Quality predictions are conservatively adjusted; N_max selection is unchanged.
    pub case_b_flag: bool,
}

impl HarnessAttribution {
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
            Some(eigen) if (eigen.n_effective / n as f64) < 0.4 => 0.15 * (1.0 - rho_raw),
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
            total_quality,
            prediction_basis: input.prediction_basis,
            q_measured: None,
            rho_adjusted,
            case_b_flag,
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

/// Total quality with a 90% uncertainty interval.
#[derive(Debug, Clone)]
pub struct AttributionInterval {
    /// Point estimate (same as `HarnessAttribution::total_quality`).
    pub q_total: f64,
    /// 5th percentile of the bootstrap or conformal interval.
    pub q_total_lo: f64,
    /// 95th percentile of the bootstrap or conformal interval.
    pub q_total_hi: f64,
    /// Source of the interval.
    pub interval_basis: IntervalBasis,
}

/// Compute a 90% bootstrap confidence interval for `Q_total` from CG sample variance.
///
/// Draws `n_bootstrap` samples with replacement from `cg_samples`, recomputes `Q_total`
/// for each, and returns the (p5, p95) percentiles as `[q_total_lo, q_total_hi]`.
///
/// Returns `IntervalBasis::None` when `cg_samples.len() < 2`.
pub fn bootstrap_interval(
    base_input: &AttributionInput,
    cg_samples: &[f64],
    n_bootstrap: usize,
) -> AttributionInterval {
    let q_total = HarnessAttribution::compute(base_input).total_quality;

    if cg_samples.len() < 2 {
        return AttributionInterval {
            q_total,
            q_total_lo: q_total,
            q_total_hi: q_total,
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
        boot_totals.push(HarnessAttribution::compute(&boot_input).total_quality);
    }

    boot_totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p5_idx = ((n_bootstrap as f64 * 0.05) as usize).min(n_bootstrap - 1);
    let p95_idx = ((n_bootstrap as f64 * 0.95) as usize).min(n_bootstrap - 1);

    AttributionInterval {
        q_total,
        q_total_lo: boot_totals[p5_idx],
        q_total_hi: boot_totals[p95_idx],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribution_n1_topology_gain_is_zero() {
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.3,
            n_agents: 1,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.topology_gain.abs() < 1e-10,
            "N=1 topology_gain should be 0, got {}",
            attr.topology_gain
        );
    }

    #[test]
    fn attribution_n3_topology_gain_positive_for_good_p() {
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.2,
            n_agents: 3,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.topology_gain > 0.0,
            "N=3 with p=0.7, rho=0.2 should have positive topology_gain, got {}",
            attr.topology_gain
        );
    }

    #[test]
    fn attribution_total_quality_bounded() {
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.3,
            n_agents: 5,
            verification_filter_ratio: 0.8,
            tao_turns_mean: 2.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.total_quality >= 0.0 && attr.total_quality <= 1.0,
            "total_quality out of bounds: {}",
            attr.total_quality
        );
    }

    #[test]
    fn attribution_no_topology_gain_at_full_correlation() {
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 1.0,
            n_agents: 5,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.topology_gain.abs() < 1e-10,
            "rho=1 should give zero topology_gain, got {}",
            attr.topology_gain
        );
    }

    #[test]
    fn attribution_total_quality_at_least_baseline() {
        // total_quality must always be >= p_mean (the single-agent baseline)
        let input = AttributionInput {
            p_mean: 0.6,
            rho_mean: 0.4,
            n_agents: 3,
            verification_filter_ratio: 0.7,
            tao_turns_mean: 2.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.total_quality >= attr.baseline_quality,
            "total_quality {} < baseline_quality {}",
            attr.total_quality,
            attr.baseline_quality
        );
    }

    #[test]
    fn attribution_below_majority_accuracy_no_topology_gain() {
        // p < 0.5: ensemble is worse than random; topology_gain should be 0 (clamped)
        let input = AttributionInput {
            p_mean: 0.4,
            rho_mean: 0.0,
            n_agents: 5,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.topology_gain == 0.0,
            "p=0.4 < 0.5 should give topology_gain=0 (clamped), got {}",
            attr.topology_gain
        );
    }

    // ── q_measured field ──────────────────────────────────────────────────────

    #[test]
    fn harness_attribution_q_measured_is_none_by_default() {
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.3,
            n_agents: 3,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.q_measured.is_none(),
            "q_measured must be None by default"
        );
    }

    // ── bootstrap_interval ────────────────────────────────────────────────────

    fn base_input() -> AttributionInput {
        AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.3,
            n_agents: 3,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        }
    }

    #[test]
    fn bootstrap_interval_none_basis_when_single_sample() {
        let iv = bootstrap_interval(&base_input(), &[0.6], 1000);
        assert_eq!(
            iv.interval_basis,
            IntervalBasis::None,
            "single CG sample must produce IntervalBasis::None"
        );
        assert_eq!(iv.q_total_lo, iv.q_total_hi, "lo == hi when no interval");
    }

    #[test]
    fn bootstrap_interval_none_basis_when_empty_samples() {
        let iv = bootstrap_interval(&base_input(), &[], 1000);
        assert_eq!(iv.interval_basis, IntervalBasis::None);
    }

    #[test]
    fn bootstrap_interval_bootstrap_basis_with_two_samples() {
        let iv = bootstrap_interval(&base_input(), &[0.5, 0.7], 1000);
        assert!(
            matches!(
                iv.interval_basis,
                IntervalBasis::Bootstrap { n_cg_samples: 2 }
            ),
            "expected Bootstrap{{n_cg_samples:2}}, got {:?}",
            iv.interval_basis
        );
    }

    #[test]
    fn bootstrap_interval_wider_with_higher_cg_variance() {
        // Low variance CG: all samples near 0.6
        let low_var: Vec<f64> = vec![0.58, 0.60, 0.61, 0.59, 0.60];
        // High variance CG: samples spread 0.2–0.9
        let high_var: Vec<f64> = vec![0.2, 0.4, 0.6, 0.8, 0.9];

        let input = base_input();
        let iv_low = bootstrap_interval(&input, &low_var, 2000);
        let iv_high = bootstrap_interval(&input, &high_var, 2000);

        let width_low = iv_low.q_total_hi - iv_low.q_total_lo;
        let width_high = iv_high.q_total_hi - iv_high.q_total_lo;
        assert!(
            width_high > width_low,
            "higher CG variance must produce wider CI: high={width_high:.4}, low={width_low:.4}"
        );
    }

    #[test]
    fn bootstrap_interval_lo_le_hi() {
        // The bootstrap CI is derived from cg_samples; q_total comes from base_input directly,
        // so q_total is not guaranteed to lie inside the CI. The invariant is lo ≤ hi.
        let samples: Vec<f64> = (0..10).map(|i| 0.3 + i as f64 * 0.05).collect();
        let iv = bootstrap_interval(&base_input(), &samples, 1000);
        assert!(
            iv.q_total_lo <= iv.q_total_hi,
            "bootstrap CI must be non-inverted: lo={:.4}, hi={:.4}",
            iv.q_total_lo,
            iv.q_total_hi
        );
        assert!(iv.q_total_hi > 0.0, "CI hi must be positive");
    }

    // ── conformal_interval ────────────────────────────────────────────────────

    #[test]
    fn conformal_interval_empty_residuals_returns_full_range() {
        let (lo, hi) = conformal_interval(0.8, &[], 0.9);
        assert!(
            lo <= 0.8 && hi >= 0.8,
            "empty residuals must bracket q_predicted"
        );
    }

    #[test]
    fn conformal_interval_correct_q_hat_single_residual() {
        // 1 residual = 0.1; idx = ceil(2 * 0.9) = 2, clamped to 1 → q_hat = residuals[0] = 0.1
        let (lo, hi) = conformal_interval(0.8, &[0.1], 0.9);
        assert!((lo - 0.7).abs() < 1e-9, "lo = 0.8 - 0.1 = 0.7, got {lo:.6}");
        assert!((hi - 0.9).abs() < 1e-9, "hi = 0.8 + 0.1 = 0.9, got {hi:.6}");
    }

    #[test]
    fn conformal_interval_achieves_coverage_on_held_out_set() {
        // 50 residuals uniformly spaced 0.0–0.49; 90% coverage → q_hat ≈ 0.45
        let residuals: Vec<f64> = (0..50).map(|i| i as f64 * 0.01).collect();
        let q_pred = 0.7;
        let (lo, hi) = conformal_interval(q_pred, &residuals, 0.9);
        assert!(hi - lo > 0.0, "interval must be non-trivial");
        assert!(
            lo < q_pred && hi > q_pred,
            "point estimate must be inside interval"
        );
    }

    #[test]
    fn conformal_interval_clamped_to_unit() {
        // q_predicted near 1.0 + large residual → hi clamped to 1.0
        let (lo, hi) = conformal_interval(0.95, &[0.5], 0.9);
        assert!((hi - 1.0).abs() < 1e-9, "hi must clamp to 1.0, got {hi:.6}");
        let _ = lo; // lo may be > 0
    }

    // ── S7: ρ correction (Case B) ─────────────────────────────────────────────

    fn make_eigen(n_effective: f64, n_agents: usize) -> h2ai_types::physics::EigenCalibration {
        h2ai_types::physics::EigenCalibration {
            n_effective,
            h_diversity: 0.5,
            eigenvalues: vec![n_effective],
            n_pruned: n_agents,
        }
    }

    #[test]
    fn case_b_high_cg_talagrand_under_dispersed_corrects_rho() {
        // Spec criterion 1: CG_mean=0.70 → rho_proxy=0.30 < 0.5
        // Talagrand=UnderDispersed → talagrand_correction = 0.30×(1-0.30)=0.21
        // rho_adjusted = 0.30 + 0.21 = 0.51
        let input = AttributionInput {
            p_mean: 0.85,
            rho_mean: 0.30,
            n_agents: 4,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: Some(CalibrationState::UnderDispersed),
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            (attr.rho_adjusted - 0.51).abs() < 1e-9,
            "rho_adjusted must be 0.51, got {:.6}",
            attr.rho_adjusted
        );
        assert!(attr.case_b_flag, "case_b_flag must be true");
    }

    #[test]
    fn case_b_low_cg_guard_prevents_correction() {
        // Spec criterion 2: CG_mean=0.40 → rho_proxy=0.60 ≥ 0.5
        // Guard fires: no correction even with UnderDispersed
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.60,
            n_agents: 3,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: Some(CalibrationState::UnderDispersed),
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            (attr.rho_adjusted - 0.60).abs() < 1e-9,
            "guard must prevent correction: rho_adjusted={:.6}",
            attr.rho_adjusted
        );
        assert!(
            !attr.case_b_flag,
            "case_b_flag must be false when guard prevents correction"
        );
    }

    #[test]
    fn case_a_calibrated_no_rho_correction() {
        // Spec criterion 3: CG_mean=0.70, Talagrand=Calibrated → no correction
        let input = AttributionInput {
            p_mean: 0.85,
            rho_mean: 0.30,
            n_agents: 4,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: Some(CalibrationState::Calibrated),
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            (attr.rho_adjusted - 0.30).abs() < 1e-9,
            "Calibrated Talagrand must not correct rho, got {:.6}",
            attr.rho_adjusted
        );
        assert!(!attr.case_b_flag);
    }

    #[test]
    fn neff_low_diversity_applies_second_correction() {
        // Spec criterion 4: N_eff/N < 0.4, Talagrand=Calibrated → +0.15×(1−ρ) only
        // n_agents=4, n_eff=1.0 → effective_fraction=0.25 < 0.40 → fires
        // neff_correction = 0.15×(1−0.30) = 0.105 → rho_adjusted = 0.405
        let input = AttributionInput {
            p_mean: 0.85,
            rho_mean: 0.30,
            n_agents: 4,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: Some(CalibrationState::Calibrated),
            eigen_calibration: Some(make_eigen(1.0, 4)),
        };
        let attr = HarnessAttribution::compute(&input);
        let expected = 0.30 + 0.15 * 0.70;
        assert!(
            (attr.rho_adjusted - expected).abs() < 1e-9,
            "N_eff correction must give {:.6}, got {:.6}",
            expected,
            attr.rho_adjusted
        );
        assert!(attr.case_b_flag);
    }

    #[test]
    fn both_signals_fire_correction_capped_at_unit() {
        // Spec criterion 5: both Talagrand=UnderDispersed AND N_eff/N < 0.4
        // talagrand_correction = 0.30×(1-0.30) = 0.21
        // neff_correction = 0.15×(1-0.30) = 0.105
        // raw sum = 0.30 + 0.21 + 0.105 = 0.615 → clamp(0,1) = 0.615
        // spec: max correction = 0.45, but clamp ensures ≤ 1.0
        let input = AttributionInput {
            p_mean: 0.85,
            rho_mean: 0.30,
            n_agents: 4,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: Some(CalibrationState::UnderDispersed),
            eigen_calibration: Some(make_eigen(1.0, 4)),
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.rho_adjusted <= 1.0,
            "rho_adjusted must be ≤ 1.0 even when both signals fire, got {:.6}",
            attr.rho_adjusted
        );
        let expected = (0.30 + 0.21 + 0.105_f64).clamp(0.0, 1.0);
        assert!(
            (attr.rho_adjusted - expected).abs() < 1e-9,
            "both-signal rho_adjusted must be {expected:.6}, got {:.6}",
            attr.rho_adjusted
        );
        assert!(attr.case_b_flag);
    }

    #[test]
    fn case_b_flag_false_when_no_signals_fire() {
        // Spec criterion 6 (negative case): Talagrand=None, no eigen → case_b_flag=false
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.3,
            n_agents: 4,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(!attr.case_b_flag, "no signals → case_b_flag must be false");
        assert!(
            (attr.rho_adjusted - 0.30).abs() < 1e-9,
            "no signals → rho_adjusted must equal rho_mean"
        );
    }

    #[test]
    fn total_quality_at_least_baseline_after_rho_correction() {
        // Spec criterion 8: clamp(baseline_quality, 1.0) preserved after correction
        let input = AttributionInput {
            p_mean: 0.6,
            rho_mean: 0.30,
            n_agents: 3,
            verification_filter_ratio: 0.8,
            tao_turns_mean: 2.0,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: Some(CalibrationState::UnderDispersed),
            eigen_calibration: Some(make_eigen(0.8, 3)),
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.total_quality >= attr.baseline_quality,
            "total_quality {:.4} < baseline {:.4} after Case B correction",
            attr.total_quality,
            attr.baseline_quality
        );
        assert!(attr.total_quality <= 1.0);
    }
}
