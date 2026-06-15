//! Ensemble calibration diagnostics inspired by weather ensemble rank histograms.
//!
//! See internal research notes Section 6.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationState {
    /// χ² test passes (histogram uniform): ensemble is well-calibrated.
    Calibrated,
    /// Tail ranks too frequent (U-shape): ensemble is over-confident.
    OverConfident,
    /// Middle ranks too frequent (Λ-shape): ensemble is under-dispersed.
    UnderDispersed,
    /// Fewer than 20 runs observed: not enough data.
    Insufficient,
}

/// Rank histogram calibration diagnostic for the proposal ensemble.
///
/// Adapted from Hamill (2001) / Talagrand (1997) numerical weather prediction diagnostics.
///
/// **Validity condition:** Rank histograms are calibrated diagnostics only when the
/// verification scores are exchangeable draws from the same probability distribution —
/// i.e., the scoring function is a calibrated probabilistic forecaster. LLM-as-judge
/// scores do not satisfy this: scores for different tasks/prompts are not exchangeable,
/// and LLM judges are known to exhibit length bias and score miscalibration. The
/// histogram shape (flat/U/Λ) is observable, but attributing it to ensemble calibration
/// vs. evaluator drift requires an independent ground-truth signal (Tier 1 oracle).
///
/// Use this diagnostic as a weak signal for relative comparisons, not as an absolute
/// calibration measurement. The chi-squared statistic is computed against a uniform
/// prior; its p-value is only meaningful when the validity condition holds.
#[derive(Debug, Clone)]
pub struct TalagrandDiagnostic {
    /// Rank histogram: histogram[r] = count of runs where runner-up had rank r.
    /// Index 0 unused. Length = `n_adapters` + 1.
    pub rank_histogram: Vec<u32>,
    /// Chi-squared statistic testing uniformity of the rank histogram.
    pub chi_sq_from_uniform: f64,
    /// Ratio of proposal spread (std dev of scores) to mean top-score gap.
    /// Ideal ≈ 1.0 (ensemble spread matches actual score variation).
    pub spread_error_ratio: f64,
    pub calibration_state: CalibrationState,
}

impl TalagrandDiagnostic {
    /// Compute the τ-spread expansion factor for the next MAPE-K iteration.
    ///
    /// - `OverConfident` (U-curve): expand by 20%, capped at `max_factor`.
    /// - `Calibrated` or `Insufficient`: no change (return `current_factor`).
    /// - `UnderDispersed` (Λ-curve): proposals are converging; logs a warning and
    ///   returns `current_factor` unchanged (expanding τ doesn't help when adapters share a bias).
    pub fn tau_expansion_factor(&self, current_factor: f64, max_factor: f64) -> f64 {
        match self.calibration_state {
            CalibrationState::OverConfident => {
                (current_factor * 1.2).min(max_factor.max(current_factor))
            }
            CalibrationState::UnderDispersed => {
                tracing::warn!(
                    chi_sq = self.chi_sq_from_uniform,
                    "Talagrand Λ-curve detected: proposals may be converging on a shared \
                     incorrect answer. Consider increasing adapter diversity."
                );
                current_factor
            }
            CalibrationState::Calibrated | CalibrationState::Insufficient => current_factor,
        }
    }

    /// Compute the next τ-spread factor using the principled KL-divergence update rule (GAP-E2).
    ///
    /// Δτ = η × (U_score − Λ_score); τ_new = clip(current_factor + Δτ, tau_min, tau_max).
    /// Replaces the heuristic 1.2× expansion with an adaptive rule that also contracts τ
    /// when the histogram is Λ-shaped (under-dispersed, proposals converging).
    pub fn tau_kl_next(&self, current_factor: f64, eta: f64, tau_min: f64, tau_max: f64) -> f64 {
        let h: Vec<f64> = self.rank_histogram[1..].iter().map(|&c| c as f64).collect();
        let delta = h2ai_autonomic::epistemic::talagrand_kl_delta_tau(&h, eta);
        (current_factor + delta).clamp(tau_min, tau_max)
    }

    /// Build a Talagrand diagnostic from a collection of per-run verification scores.
    ///
    /// `per_run_scores`: each element is a Vec of N adapter verification scores for one run.
    /// All inner Vecs must have the same length N ≥ 2.
    ///
    /// Returns `None` if `per_run_scores` is empty or inner Vecs are empty or length < 2.
    #[must_use]
    pub fn from_verification_scores(per_run_scores: &[Vec<f64>]) -> Option<Self> {
        if per_run_scores.is_empty() {
            return None;
        }
        let n = per_run_scores[0].len();
        if n < 2 {
            return None;
        }

        let mut histogram = vec![0u32; n + 1];
        let mut spread_sum = 0.0f64;
        let mut gap_sum = 0.0f64;

        for scores in per_run_scores {
            if scores.len() != n {
                continue;
            }
            // Find the top score and its position
            let (top_idx, &top) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &f64::NEG_INFINITY));

            // Find the second-highest score (excluding the top)
            let mut second = f64::NEG_INFINITY;
            for (i, &s) in scores.iter().enumerate() {
                if i != top_idx && s > second {
                    second = s;
                }
            }

            // Rank is the count of how many scores are strictly greater than second
            let rank = if second == f64::NEG_INFINITY {
                n / 2
            } else {
                scores.iter().filter(|&&s| s > second).count()
            };
            histogram[rank.min(n)] += 1;

            let mean = scores.iter().sum::<f64>() / n as f64;
            let spread = (scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            let gap = top - mean;
            spread_sum += spread;
            gap_sum += gap;
        }

        let t = per_run_scores.len() as f64;
        let expected = t / n as f64;
        let chi_sq: f64 = histogram
            .iter()
            .skip(1)
            .map(|&c| (f64::from(c) - expected).powi(2) / expected.max(1.0))
            .sum();

        let spread_error_ratio = if gap_sum > 1e-10 {
            spread_sum / gap_sum
        } else {
            1.0
        };

        let state = if t < 20.0 {
            CalibrationState::Insufficient
        } else if chi_sq < 3.84 {
            CalibrationState::Calibrated
        } else {
            let tail_count = histogram[1] + histogram[n];
            let tail_rate = f64::from(tail_count) / t;
            if tail_rate > 2.0 / n as f64 {
                CalibrationState::OverConfident
            } else {
                CalibrationState::UnderDispersed
            }
        };

        Some(Self {
            rank_histogram: histogram,
            chi_sq_from_uniform: chi_sq,
            spread_error_ratio,
            calibration_state: state,
        })
    }
}
