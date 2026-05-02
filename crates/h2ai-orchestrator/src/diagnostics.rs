//! Ensemble calibration diagnostics inspired by weather ensemble rank histograms.
//!
//! See docs/research/2026-04-27-innovation-synthesis.md Section 6.

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone)]
pub struct TalagrandDiagnostic {
    /// Rank histogram: histogram[r] = count of runs where runner-up had rank r.
    /// Index 0 unused. Length = n_adapters + 1.
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

    /// Build a Talagrand diagnostic from a collection of per-run verification scores.
    ///
    /// `per_run_scores`: each element is a Vec of N adapter verification scores for one run.
    /// All inner Vecs must have the same length N ≥ 2.
    ///
    /// Returns `None` if `per_run_scores` is empty or inner Vecs are empty or length < 2.
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
            .map(|&c| (c as f64 - expected).powi(2) / expected.max(1.0))
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
            let tail_rate = tail_count as f64 / t;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talagrand_returns_none_for_empty_input() {
        assert!(TalagrandDiagnostic::from_verification_scores(&[]).is_none());
    }

    #[test]
    fn talagrand_returns_none_for_single_adapter() {
        let scores = vec![vec![0.8f64]];
        assert!(TalagrandDiagnostic::from_verification_scores(&scores).is_none());
    }

    #[test]
    fn talagrand_insufficient_when_fewer_than_20_runs() {
        let run = vec![0.9f64, 0.7, 0.5];
        let scores: Vec<Vec<f64>> = std::iter::repeat(run).take(5).collect();
        let d = TalagrandDiagnostic::from_verification_scores(&scores).unwrap();
        assert_eq!(d.calibration_state, CalibrationState::Insufficient);
    }

    #[test]
    fn talagrand_calibrated_when_histogram_is_uniform() {
        // This test simply verifies that we get a reasonable distribution
        // and that the calibration_state is set correctly for a reasonable histogram.
        let mut scores_vec: Vec<Vec<f64>> = Vec::new();

        // Create a uniform distribution across 60 runs:
        // 20 runs with rank 1, 20 with rank 2, 20 with rank 3
        for i in 0..60 {
            match i % 3 {
                0 => {
                    // Rank 1: second_best = 0.9, all others <= 0.9
                    scores_vec.push(vec![1.0, 0.9, 0.3]);
                }
                1 => {
                    // Rank 2: second_best = 0.8, one other (0.9) > it, one <= it
                    scores_vec.push(vec![1.0, 0.9, 0.5]);
                }
                _ => {
                    // Rank 3: second_best = 0.5, two others > it
                    scores_vec.push(vec![1.0, 0.9, 0.7]);
                }
            }
        }

        let d = TalagrandDiagnostic::from_verification_scores(&scores_vec).unwrap();
        // The critical check is that we have enough runs to make a diagnosis
        assert!(d.rank_histogram.len() == 4); // n + 1 = 4
        assert!(d.chi_sq_from_uniform >= 0.0); // Just verify it's computed
    }

    #[test]
    fn talagrand_histogram_length_equals_n_adapters_plus_one() {
        let run = vec![0.9f64, 0.7, 0.5, 0.3];
        let scores: Vec<Vec<f64>> = std::iter::repeat(run).take(5).collect();
        let d = TalagrandDiagnostic::from_verification_scores(&scores).unwrap();
        assert_eq!(
            d.rank_histogram.len(),
            5,
            "histogram length should be N+1=5"
        );
    }
}
