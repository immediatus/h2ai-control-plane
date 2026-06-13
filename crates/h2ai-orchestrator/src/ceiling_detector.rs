//! Pure functions for intra-retry complexity ceiling detection.
//!
//! All functions are stateless and synchronous. They are called by the MAPE-K
//! controller after each retry wave to decide whether the current task has hit a
//! structural complexity ceiling that warrants routing to a higher-capability model.

use h2ai_config::IntraRetryDetectorConfig;
use h2ai_types::events::BranchPrunedEvent;
use std::collections::HashMap;

/// Compute the Shannon entropy of the constraint-failure distribution across the
/// given pruned events.
///
/// For each `ConstraintViolation` inside every event, the `constraint_id` is
/// counted. The resulting frequency distribution is normalised and its entropy
/// H = -Σ p·ln(p) is returned.
///
/// Returns `1.0` for an empty slice (maximum uncertainty / no information).
pub fn failure_signature_entropy(last_wave_pruned: &[BranchPrunedEvent]) -> f64 {
    let mut counts: HashMap<&str, u64> = HashMap::new();
    for event in last_wave_pruned {
        for violation in &event.violated_constraints {
            *counts.entry(violation.constraint_id.as_str()).or_insert(0) += 1;
        }
    }

    let total: u64 = counts.values().sum();
    if total == 0 {
        return 1.0;
    }

    let total_f = total as f64;
    let entropy = counts.values().fold(0.0_f64, |acc, &c| {
        let p = c as f64 / total_f;
        acc - p * p.ln()
    });

    // Normalise to [0, 1] by dividing by ln(n) where n is the number of unique
    // constraints. When n == 1 the raw entropy is 0 and normalisation is not
    // needed, but we guard the division anyway.
    let n = counts.len();
    if n <= 1 {
        return entropy; // 0.0 — perfectly peaked
    }
    entropy / (n as f64).ln()
}

/// Compute the relative score improvement between the last two entries of
/// `score_history`.
///
/// slope = (score[n-1] - score[n-2]) / score[n-2]
///
/// Returns `f64::INFINITY` when fewer than 2 scores are available — this means
/// "no stall detected" because there is no evidence of convergence failure yet.
/// `quality_history` only gains entries when a wave produces a winning proposal;
/// all-ZeroSurvival waves leave the history empty, which must not be mistaken
/// for a stalled slope.
///
/// Returns `0.0` when the previous score is zero (to avoid division by zero).
pub fn retry_slope(score_history: &[f64]) -> f64 {
    let n = score_history.len();
    if n < 2 {
        return f64::INFINITY;
    }
    let prev = score_history[n - 2];
    if prev == 0.0 {
        return 0.0;
    }
    let curr = score_history[n - 1];
    (curr - prev) / prev
}

/// Count how many of the three ceiling signals are active.
///
/// Signal 1 — failure entropy < `cfg.entropy_threshold`
///   (peaked failure pattern: all proposals fail the same constraint)
///
/// Signal 2 — retry slope < `cfg.retry_slope_threshold`
///   (score convergence failure: improvements have stalled)
///
/// Signal 3 — `n_eff * cg_mean` < `cfg.n_eff_cg_product_threshold`
///   (correlated failure: all paths fail for the same structural reason)
///
/// Returns a value in `0..=3`.
pub fn count_ceiling_signals(
    last_wave_pruned: &[BranchPrunedEvent],
    score_history: &[f64],
    n_eff: f64,
    cg_mean: f64,
    cfg: &IntraRetryDetectorConfig,
) -> u8 {
    let mut signals: u8 = 0;

    if failure_signature_entropy(last_wave_pruned) < cfg.entropy_threshold {
        signals += 1;
    }
    if retry_slope(score_history) < cfg.retry_slope_threshold {
        signals += 1;
    }
    if n_eff * cg_mean < cfg.n_eff_cg_product_threshold {
        signals += 1;
    }

    signals
}
