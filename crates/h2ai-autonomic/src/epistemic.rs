#![allow(clippy::cast_precision_loss)]
use h2ai_context::embedding::{cosine_similarity, EmbeddingModel};
use h2ai_types::events::{ConstraintViolation, FailureMode};
use h2ai_types::sizing::EigenCalibration;
use nalgebra::DMatrix;

/// Compute `N_eff` (effective independent adapters) from a set of proposal or output texts.
///
/// Embeds each text, builds the N×N cosine matrix C (diagonal = 1.0), normalises
/// K = C / N so trace(K) = 1, then computes `N_eff` via `EigenCalibration::from_cosine_matrix`.
/// Returns 1.0 for fewer than 2 texts (degenerate — only one perspective).
pub fn compute_n_eff_cosine(texts: &[String], model: &dyn EmbeddingModel, delta: f64) -> f64 {
    let n = texts.len();
    if n < 2 {
        return 1.0;
    }
    let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| model.embed(t)).collect();

    // Build raw cosine matrix C (symmetric, diagonal = 1.0).
    let mut c = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        c[(i, i)] = 1.0;
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]).max(0.0);
            c[(i, j)] = sim;
            c[(j, i)] = sim;
        }
    }

    // Normalise: K = C / N so trace(K) = 1 and eigenvalues sum to 1.
    let k = c / n as f64;
    EigenCalibration::from_cosine_matrix(&k, delta).n_effective
}

/// Compute the mean pairwise cosine similarity across all (i, j) pairs (i < j) in `texts`.
///
/// Returns `None` for fewer than 2 texts (no pairs to compare).
/// Clamps raw cosine to `[0.0, 1.0]` before averaging.
pub fn mean_pairwise_cosine(texts: &[String], model: &dyn EmbeddingModel) -> Option<f64> {
    let n = texts.len();
    if n < 2 {
        return None;
    }
    let embeddings: Vec<Vec<f32>> = texts.iter().map(|t| model.embed(t)).collect();
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]).max(0.0);
            sum += sim;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    Some(sum / count as f64)
}

/// Classify a zero-survival event as `ConstrainedExploration` or `ModeCollapse`.
///
/// Boundary: `n_eff > diversity_threshold × n_requested` → `ConstrainedExploration`.
/// When `diversity_threshold` is 0.0, the boundary is 0.0 — any positive `N_eff`
/// (which always ≥ 1.0) will produce `ConstrainedExploration`. Set `diversity_threshold`
/// to a meaningful value (e.g. 0.5) in `H2AIConfig` for production routing.
#[must_use]
pub fn classify_failure_mode(
    n_eff: f64,
    n_requested: usize,
    diversity_threshold: f64,
) -> FailureMode {
    if n_eff > diversity_threshold * n_requested as f64 {
        FailureMode::ConstrainedExploration
    } else {
        FailureMode::ModeCollapse
    }
}

// ── GAP-F7: ConstraintRepairPlan — structured retry instructions ───────────────

/// Per-constraint repair guidance for one retry wave.
#[derive(Debug, Clone)]
pub struct ConstraintRepairEntry {
    pub constraint_id: String,
    pub severity_label: String,
    pub score: f64,
    /// What the constraint requires. From `criteria_pass` → `constraint_description` fallback.
    pub rule: String,
    /// What the verifier found wrong. From `verifier_reason`; failed check indices appended
    /// when `check_verdicts` contains false entries.
    pub what_failed: String,
    /// Actionable repair guidance. From `remediation_hint`; generic fallback when absent.
    pub what_to_try: String,
}

/// Machine-actionable repair plan for one retry wave (GAP-F7).
/// Renders to a structured prompt block injected into explorer context for waves 2+.
#[derive(Debug, Clone)]
pub struct ConstraintRepairPlan {
    pub entries: Vec<ConstraintRepairEntry>,
}

impl ConstraintRepairPlan {
    /// Render as a structured prompt section.
    /// Raw proposal text is never included — only constraint-derived information.
    pub fn render(&self) -> String {
        let mut parts = vec![
            "Your previous attempt violated these constraints. Revise your approach:\n".to_string(),
        ];
        for e in &self.entries {
            parts.push(format!(
                "### [{id}] score={score:.2} [{sev}]\n\
                 **Rule:** {rule}\n\
                 **What failed:** {what_failed}\n\
                 **Try:** {what_to_try}",
                id = e.constraint_id,
                score = e.score,
                sev = e.severity_label,
                rule = e.rule,
                what_failed = e.what_failed,
                what_to_try = e.what_to_try,
            ));
        }
        parts.join("\n\n")
    }
}

/// Build a `ConstraintRepairPlan` from a list of violations.
/// Returns `None` when `violations` is empty.
#[must_use]
pub fn synthesize_repair_plan(violations: &[ConstraintViolation]) -> Option<ConstraintRepairPlan> {
    if violations.is_empty() {
        return None;
    }
    let entries = violations
        .iter()
        .map(|v| {
            let rule = v
                .criteria_pass
                .as_deref()
                .filter(|s| !s.is_empty())
                .or(if v.constraint_description.is_empty() {
                    None
                } else {
                    Some(v.constraint_description.as_str())
                })
                .unwrap_or("see constraint definition")
                .to_string();

            let mut what_failed = v
                .verifier_reason
                .as_deref()
                .unwrap_or(&format!("constraint not satisfied (score={:.2})", v.score))
                .to_string();
            // Append failed check indices when available.
            let failed_checks: Vec<usize> = v
                .check_verdicts
                .iter()
                .enumerate()
                .filter_map(|(i, &ok)| if !ok { Some(i + 1) } else { None })
                .collect();
            if !failed_checks.is_empty() {
                let indices = failed_checks
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                what_failed.push_str(&format!(" (checks failed: {indices})"));
            }

            let what_to_try = v
                .remediation_hint
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or("revise the approach to satisfy this constraint")
                .to_string();

            ConstraintRepairEntry {
                constraint_id: v.constraint_id.clone(),
                severity_label: v.severity_label.clone(),
                score: v.score,
                rule,
                what_failed,
                what_to_try,
            }
        })
        .collect();
    Some(ConstraintRepairPlan { entries })
}

/// Synthesise a structured repair plan and render it as a prompt string.
/// Returns `None` when `violations` is empty (no wave context to inject).
#[must_use]
pub fn synthesize_tombstone(violations: &[ConstraintViolation]) -> Option<String> {
    synthesize_repair_plan(violations).map(|p| p.render())
}

// ── Pipeline Resilience: Frozen Verifier Detection ────────────────────────────

/// Signal emitted when detect_frozen_verifier identifies a stuck verifier judgment.
#[derive(Debug, Clone)]
pub struct FrozenVerifierSignal {
    pub constraint_id: String,
    /// Wave index at which the frozen pattern first became detectable (caller fills this in).
    pub frozen_since_wave: u32,
    /// Per-wave mean scores for this constraint (the detection window).
    pub per_wave_scores: Vec<f64>,
    /// Most recent verifier reason from reason_history.
    pub sample_reason: String,
}

/// Pure function: detect a stuck verifier for a single constraint.
///
/// Fires when ALL five conditions hold:
/// 1. wave_scores.len() >= cfg.min_waves_to_detect
/// 2. score_range(wave_scores) < cfg.score_variance_threshold
/// 3. other_constraint_trends is non-empty
/// 4. At least one entry in other_constraint_trends is monotonically non-decreasing
///    with at least one strict increase AND mean(last N) > cfg.other_constraint_success_threshold
/// 5. mean pairwise Jaccard of reason_history > cfg.reason_jaccard_threshold
#[must_use]
pub fn detect_frozen_verifier(
    constraint_id: &str,
    wave_scores: &[f64],
    reason_history: &[String],
    other_constraint_trends: &[&[f64]],
    cfg: &h2ai_config::VerifierFreezeConfig,
) -> Option<FrozenVerifierSignal> {
    if wave_scores.len() < cfg.min_waves_to_detect as usize {
        return None;
    }
    if other_constraint_trends.is_empty() {
        return None;
    }
    let range = score_range(wave_scores);
    if range >= cfg.score_variance_threshold {
        return None;
    }
    let any_other_succeeding = other_constraint_trends.iter().any(|scores| {
        is_monotonically_improving(scores)
            && mean_last_n(scores, cfg.min_waves_to_detect as usize)
                > cfg.other_constraint_success_threshold
    });
    if !any_other_succeeding {
        return None;
    }
    if reason_history.len() < 2 {
        return None;
    }
    let mean_j = mean_pairwise_jaccard_str(reason_history);
    if mean_j <= cfg.reason_jaccard_threshold {
        return None;
    }

    let sample_reason = reason_history.last().cloned().unwrap_or_default();
    Some(FrozenVerifierSignal {
        constraint_id: constraint_id.to_string(),
        frozen_since_wave: 0,
        per_wave_scores: wave_scores.to_vec(),
        sample_reason,
    })
}

fn score_range(scores: &[f64]) -> f64 {
    if scores.len() < 2 {
        return 0.0;
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}

fn is_monotonically_improving(scores: &[f64]) -> bool {
    if scores.len() < 2 {
        return false;
    }
    let mut had_strict_increase = false;
    for window in scores.windows(2) {
        if window[1] < window[0] {
            return false;
        }
        if window[1] > window[0] {
            had_strict_increase = true;
        }
    }
    had_strict_increase
}

fn mean_last_n(scores: &[f64], n: usize) -> f64 {
    let slice = if scores.len() > n {
        &scores[scores.len() - n..]
    } else {
        scores
    };
    if slice.is_empty() {
        return 0.0;
    }
    slice.iter().sum::<f64>() / slice.len() as f64
}

fn mean_pairwise_jaccard_str(reasons: &[String]) -> f64 {
    let n = reasons.len();
    if n < 2 {
        return 1.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += h2ai_constraints::ambiguity::jaccard(&reasons[i], &reasons[j]);
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        sum / count as f64
    }
}

// ── GAP-E2: Talagrand KL τ-spread update rule ────────────────────────────────

/// Compute the KL-divergence-based τ-spread delta for the Talagrand update rule.
///
/// `histogram` — normalised rank counts for ranks 1..N (pass `&rank_histogram[1..]`; index 0
/// of the raw `TalagrandDiagnostic::rank_histogram` is unused and must be skipped by the caller).
/// `eta` — learning rate controlling the magnitude of τ adjustment.
///
/// Returns Δτ = η × (U_score − Λ_score), where:
/// - U_score = var(H) / mean(H): elevated when the histogram is U-shaped (over-confident).
/// - Λ_score = max(H[middle]) / mean(H[edges]): elevated when centre mass exceeds edge mass.
///
/// Positive Δτ → expand τ-spread (U-shape); negative Δτ → contract τ-spread (Λ-shape).
/// Caller clips the result to [τ_min, τ_max]. Returns 0.0 for < 3 bins or zero total.
#[must_use]
pub fn talagrand_kl_delta_tau(histogram: &[f64], eta: f64) -> f64 {
    let n = histogram.len();
    if n < 3 {
        return 0.0;
    }
    let total: f64 = histogram.iter().sum();
    if total < 1e-10 {
        return 0.0;
    }
    let h: Vec<f64> = histogram.iter().map(|&c| c / total).collect();

    // U_score: dispersion relative to the uniform mean 1/N.
    let mean_h = 1.0 / n as f64;
    let var_h: f64 = h.iter().map(|x| (x - mean_h).powi(2)).sum::<f64>() / n as f64;
    let u_score = var_h / mean_h;

    // Λ_score: peak of middle bins relative to average of the extreme bins.
    let h_middle_max = h[1..n - 1]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let h_edges_mean = (h[0] + h[n - 1]) / 2.0;
    let lambda_score = if h_edges_mean > 1e-10 {
        h_middle_max / h_edges_mean
    } else {
        0.0
    };

    eta * (u_score - lambda_score)
}
