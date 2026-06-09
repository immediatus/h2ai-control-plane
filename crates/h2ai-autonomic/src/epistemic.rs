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
        let mut parts =
            vec!["Your previous attempt violated these constraints. Revise your approach:\n"
                .to_string()];
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
                .or_else(|| {
                    if v.constraint_description.is_empty() {
                        None
                    } else {
                        Some(v.constraint_description.as_str())
                    }
                })
                .unwrap_or("see constraint definition")
                .to_string();

            let mut what_failed = v
                .verifier_reason
                .as_deref()
                .unwrap_or(&format!(
                    "constraint not satisfied (score={:.2})",
                    v.score
                ))
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

#[cfg(test)]
mod repair_plan_tests {
    use super::*;
    use h2ai_types::events::ConstraintViolation;

    fn v(id: &str, sev: &str, score: f64) -> ConstraintViolation {
        ConstraintViolation {
            constraint_id: id.to_string(),
            score,
            severity_label: sev.to_string(),
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
        }
    }

    #[test]
    fn empty_violations_returns_none() {
        assert!(synthesize_repair_plan(&[]).is_none());
        assert!(synthesize_tombstone(&[]).is_none());
    }

    #[test]
    fn render_contains_constraint_id_and_score() {
        let plan = synthesize_repair_plan(&[v("C-001", "Hard", 0.32)]).unwrap();
        let s = plan.render();
        assert!(s.contains("C-001"));
        assert!(s.contains("0.32"));
        assert!(s.contains("Hard"));
    }

    #[test]
    fn rule_uses_criteria_pass_first() {
        let mut violation = v("C-1", "Hard", 0.0);
        violation.criteria_pass = Some("use atomic Lua EVAL".into());
        violation.constraint_description = "fallback description".into();
        let plan = synthesize_repair_plan(&[violation]).unwrap();
        let s = plan.render();
        assert!(s.contains("atomic Lua EVAL"), "criteria_pass must win");
        assert!(!s.contains("fallback description"));
    }

    #[test]
    fn rule_falls_back_to_description_when_no_criteria_pass() {
        let mut violation = v("C-1", "Hard", 0.0);
        violation.criteria_pass = None;
        violation.constraint_description = "use circuit breakers".into();
        let plan = synthesize_repair_plan(&[violation]).unwrap();
        assert!(plan.render().contains("circuit breakers"));
    }

    #[test]
    fn what_to_try_uses_remediation_hint() {
        let mut violation = v("C-1", "Hard", 0.0);
        violation.remediation_hint = Some("wrap calls with Resilience4j".into());
        let plan = synthesize_repair_plan(&[violation]).unwrap();
        assert!(plan.render().contains("Resilience4j"));
    }

    #[test]
    fn what_failed_uses_verifier_reason() {
        let mut violation = v("C-1", "Hard", 0.2);
        violation.verifier_reason = Some("non-atomic GET-SET detected".into());
        let plan = synthesize_repair_plan(&[violation]).unwrap();
        assert!(plan.render().contains("non-atomic GET-SET detected"));
    }

    #[test]
    fn failed_check_indices_appended_to_what_failed() {
        let mut violation = v("C-1", "Hard", 0.5);
        violation.check_verdicts = vec![true, false, true, false];
        let plan = synthesize_repair_plan(&[violation]).unwrap();
        let s = plan.render();
        // Checks 2 and 4 failed (1-indexed)
        assert!(s.contains("checks failed: 2, 4"), "got: {s}");
    }

    #[test]
    fn raw_proposal_text_never_appears() {
        // The LLM's actual proposal text must never be injected (anchoring hazard).
        // This is enforced structurally: synthesize_repair_plan only reads typed fields.
        let raw_proposal = "The system uses PKCE with rolling refresh tokens";
        let mut violation = v("C-1", "Hard", 0.0);
        violation.verifier_reason = Some("missing Lua atomicity".into());
        let plan = synthesize_repair_plan(&[violation]).unwrap();
        assert!(!plan.render().contains(raw_proposal));
    }

    #[test]
    fn multiple_violations_all_rendered() {
        let vs = vec![v("A-1", "Hard", 0.1), v("B-2", "Soft", 0.3)];
        let plan = synthesize_repair_plan(&vs).unwrap();
        let s = plan.render();
        assert!(s.contains("A-1"));
        assert!(s.contains("B-2"));
    }

    #[test]
    fn tombstone_delegates_to_render() {
        let s = synthesize_tombstone(&[v("C-1", "Hard", 0.0)]).unwrap();
        // Must contain the constraint ID (same as repair plan)
        assert!(s.contains("C-1"));
    }
}

#[cfg(test)]
mod pairwise_cosine_tests {
    use super::*;

    struct FakeEmbedder {
        embeddings: std::collections::HashMap<String, Vec<f32>>,
    }

    impl EmbeddingModel for FakeEmbedder {
        fn embed(&self, text: &str) -> Vec<f32> {
            self.embeddings.get(text).cloned().unwrap_or_default()
        }
    }

    #[test]
    fn mean_pairwise_cosine_returns_none_for_single_text() {
        let model = FakeEmbedder {
            embeddings: Default::default(),
        };
        let result = mean_pairwise_cosine(&["hello".to_string()], &model);
        assert!(result.is_none());
    }

    #[test]
    fn mean_pairwise_cosine_identical_texts_returns_one() {
        let mut emb = std::collections::HashMap::new();
        emb.insert("a".to_string(), vec![1.0_f32, 0.0]);
        emb.insert("b".to_string(), vec![1.0_f32, 0.0]);
        let model = FakeEmbedder { embeddings: emb };
        let result = mean_pairwise_cosine(&["a".to_string(), "b".to_string()], &model).unwrap();
        assert!((result - 1.0).abs() < 1e-5, "expected ~1.0 got {result}");
    }

    #[test]
    fn mean_pairwise_cosine_orthogonal_returns_zero() {
        let mut emb = std::collections::HashMap::new();
        emb.insert("x".to_string(), vec![1.0_f32, 0.0]);
        emb.insert("y".to_string(), vec![0.0_f32, 1.0]);
        let model = FakeEmbedder { embeddings: emb };
        let result = mean_pairwise_cosine(&["x".to_string(), "y".to_string()], &model).unwrap();
        assert!(result.abs() < 1e-5, "expected ~0.0 got {result}");
    }

    #[test]
    fn mean_pairwise_cosine_three_texts_averages_pairs() {
        let mut emb = std::collections::HashMap::new();
        for k in ["a", "b", "c"] {
            emb.insert(k.to_string(), vec![1.0_f32, 0.0]);
        }
        let model = FakeEmbedder { embeddings: emb };
        let texts: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let result = mean_pairwise_cosine(&texts, &model).unwrap();
        assert!((result - 1.0).abs() < 1e-5, "expected ~1.0 got {result}");
    }
}
