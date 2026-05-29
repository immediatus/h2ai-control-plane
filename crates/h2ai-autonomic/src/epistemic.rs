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

/// Synthesise a dense Constraint Violation Tombstone from a list of violations.
///
/// Extracts only constraint IDs, severity labels, and per-constraint scores — never
/// raw proposal text or remediation hints. Keeps context window α low and avoids
/// "Lost in the Middle" attention degradation on re-tries.
/// Returns `None` when `violations` is empty.
#[must_use]
pub fn synthesize_tombstone(violations: &[ConstraintViolation]) -> Option<String> {
    if violations.is_empty() {
        return None;
    }
    let mut lines = vec!["Previous attempts failed the following constraints:".to_string()];
    for v in violations {
        lines.push(format!(
            "• {} [{}]: score={:.2} — constraint not satisfied",
            v.constraint_id, v.severity_label, v.score
        ));
    }
    lines.push("Do not repeat these patterns.".to_string());
    Some(lines.join("\n"))
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
