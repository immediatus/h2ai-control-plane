use crate::embedding::{semantic_jaccard as embed_semantic_jaccard, EmbeddingModel};

/// Semantic similarity between two text strings in [0, 1].
///
/// Delegates to [`crate::embedding::semantic_jaccard`]:
/// - **With model**: cosine similarity between embedding vectors
/// - **Without model** (`None`): token-level Jaccard similarity
pub fn semantic_jaccard(a: &str, b: &str, model: Option<&dyn EmbeddingModel>) -> f64 {
    embed_semantic_jaccard(a, b, model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_jaccard_none_identical_text_is_one() {
        let text = "stateless jwt auth token ADR-001";
        assert!((semantic_jaccard(text, text, None) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn semantic_jaccard_none_disjoint_text_is_zero() {
        assert_eq!(
            semantic_jaccard("jwt stateless auth", "redis cache store", None),
            0.0
        );
    }
}
