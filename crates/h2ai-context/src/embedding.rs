use crate::jaccard::{jaccard, tokenize};
#[cfg(feature = "fastembed-embed")]
use std::sync::Mutex;

/// A text embedding model that maps strings to dense float vectors.
///
/// Implement this trait to plug in any embedding model (sentence-transformers,
/// ONNX Runtime, remote API, etc.). The returned vector must be L2-normalised
/// for `semantic_jaccard` to give cosine similarity correctly; if your model
/// returns un-normalised vectors, normalise them in your `embed` implementation.
///
/// # Zero-cost fallback
///
/// All functions that accept `Option<&dyn EmbeddingModel>` fall back to
/// token-level Jaccard similarity when `None` is passed — existing deployments
/// pay no extra cost until they supply a model.
pub trait EmbeddingModel: Send + Sync {
    /// Embed `text` into a dense float vector.
    ///
    /// The vector dimension must be consistent across all calls on the same
    /// model instance. Returning an empty vec is treated as "no embedding
    /// available" and triggers the token-Jaccard fallback.
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// Compute cosine similarity between two embedding vectors.
///
/// Returns 0.0 for empty or zero-norm vectors. Both vectors must have the
/// same dimension; mismatched dimensions return 0.0.
///
/// Cosine similarity ∈ [-1, 1]. For sentence embeddings from standard
/// models (all-MiniLM, sentence-BERT) the range is effectively [0, 1].
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0) as f64
}

/// Semantic similarity between two text strings in [0, 1].
///
/// - **With model**: cosine similarity between embedding vectors. Catches
///   semantic equivalents that token overlap misses ("redis" ≈ "key-value store").
/// - **Without model** (`None`): token-level Jaccard similarity — zero extra
///   cost, identical to the existing implementation.
///
/// The return value is in [0, 1] in both modes: it can be interpreted as
/// the probability that two texts share the same semantic domain.
pub fn semantic_jaccard(a: &str, b: &str, model: Option<&dyn EmbeddingModel>) -> f64 {
    match model {
        Some(m) => {
            let va = m.embed(a);
            let vb = m.embed(b);
            if va.is_empty() || vb.is_empty() {
                // Model returned no vector — fall back gracefully
                jaccard(&tokenize(a), &tokenize(b))
            } else {
                // Cosine similarity ∈ [-1, 1]; clamp to [0, 1] — negative cosine
                // means opposite directions which we treat as zero similarity.
                cosine_similarity(&va, &vb).max(0.0)
            }
        }
        None => jaccard(&tokenize(a), &tokenize(b)),
    }
}

/// Concrete `EmbeddingModel` backed by fastembed-rs.
///
/// Model weights are downloaded on first construction and cached to `~/.cache/fastembed/`.
/// Construction embeds a single warmup string to force ONNX model loading before the first
/// calibration request — callers pay zero cold-start latency at task time.
/// All returned vectors are L2-normalised (cosine similarity = dot product).
///
/// Requires the `fastembed-embed` Cargo feature and ONNX Runtime on the host.
/// Build with: `cargo build -p h2ai-context --features fastembed-embed`
#[cfg(feature = "fastembed-embed")]
pub struct FastEmbedModel {
    inner: Mutex<fastembed::TextEmbedding>,
}

#[cfg(feature = "fastembed-embed")]
impl FastEmbedModel {
    /// Construct with the default model (`all-MiniLM-L6-v2`, 22 MB).
    /// Prefer `new_with` when the operator has configured a model in `H2AIConfig`.
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::new_with_fastembed_model(fastembed::EmbeddingModel::AllMiniLML6V2)
    }

    /// Construct with the model specified in `H2AIConfig::embedding_model_name`.
    pub fn new_with(
        name: &h2ai_config::EmbeddingModelName,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        use h2ai_config::EmbeddingModelName;
        let fastembed_model = match name {
            EmbeddingModelName::AllMiniLmL6V2 => fastembed::EmbeddingModel::AllMiniLML6V2,
            EmbeddingModelName::BgeSmallEnV1_5 => fastembed::EmbeddingModel::BGESmallENV15,
        };
        Self::new_with_fastembed_model(fastembed_model)
    }

    fn new_with_fastembed_model(
        model: fastembed::EmbeddingModel,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let opts = fastembed::InitOptions {
            model_name: model,
            show_download_progress: true,
            ..Default::default()
        };
        let inner = fastembed::TextEmbedding::try_new(opts)?;
        let instance = Self {
            inner: Mutex::new(inner),
        };
        // Warmup: force ONNX model load now so the first calibration request pays no cold-start cost.
        let _ = instance.embed("warmup");
        Ok(instance)
    }
}

#[cfg(feature = "fastembed-embed")]
impl EmbeddingModel for FastEmbedModel {
    fn embed(&self, text: &str) -> Vec<f32> {
        let model = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match model.embed(vec![text], None) {
            Ok(mut embeddings) => {
                if embeddings.is_empty() {
                    return vec![];
                }
                l2_normalize(embeddings.remove(0))
            }
            Err(_) => vec![],
        }
    }
}

#[cfg(feature = "fastembed-embed")]
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Stub model for testing ────────────────────────────────────────────────

    /// Returns a fixed L2-normalised vector per text, keyed on content.
    struct StubEmbeddingModel;

    impl EmbeddingModel for StubEmbeddingModel {
        fn embed(&self, text: &str) -> Vec<f32> {
            // Two semantic clusters: "stateless auth" cluster and "redis cache" cluster.
            // Texts within a cluster get identical normalised vectors → cosine = 1.0.
            // Cross-cluster texts get orthogonal vectors → cosine = 0.0.
            if text.contains("jwt") || text.contains("auth") || text.contains("token") {
                vec![1.0, 0.0] // auth cluster
            } else if text.contains("redis") || text.contains("cache") || text.contains("key-value")
            {
                vec![0.0, 1.0] // redis cluster
            } else {
                vec![0.0, 0.0] // unknown → triggers fallback path (zero norm)
            }
        }
    }

    struct EmptyEmbeddingModel;

    impl EmbeddingModel for EmptyEmbeddingModel {
        fn embed(&self, _text: &str) -> Vec<f32> {
            vec![]
        }
    }

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn cosine_identical_unit_vectors_is_one() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors_clamped_to_negative() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        // raw cosine = -1.0; not clamped here (caller decides)
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_mismatched_lengths_returns_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_empty_vectors_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_zero_norm_vector_returns_zero() {
        let z = vec![0.0f32, 0.0];
        let v = vec![1.0f32, 0.0];
        assert_eq!(cosine_similarity(&z, &v), 0.0);
    }

    // ── semantic_jaccard — None (token fallback) ─────────────────────────────

    #[test]
    fn semantic_jaccard_none_identical_text_is_one() {
        let text = "stateless jwt auth token ADR-001";
        assert!((semantic_jaccard(text, text, None) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn semantic_jaccard_none_disjoint_text_is_zero() {
        let a = "stateless jwt auth token";
        let b = "redis cache key-value store";
        assert_eq!(semantic_jaccard(a, b, None), 0.0);
    }

    #[test]
    fn semantic_jaccard_none_partial_overlap_between_zero_and_one() {
        let a = "jwt stateless token authentication";
        let b = "jwt bearer authentication mechanism";
        let j = semantic_jaccard(a, b, None);
        assert!(
            j > 0.0 && j < 1.0,
            "partial overlap must be in (0,1), got {j}"
        );
    }

    // ── semantic_jaccard — with model ─────────────────────────────────────────

    #[test]
    fn semantic_jaccard_model_same_cluster_is_one() {
        // "jwt" and "auth token" are both in the auth cluster → cosine = 1.0
        let model = StubEmbeddingModel;
        let sim = semantic_jaccard("jwt access token", "auth bearer token", Some(&model));
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "same semantic cluster must score 1.0, got {sim}"
        );
    }

    #[test]
    fn semantic_jaccard_model_different_cluster_is_zero() {
        // auth cluster vs redis cluster → cosine = 0.0
        let model = StubEmbeddingModel;
        let sim = semantic_jaccard("jwt auth token", "redis cache key-value", Some(&model));
        assert!(
            sim.abs() < 1e-6,
            "different clusters must score 0.0, got {sim}"
        );
    }

    #[test]
    fn semantic_jaccard_model_synonyms_score_same_as_literal_match() {
        // Without model, "key-value store" ≠ "redis" (zero Jaccard overlap).
        // With model they're in the same cluster and score 1.0.
        let model = StubEmbeddingModel;
        let with_model = semantic_jaccard("redis cache", "key-value store", Some(&model));
        let without_model = semantic_jaccard("redis cache", "key-value store", None);
        assert!(
            with_model > without_model,
            "semantic model must close the synonym gap: with={with_model} without={without_model}"
        );
    }

    #[test]
    fn semantic_jaccard_model_empty_embed_falls_back_to_token_jaccard() {
        // Empty embedding model → falls back to token Jaccard
        let model = EmptyEmbeddingModel;
        let text = "stateless jwt auth token";
        let sim_model = semantic_jaccard(text, text, Some(&model));
        let sim_token = semantic_jaccard(text, text, None);
        assert!(
            (sim_model - sim_token).abs() < 1e-9,
            "empty embed must fall back to token Jaccard: model={sim_model} token={sim_token}"
        );
    }

    #[test]
    fn semantic_jaccard_negative_cosine_clamped_to_zero() {
        // A model that returns anti-correlated embeddings should produce sim=0, not negative
        struct AntiModel;
        impl EmbeddingModel for AntiModel {
            fn embed(&self, text: &str) -> Vec<f32> {
                if text.starts_with('a') {
                    vec![1.0, 0.0]
                } else {
                    vec![-1.0, 0.0]
                }
            }
        }
        let model = AntiModel;
        let sim = semantic_jaccard("alpha", "beta", Some(&model));
        assert!(
            sim >= 0.0,
            "semantic_jaccard must be non-negative, got {sim}"
        );
    }

    // ── FastEmbedModel integration test (requires model download) ─────────────
    // Run with: cargo test -p h2ai-context --features fastembed-embed -- embedding_model_cosine_paraphrases --ignored
    #[cfg(feature = "fastembed-embed")]
    #[test]
    #[ignore = "requires fastembed model download (~90MB) and ORT"]
    fn embedding_model_cosine_paraphrases() {
        let model = super::FastEmbedModel::new().expect("FastEmbedModel::new");
        let a = model.embed("the payment budget is exhausted");
        let b = model.embed("the spending limit has been reached");
        let c = model.embed("the weather is sunny today");
        let sim_ab: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let sim_ac: f32 = a.iter().zip(&c).map(|(x, y)| x * y).sum();
        assert!(sim_ab > 0.80, "paraphrases should score high: {sim_ab}");
        assert!(sim_ac < 0.30, "off-topic should score low: {sim_ac}");
    }
}
