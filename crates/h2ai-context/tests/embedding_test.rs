use h2ai_context::embedding::{cosine_similarity, semantic_jaccard, EmbeddingModel};

// ── Mock declarations ─────────────────────────────────────────────────────────

mockall::mock! {
    pub StubEmbeddingModel {}
    impl EmbeddingModel for StubEmbeddingModel {
        fn embed(&self, text: &str) -> Vec<f32>;
    }
}

// ── Stub factories ────────────────────────────────────────────────────────────

/// Returns a fixed L2-normalised vector per text, keyed on content.
/// Two semantic clusters: "stateless auth" cluster and "redis cache" cluster.
fn stub_embedding_model() -> MockStubEmbeddingModel {
    let mut m = MockStubEmbeddingModel::new();
    m.expect_embed().returning(|text| {
        if text.contains("jwt") || text.contains("auth") || text.contains("token") {
            vec![1.0, 0.0] // auth cluster
        } else if text.contains("redis") || text.contains("cache") || text.contains("key-value") {
            vec![0.0, 1.0] // redis cluster
        } else {
            vec![0.0, 0.0] // unknown → triggers fallback path (zero norm)
        }
    });
    m
}

fn empty_embedding_model() -> MockStubEmbeddingModel {
    let mut m = MockStubEmbeddingModel::new();
    m.expect_embed().returning(|_| vec![]);
    m
}

// ── cosine_similarity ─────────────────────────────────────────────────────────

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
#[allow(clippy::float_cmp)]
fn cosine_mismatched_lengths_returns_zero() {
    let a = vec![1.0f32, 0.0];
    let b = vec![1.0f32, 0.0, 0.0];
    assert_eq!(cosine_similarity(&a, &b), 0.0);
}

#[test]
#[allow(clippy::float_cmp)]
fn cosine_empty_vectors_returns_zero() {
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
}

#[test]
#[allow(clippy::float_cmp)]
fn cosine_zero_norm_vector_returns_zero() {
    let z = vec![0.0f32, 0.0];
    let v = vec![1.0f32, 0.0];
    assert_eq!(cosine_similarity(&z, &v), 0.0);
}

// ── semantic_jaccard — None (token fallback) ─────────────────────────────────

#[test]
fn semantic_jaccard_none_identical_text_is_one() {
    let text = "stateless jwt auth token ADR-001";
    assert!((semantic_jaccard(text, text, None) - 1.0).abs() < 1e-9);
}

#[test]
#[allow(clippy::float_cmp)]
fn semantic_jaccard_none_disjoint_text_is_zero() {
    let a = "stateless jwt auth token";
    let b = "redis cache key-value store";
    assert_eq!(semantic_jaccard(a, b, None), 0.0);
}

#[test]
#[allow(clippy::float_cmp)]
fn semantic_jaccard_none_different_text_is_zero() {
    // Without embedding model, fallback is exact equality — partial overlap gives 0.0.
    let a = "jwt stateless token authentication";
    let b = "jwt bearer authentication mechanism";
    let j = semantic_jaccard(a, b, None);
    assert_eq!(j, 0.0, "non-identical strings without model must give 0.0");
}

// ── semantic_jaccard — with model ─────────────────────────────────────────────

#[test]
fn semantic_jaccard_model_same_cluster_is_one() {
    // "jwt" and "auth token" are both in the auth cluster → cosine = 1.0
    let model = stub_embedding_model();
    let sim = semantic_jaccard("jwt access token", "auth bearer token", Some(&model));
    assert!(
        (sim - 1.0).abs() < 1e-6,
        "same semantic cluster must score 1.0, got {sim}"
    );
}

#[test]
fn semantic_jaccard_model_different_cluster_is_zero() {
    // auth cluster vs redis cluster → cosine = 0.0
    let model = stub_embedding_model();
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
    let model = stub_embedding_model();
    let with_model = semantic_jaccard("redis cache", "key-value store", Some(&model));
    let without_model = semantic_jaccard("redis cache", "key-value store", None);
    assert!(
        with_model > without_model,
        "semantic model must close the synonym gap: with={with_model} without={without_model}"
    );
}

#[test]
#[allow(clippy::float_cmp)]
fn semantic_jaccard_model_empty_embed_falls_back_to_exact_equality() {
    // Empty embedding model → falls back to exact equality
    let model = empty_embedding_model();
    let text = "stateless jwt auth token";
    let sim = semantic_jaccard(text, text, Some(&model));
    assert!(
        (sim - 1.0).abs() < 1e-9,
        "empty embed on identical text must give 1.0"
    );
    let sim_diff = semantic_jaccard(text, "different text", Some(&model));
    assert_eq!(sim_diff, 0.0, "empty embed on different text must give 0.0");
}

#[test]
fn semantic_jaccard_negative_cosine_clamped_to_zero() {
    // A model that returns anti-correlated embeddings should produce sim=0, not negative
    mockall::mock! {
        pub AntiModel {}
        impl EmbeddingModel for AntiModel {
            fn embed(&self, text: &str) -> Vec<f32>;
        }
    }
    let mut model = MockAntiModel::new();
    model.expect_embed().returning(|text| {
        if text.starts_with('a') {
            vec![1.0, 0.0]
        } else {
            vec![-1.0, 0.0]
        }
    });
    let sim = semantic_jaccard("alpha", "beta", Some(&model));
    assert!(
        sim >= 0.0,
        "semantic_jaccard must be non-negative, got {sim}"
    );
}
