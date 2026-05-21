use h2ai_context::fusion::{bm25_search, rrf_fuse, RRF_K};

// ── bm25_search: invalid query falls back to all-docs-zero-score ─────────────

#[test]
fn bm25_search_invalid_query_returns_all_docs_with_zero_score() {
    // A query with only special characters that tantivy cannot parse
    // triggers the Err(_) fallback path → all docs get score 0.0
    let docs = ["alpha doc", "beta doc", "gamma doc"];
    // Unbalanced quotes cause a parse error in tantivy
    let result = bm25_search("\"unclosed quote", &docs);
    assert_eq!(
        result.len(),
        3,
        "all docs must be returned on parse failure"
    );
    // All scores must be 0.0 (zero-score fill from the fallback path)
    for (_, score) in &result {
        assert!(
            (*score - 0.0).abs() < 1e-9,
            "fallback path must return score 0.0, got {score}"
        );
    }
}

#[test]
fn rrf_fuse_single_list_preserves_order() {
    let list = vec![(0usize, 0.9), (1, 0.7), (2, 0.3)];
    let fused = rrf_fuse(&[list], RRF_K);
    assert_eq!(fused[0].0, 0);
    assert_eq!(fused[1].0, 1);
    assert_eq!(fused[2].0, 2);
}

#[test]
fn rrf_fuse_two_agreeing_lists_amplifies_top_doc() {
    let list_a = vec![(0usize, 0.9), (1, 0.5)];
    let list_b = vec![(0usize, 0.8), (1, 0.4)];
    let fused = rrf_fuse(&[list_a, list_b], RRF_K);
    assert_eq!(fused[0].0, 0, "doc ranked 1st in both lists must win");
    assert!(fused[0].1 > fused[1].1);
}

#[test]
fn rrf_fuse_disagreeing_lists_give_equal_scores() {
    let list_a = vec![(0usize, 0.9), (1, 0.5)];
    let list_b = vec![(1usize, 0.9), (0, 0.5)];
    let fused = rrf_fuse(&[list_a, list_b], RRF_K);
    let score_0 = fused.iter().find(|(i, _)| *i == 0).unwrap().1;
    let score_1 = fused.iter().find(|(i, _)| *i == 1).unwrap().1;
    assert!(
        (score_0 - score_1).abs() < 1e-9,
        "mirrored ranks → equal RRF score"
    );
}

#[test]
fn rrf_fuse_empty_input_returns_empty() {
    let fused: Vec<(usize, f64)> = rrf_fuse(&[], RRF_K);
    assert!(fused.is_empty());
}

#[test]
fn bm25_search_relevant_doc_ranks_first() {
    let docs = [
        "jwt authentication stateless token bearer",
        "redis cache store eviction",
        "tcp socket connection timeout",
    ];
    let result = bm25_search("jwt authentication", &docs);
    assert_eq!(result[0].0, 0, "jwt doc must rank first for jwt query");
}

#[test]
fn bm25_search_empty_docs_returns_empty() {
    let result = bm25_search("query", &[]);
    assert!(result.is_empty());
}

#[test]
fn bm25_search_returns_all_docs() {
    let docs = ["alpha", "beta", "gamma"];
    let result = bm25_search("alpha beta", &docs);
    assert_eq!(result.len(), 3);
}
