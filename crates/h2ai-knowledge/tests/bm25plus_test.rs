use h2ai_knowledge::types::{
    KnowledgeNode, KnowledgeQuery, KnowledgeResult, NodeDepth, NodeSource, RetrievalMode,
    SearchScope,
};

#[test]
fn types_compile() {
    static ALL: &[NodeDepth] = &[NodeDepth::Global, NodeDepth::Topic, NodeDepth::Leaf];
    let node = KnowledgeNode {
        id: "C-004".into(),
        depth: NodeDepth::Leaf,
        synthesis: "atomic debit".into(),
        invariants: vec!["no double-spend".into()],
        failure_modes: vec!["FM-001".into()],
        domains: vec!["financial_systems".into()],
        entry_points: vec![],
        tensions: vec![],
        cross_references: vec![],
        related: vec!["C-005".into()],
        source: NodeSource::YamlConstraint { id: "C-004".into() },
        importance: 0.7,
    };
    assert_eq!(node.id, "C-004");
    let q = KnowledgeQuery {
        text: "atomic debit idempotency",
        tags: &[],
        explicit_ids: &[],
        top_k: 10,
        depths: ALL,
        mode: RetrievalMode::TreeTraversal,
        scope: SearchScope::Auto,
        expand_hops: 1,
    };
    assert_eq!(q.top_k, 10);

    let result = KnowledgeResult {
        nodes: vec![(node, 1.0)],
        global_included: false,
        surfaced_tensions: vec![],
        ppr_expanded: false,
    };
    assert_eq!(result.nodes.len(), 1);
}

use h2ai_knowledge::bm25plus::Bm25PlusRetriever;

#[test]
fn bm25plus_longer_doc_not_penalized() {
    let long_doc = "atomicity ".to_string() + &"filler ".repeat(299);
    let short_doc = "atomicity ".to_string() + &"filler ".repeat(49);
    let retriever = Bm25PlusRetriever::build(
        [("long", long_doc.as_str()), ("short", short_doc.as_str())].into_iter(),
    );
    let results = retriever.query("atomicity", 2);
    assert_eq!(results.len(), 2);
    assert!(
        results.iter().all(|c| c.score > 0.0),
        "all matching docs must score > 0"
    );
    // Short doc has lower length penalty so it outranks the long one
    assert!(
        results[0].score > results[1].score,
        "short doc must rank above long doc"
    );
}

#[test]
fn bm25plus_empty_corpus_returns_empty() {
    let retriever = Bm25PlusRetriever::build(std::iter::empty::<(&str, &str)>());
    assert!(retriever.query("any query", 10).is_empty());
}

#[test]
fn bm25plus_zero_topk_returns_empty() {
    let retriever = Bm25PlusRetriever::build([("doc1", "hello world")].into_iter());
    assert!(retriever.query("hello", 0).is_empty());
}

#[test]
fn bm25plus_no_matching_terms_returns_empty() {
    let retriever = Bm25PlusRetriever::build([("doc1", "hello world")].into_iter());
    let results = retriever.query("xyz_not_found", 10);
    assert!(results.is_empty());
}

#[test]
fn bm25plus_ranks_by_relevance() {
    let retriever = Bm25PlusRetriever::build(
        [
            ("high", "idempotency key atomic debit idempotency"),
            ("low", "audit trail logging"),
        ]
        .into_iter(),
    );
    let results = retriever.query("idempotency", 2);
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "high");
}

#[test]
fn bm25plus_len_and_is_empty() {
    let empty = Bm25PlusRetriever::build(std::iter::empty::<(&str, &str)>());
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());

    let non_empty = Bm25PlusRetriever::build([("d1", "hello world rust")].into_iter());
    assert_eq!(non_empty.len(), 1);
    assert!(!non_empty.is_empty());
}

#[test]
fn bm25plus_all_stopword_query_returns_empty() {
    // All tokens in the query are stopwords (len < 3 or in stopword list).
    // "the and for are" — all stopwords → tokenize returns empty map → early return.
    let retriever = Bm25PlusRetriever::build([("doc1", "atomicity idempotency")].into_iter());
    let results = retriever.query("the and for are", 10);
    assert!(
        results.is_empty(),
        "all-stopword query must return empty results"
    );
}

#[test]
fn bm25plus_query_term_in_idf_but_zero_tf_skipped() {
    // doc2 does NOT contain "idempotency" — tf=0 path inside bm25plus_score.
    // doc1 DOES contain it. Both must work without panic.
    let retriever = Bm25PlusRetriever::build(
        [
            ("doc1", "idempotency atomic debit"),
            ("doc2", "audit trail logging"),
        ]
        .into_iter(),
    );
    let results = retriever.query("idempotency", 2);
    // Only doc1 should match (doc2 has tf=0 for "idempotency")
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc1");
}
