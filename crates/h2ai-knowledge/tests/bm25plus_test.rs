use h2ai_knowledge::types::{
    KnowledgeNode, KnowledgeQuery, KnowledgeResult, NodeDepth, NodeSource, RetrievalMode,
    SearchScope,
};

#[test]
fn types_compile() {
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

    static ALL: &[NodeDepth] = &[NodeDepth::Global, NodeDepth::Topic, NodeDepth::Leaf];
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
