use h2ai_knowledge::factory::{
    KnowledgeConfig, KnowledgeProviderFactory, ProviderKind, SourceKind,
};
use h2ai_knowledge::provider::{Bm25WikiProvider, KnowledgeProvider};
use h2ai_knowledge::source::YamlDirSource;
use h2ai_knowledge::types::{KnowledgeQuery, NodeDepth, RetrievalMode, SearchScope};
use std::path::PathBuf;
use std::sync::Arc;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/constraints")
}

#[tokio::test]
async fn provider_is_ready_after_build() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    assert!(provider.is_ready());
    assert_eq!(
        provider.leaf_count(),
        3,
        "should have 3 leaf constraints loaded"
    );
    assert_eq!(provider.topic_count(), 1, "should have 1 wiki topic node");
}

#[tokio::test]
async fn provider_has_global_node_after_build() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    let global = provider.global_summary().await;
    assert!(global.is_some());
    assert_eq!(global.unwrap().depth, NodeDepth::Global);
}

#[tokio::test]
async fn provider_fallback_no_wiki_dir() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("C-001.yaml"),
        r#"id: C-001
title: "Test constraint"
severity: hard
domains:
  - test_domain
criteria:
  pass: "Test passes."
  fail: "Test fails."
"#,
    )
    .unwrap();

    let source = Arc::new(YamlDirSource::new(tmp.path()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    assert!(
        provider.is_ready(),
        "provider must be ready even without wiki/"
    );
    let global = provider.global_summary().await;
    assert!(global.is_some(), "synthetic global node must be created");
    // Verify the global node is synthetic (built from topic summaries, not from _overview.yaml)
    use h2ai_knowledge::types::NodeSource;
    assert!(
        matches!(global.unwrap().source, NodeSource::Synthetic),
        "global node source must be Synthetic when no _overview.yaml exists"
    );
}

#[tokio::test]
async fn provider_empty_source_does_not_panic() {
    let tmp = tempfile::tempdir().unwrap();
    let source = Arc::new(YamlDirSource::new(tmp.path()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    assert!(provider.is_ready());
    let global = provider.global_summary().await;
    // Empty corpus: no leaves → no topic syntheses → no global node
    assert!(
        global.is_none(),
        "empty corpus should produce no global node"
    );
}

#[tokio::test]
async fn provider_global_included_when_depth_requested() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    let q = KnowledgeQuery::all_depths("atomic debit idempotency");
    let result = provider.query(&q).await;
    assert!(
        result.global_included,
        "global node must be included when depths contains Global"
    );
    assert!(
        result
            .nodes
            .iter()
            .any(|(n, _)| n.depth == NodeDepth::Global),
        "global node must appear in results"
    );
}

#[tokio::test]
async fn provider_topic_nodes_in_results() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    let q = KnowledgeQuery::all_depths("budget atomicity idempotency financial");
    let result = provider.query(&q).await;
    let has_topic = result
        .nodes
        .iter()
        .any(|(n, _)| n.depth == NodeDepth::Topic);
    assert!(
        has_topic,
        "query matching financial-systems topic must return topic node"
    );
}

#[tokio::test]
async fn provider_explicit_ids_bypass_bm25() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    let explicit = vec!["C-004".to_string(), "C-008".to_string()];
    let q = KnowledgeQuery {
        text: "unrelated query text that won't match",
        tags: &[],
        explicit_ids: &explicit,
        top_k: 10,
        depths: &[NodeDepth::Leaf],
        mode: RetrievalMode::TreeTraversal,
        scope: SearchScope::Auto,
        expand_hops: 0,
    };
    let result = provider.query(&q).await;
    let ids: Vec<&str> = result.nodes.iter().map(|(n, _)| n.id.as_str()).collect();
    assert!(ids.contains(&"C-004"), "explicit C-004 must be in results");
    assert!(ids.contains(&"C-008"), "explicit C-008 must be in results");
}

#[tokio::test]
async fn provider_explicit_ids_includes_topic_entry_points() {
    // When constraint IDs are given as explicit_ids, wiki topic nodes whose
    // entry_points list those IDs must also be returned (vocabulary-gap bypass).
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    let explicit = vec!["C-004".to_string()];
    let q = KnowledgeQuery {
        text: "unrelated query text that won't match",
        tags: &[],
        explicit_ids: &explicit,
        top_k: 10,
        depths: &[NodeDepth::Topic, NodeDepth::Leaf],
        mode: RetrievalMode::TreeTraversal,
        scope: SearchScope::Auto,
        expand_hops: 0,
    };
    let result = provider.query(&q).await;
    let ids: Vec<&str> = result.nodes.iter().map(|(n, _)| n.id.as_str()).collect();
    assert!(
        ids.contains(&"C-004"),
        "explicit C-004 leaf must be in results"
    );
    assert!(
        ids.contains(&"topic:financial-systems"),
        "topic node with entry_point C-004 must be included"
    );
}

#[tokio::test]
async fn provider_ppr_expands_related_leaves() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    let q = KnowledgeQuery {
        text: "C-004 idempotency atomic debit",
        tags: &[],
        explicit_ids: &[],
        top_k: 10,
        depths: &[NodeDepth::Leaf],
        mode: RetrievalMode::TreeTraversal,
        scope: SearchScope::Auto,
        expand_hops: 1,
    };
    let result = provider.query(&q).await;
    let ids: Vec<&str> = result.nodes.iter().map(|(n, _)| n.id.as_str()).collect();
    assert!(ids.contains(&"C-004"), "direct BM25 hit must be present");
    let ppr_added = ids.contains(&"C-005") || ids.contains(&"C-008");
    assert!(ppr_added, "PPR must expand at least one neighbour of C-004");
}

#[tokio::test]
async fn provider_ppr_expansion_flag() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    let q = KnowledgeQuery {
        text: "C-004 idempotency",
        tags: &[],
        explicit_ids: &[],
        top_k: 10,
        depths: &[NodeDepth::Leaf],
        mode: RetrievalMode::TreeTraversal,
        scope: SearchScope::Auto,
        expand_hops: 1,
    };
    let result = provider.query(&q).await;
    assert!(
        result.ppr_expanded,
        "expand_hops=1 must set ppr_expanded=true when graph has edges"
    );
}

#[tokio::test]
async fn provider_tension_surfaced() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, Default::default()).await;
    let q = KnowledgeQuery {
        text: "atomic debit partition tolerance distributed lock financial",
        tags: &[],
        explicit_ids: &[],
        top_k: 10,
        depths: &[NodeDepth::Topic, NodeDepth::Leaf],
        mode: RetrievalMode::CollapsedTree,
        scope: SearchScope::Auto,
        expand_hops: 0,
    };
    let result = provider.query(&q).await;
    assert!(!result.nodes.is_empty(), "query must return nodes");
}

#[tokio::test]
async fn passthrough_provider_is_ready() {
    use h2ai_knowledge::provider::PassthroughProvider;
    let tmp = tempfile::tempdir().unwrap();
    let provider = PassthroughProvider::new_from_path(tmp.path());
    assert!(provider.is_ready());
}

#[tokio::test]
async fn factory_builds_bm25wiki_provider() {
    let cfg = KnowledgeConfig {
        provider: ProviderKind::Bm25Wiki,
        source: SourceKind::YamlDir {
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/constraints"),
        },
        scoring: Default::default(),
    };
    let provider = KnowledgeProviderFactory::build_provider(&cfg).await;
    assert!(provider.is_ready());
    assert_eq!(*provider.kind(), ProviderKind::Bm25Wiki);
}
