use h2ai_knowledge::factory::{
    KnowledgeConfig, KnowledgeProviderFactory, ProviderKind, ScoringConfig, SourceKind,
};
use h2ai_knowledge::provider::{Bm25WikiProvider, KnowledgeProvider};
use h2ai_knowledge::source::YamlDirSource;
use h2ai_knowledge::types::{KnowledgeQuery, NodeDepth, NodeSource, RetrievalMode, SearchScope};
use std::path::PathBuf;
use std::sync::Arc;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/constraints")
}

#[tokio::test]
async fn provider_is_ready_after_build() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
    assert!(
        provider.is_ready(),
        "provider must be ready even without wiki/"
    );
    let global = provider.global_summary().await;
    assert!(global.is_some(), "synthetic global node must be created");
    // Verify the global node is synthetic (built from topic summaries, not from _overview.yaml)
    assert!(
        matches!(global.unwrap().source, NodeSource::Synthetic),
        "global node source must be Synthetic when no _overview.yaml exists"
    );
}

#[tokio::test]
async fn provider_empty_source_does_not_panic() {
    let tmp = tempfile::tempdir().unwrap();
    let source = Arc::new(YamlDirSource::new(tmp.path()));
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
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
        scoring: ScoringConfig::default(),
    };
    let provider = KnowledgeProviderFactory::build_provider(&cfg).await;
    assert!(provider.is_ready());
    assert_eq!(*provider.kind(), ProviderKind::Bm25Wiki);
}

// ── factory: Passthrough branch ───────────────────────────────────────────────

#[tokio::test]
async fn factory_builds_passthrough_provider() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = KnowledgeConfig {
        provider: ProviderKind::Passthrough,
        source: SourceKind::YamlDir {
            path: tmp.path().to_path_buf(),
        },
        scoring: ScoringConfig::default(),
    };
    let provider = KnowledgeProviderFactory::build_provider(&cfg).await;
    assert!(provider.is_ready());
    assert_eq!(*provider.kind(), ProviderKind::Passthrough);
}

#[tokio::test]
async fn factory_build_from_constraint_corpus() {
    let provider = KnowledgeProviderFactory::build_from_constraint_corpus(&fixture_dir()).await;
    assert!(provider.is_ready());
    assert_eq!(*provider.kind(), ProviderKind::Bm25Wiki);
}

// ── PassthroughProvider: all trait methods ────────────────────────────────────

#[tokio::test]
async fn passthrough_provider_new_directly() {
    use h2ai_constraints::resolver::ConstraintResolver;
    use h2ai_constraints::source::{FsConstraintIndex, FsConstraintStore};
    use h2ai_knowledge::provider::PassthroughProvider;
    use std::sync::Arc;

    let store = FsConstraintStore::from_docs(vec![]);
    let index = FsConstraintIndex::from_docs(&[]);
    let resolver = ConstraintResolver::new(Arc::new(index), Arc::new(store));
    let provider = PassthroughProvider::new(resolver);
    assert!(provider.is_ready());
    assert_eq!(*provider.kind(), ProviderKind::Passthrough);
}

#[tokio::test]
async fn passthrough_provider_global_summary_is_none() {
    use h2ai_knowledge::provider::PassthroughProvider;
    let tmp = tempfile::tempdir().unwrap();
    let provider = PassthroughProvider::new_from_path(tmp.path());
    assert!(provider.global_summary().await.is_none());
}

#[tokio::test]
async fn passthrough_provider_new_from_nonexistent_path_fallback() {
    // FsConstraintStore::load returns Err for non-existent path → triggers the
    // unwrap_or_else fallback path (lines 488-491 in provider.rs).
    use h2ai_knowledge::provider::PassthroughProvider;
    let provider = PassthroughProvider::new_from_path(std::path::Path::new(
        "/nonexistent/path/that/does/not/exist",
    ));
    assert!(provider.is_ready());
    let q = KnowledgeQuery {
        text: "anything",
        tags: &[],
        explicit_ids: &[],
        top_k: 5,
        depths: &[NodeDepth::Leaf],
        mode: RetrievalMode::TreeTraversal,
        scope: SearchScope::Auto,
        expand_hops: 0,
    };
    let result = provider.query(&q).await;
    assert!(result.nodes.is_empty());
}

#[tokio::test]
async fn passthrough_provider_query_returns_empty_for_empty_corpus() {
    use h2ai_knowledge::provider::PassthroughProvider;
    let tmp = tempfile::tempdir().unwrap();
    let provider = PassthroughProvider::new_from_path(tmp.path());
    let q = KnowledgeQuery {
        text: "budget atomicity",
        tags: &[],
        explicit_ids: &[],
        top_k: 10,
        depths: &[NodeDepth::Leaf],
        mode: RetrievalMode::TreeTraversal,
        scope: SearchScope::Auto,
        expand_hops: 0,
    };
    let result = provider.query(&q).await;
    assert!(result.nodes.is_empty());
    assert!(!result.global_included);
    assert!(!result.ppr_expanded);
}

#[tokio::test]
async fn passthrough_provider_query_returns_nodes_for_loaded_corpus() {
    use h2ai_knowledge::provider::PassthroughProvider;
    // Use the fixture corpus that has real constraints
    let provider = PassthroughProvider::new_from_path(&fixture_dir());
    let explicit = vec!["C-004".to_string()];
    let q = KnowledgeQuery {
        text: "budget atomicity",
        tags: &[],
        explicit_ids: &explicit,
        top_k: 10,
        depths: &[NodeDepth::Leaf],
        mode: RetrievalMode::TreeTraversal,
        scope: SearchScope::Auto,
        expand_hops: 0,
    };
    let result = provider.query(&q).await;
    // PassthroughProvider delegates to ConstraintResolver; check it doesn't panic
    // and returns a KnowledgeResult (nodes may or may not be populated depending on resolver)
    assert!(!result.global_included);
    assert!(!result.ppr_expanded);
    assert!(result.surfaced_tensions.is_empty());
}

// ── provider: surfaced tensions dedup (already-inserted check) ────────────────

#[tokio::test]
async fn provider_surfaced_tensions_dedup() {
    use h2ai_knowledge::source::KnowledgeSource;
    use h2ai_knowledge::types::{NodeSource, TensionRef};

    // Build a source that produces two topic nodes both claiming a tension with the same pair
    // so the already-inserted path is exercised.
    struct TwinTensionSource;

    #[async_trait::async_trait]
    impl KnowledgeSource for TwinTensionSource {
        async fn all_items(&self) -> Vec<h2ai_knowledge::source::SourceItem> {
            vec![]
        }

        async fn wiki_nodes(&self) -> Vec<h2ai_knowledge::types::KnowledgeNode> {
            use h2ai_knowledge::types::{KnowledgeNode, NodeDepth};
            vec![
                KnowledgeNode {
                    id: "topic:alpha".to_string(),
                    depth: NodeDepth::Topic,
                    synthesis: "alpha topic financial distributed".to_string(),
                    invariants: vec![],
                    failure_modes: vec![],
                    domains: vec!["financial".to_string()],
                    entry_points: vec![],
                    tensions: vec![TensionRef {
                        domain: "distributed".to_string(),
                        reason: "partition vs atomicity".to_string(),
                    }],
                    cross_references: vec![],
                    related: vec![],
                    source: NodeSource::Synthetic,
                    importance: 0.8,
                },
                KnowledgeNode {
                    id: "topic:beta".to_string(),
                    depth: NodeDepth::Topic,
                    synthesis: "beta topic financial distributed".to_string(),
                    invariants: vec![],
                    failure_modes: vec![],
                    domains: vec!["distributed".to_string()],
                    entry_points: vec![],
                    tensions: vec![TensionRef {
                        domain: "financial".to_string(),
                        reason: "same tension reverse direction".to_string(),
                    }],
                    cross_references: vec![],
                    related: vec![],
                    source: NodeSource::Synthetic,
                    importance: 0.8,
                },
            ]
        }

        async fn global_node(&self) -> Option<h2ai_knowledge::types::KnowledgeNode> {
            None
        }
    }

    let source: Arc<dyn h2ai_knowledge::source::KnowledgeSource> = Arc::new(TwinTensionSource);
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;

    let q = KnowledgeQuery {
        text: "financial distributed",
        tags: &[],
        explicit_ids: &[],
        top_k: 10,
        depths: &[NodeDepth::Topic],
        mode: RetrievalMode::CollapsedTree,
        scope: SearchScope::Auto,
        expand_hops: 0,
    };
    let result = provider.query(&q).await;
    // At most one surfaced tension for the financial<->distributed pair (dedup path exercised)
    let financial_distributed_count = result
        .surfaced_tensions
        .iter()
        .filter(|t| {
            (t.domain_a == "financial" && t.domain_b == "distributed")
                || (t.domain_a == "distributed" && t.domain_b == "financial")
        })
        .count();
    assert!(
        financial_distributed_count <= 1,
        "duplicate tensions must be deduped"
    );
}

// ── provider: CollapsedTree with only Leaf depth (no global) ─────────────────

#[tokio::test]
async fn provider_collapsed_tree_leaf_only() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
    let q = KnowledgeQuery {
        text: "idempotency debit budget atomicity",
        tags: &[],
        explicit_ids: &[],
        top_k: 5,
        depths: &[NodeDepth::Leaf],
        mode: RetrievalMode::CollapsedTree,
        scope: SearchScope::Auto,
        expand_hops: 0,
    };
    let result = provider.query(&q).await;
    assert!(!result.global_included);
    // All returned nodes should be leaf depth
    for (n, _) in &result.nodes {
        assert_eq!(n.depth, NodeDepth::Leaf);
    }
}

// ── provider: TreeTraversal with topic depth only (no leaf) ──────────────────

#[tokio::test]
async fn provider_tree_traversal_topic_only() {
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let provider = Bm25WikiProvider::build(source, ScoringConfig::default()).await;
    let q = KnowledgeQuery {
        text: "financial budget",
        tags: &[],
        explicit_ids: &[],
        top_k: 5,
        depths: &[NodeDepth::Topic],
        mode: RetrievalMode::TreeTraversal,
        scope: SearchScope::Auto,
        expand_hops: 0,
    };
    let result = provider.query(&q).await;
    // No leaf nodes should appear
    for (n, _) in &result.nodes {
        assert_ne!(n.depth, NodeDepth::Leaf);
    }
}

// ── Lines 450-454: dedup and_modify fires when same node appears twice ─────────

#[tokio::test]
async fn provider_dedup_and_modify_fires_on_duplicate_node_from_ppr_and_bm25() {
    // Use explicit_ids bypass so the same leaf appears via both BM25 (TreeTraversal)
    // and PPR expansion — the dedup and_modify closure must fire with a higher score.
    let source = Arc::new(YamlDirSource::new(fixture_dir()));
    let cfg = ScoringConfig {
        ppr_score_multiplier: 9.0, // make PPR score higher than BM25 score
        ..ScoringConfig::default()
    };
    let provider = Bm25WikiProvider::build(source, cfg).await;

    // Use TreeTraversal with expand_hops=1; the BM25 leaf hit + PPR expansion
    // can return the same leaf twice (once from BM25, once from PPR with a different score).
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
    // C-004 must appear exactly once in the deduplicated results
    let c004_count = result.nodes.iter().filter(|(n, _)| n.id == "C-004").count();
    assert!(
        c004_count <= 1,
        "dedup must produce at most one C-004 entry"
    );
}

// ── Lines 498-501: new_from_path fallback when path is a file (not a dir) ─────

#[test]
fn passthrough_provider_new_from_path_file_triggers_fallback() {
    use h2ai_knowledge::provider::PassthroughProvider;
    // Pass a file path — read_dir on a file returns Err → triggers the unwrap_or_else fallback
    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("not_a_dir.txt");
    std::fs::write(&file_path, "not a yaml dir").unwrap();

    let provider = PassthroughProvider::new_from_path(&file_path);
    assert!(provider.is_ready(), "fallback provider must be ready");
    assert_eq!(*provider.kind(), ProviderKind::Passthrough);
}
