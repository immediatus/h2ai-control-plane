use h2ai_constraints::index::ConstraintIndex;
use h2ai_constraints::resolver::ConstraintResolver;
use h2ai_constraints::source::ConstraintError;
use h2ai_constraints::store::ConstraintStore;
use h2ai_constraints::types::{
    CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
};
use std::sync::Arc;

// ── Mock declarations ─────────────────────────────────────────────────────────

mockall::mock! {
    pub Index {}
    #[async_trait::async_trait]
    impl ConstraintIndex for Index {
        async fn find_by_ids(&self, ids: &[String]) -> Vec<String>;
        async fn find_by_tags(&self, tags: &[String]) -> Vec<String>;
        async fn search(&self, query: &str, top_k: usize) -> Vec<String>;
    }
}

mockall::mock! {
    pub Store {}
    #[async_trait::async_trait]
    impl ConstraintStore for Store {
        async fn load(&self, id: &str) -> Result<ConstraintDoc, ConstraintError>;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_doc(id: &str, domains: &[&str], tags: &[&str]) -> ConstraintDoc {
    ConstraintDoc {
        id: id.to_string(),
        source_file: format!("{id}.yaml"),
        description: format!("{id} description keyword"),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![ConstraintPredicate::LlmJudge {
                rubric: "test rubric".into(),
            }],
        },
        remediation_hint: None,
        domains: domains
            .iter()
            .map(std::string::ToString::to_string)
            .collect(),
        mandatory_for_tags: tags.iter().map(std::string::ToString::to_string).collect(),
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }
}

fn make_resolver(docs: Vec<ConstraintDoc>) -> ConstraintResolver {
    let docs_clone = docs.clone();
    let mut index = MockIndex::new();
    let mut store = MockStore::new();

    // find_by_ids: return ids that exist in docs
    {
        let docs_for_ids = docs.clone();
        index.expect_find_by_ids().returning(move |ids| {
            ids.iter()
                .filter(|id| docs_for_ids.iter().any(|d| &d.id == *id))
                .cloned()
                .collect()
        });
    }

    // find_by_tags: return ids of docs matching any tag
    {
        let docs_for_tags = docs.clone();
        index.expect_find_by_tags().returning(move |tags| {
            docs_for_tags
                .iter()
                .filter(|d| {
                    tags.iter()
                        .any(|t| d.domains.contains(t) || d.mandatory_for_tags.contains(t))
                })
                .map(|d| d.id.clone())
                .collect()
        });
    }

    // search: simple keyword match
    {
        let docs_for_search = docs.clone();
        index.expect_search().returning(move |query, top_k| {
            docs_for_search
                .iter()
                .filter(|d| d.description.contains(query) || d.id.contains(query))
                .take(top_k)
                .map(|d| d.id.clone())
                .collect()
        });
    }

    // load: return doc by id
    store.expect_load().returning(move |id| {
        docs_clone
            .iter()
            .find(|d| d.id == id)
            .cloned()
            .ok_or_else(|| ConstraintError::NotFound(id.to_string()))
    });

    ConstraintResolver::new(Arc::new(index), Arc::new(store))
}

// ── Explicit IDs path ─────────────────────────────────────────────────────────

#[tokio::test]
async fn resolve_explicit_ids_returns_matching_docs() {
    let docs = vec![
        make_doc("C-001", &["billing"], &[]),
        make_doc("C-002", &["audit"], &[]),
    ];
    let resolver = make_resolver(docs);

    let result = resolver.resolve(&["C-001".to_string()], &[], "").await;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id, "C-001");
}

// ── Tags + query path (union) ─────────────────────────────────────────────────

#[tokio::test]
async fn resolve_tags_and_query_returns_union() {
    let docs = vec![
        make_doc("C-TAG", &["billing"], &[]),
        make_doc("C-QUERY-keyword", &["other"], &[]),
    ];
    let resolver = make_resolver(docs);

    let result = resolver
        .resolve(&[], &["billing".to_string()], "keyword")
        .await;
    let ids: Vec<&str> = result.iter().map(|d| d.id.as_str()).collect();
    assert!(ids.contains(&"C-TAG"), "tag match must be included");
    assert!(
        ids.contains(&"C-QUERY-keyword"),
        "search match must be included"
    );
}

// ── Tags only path ────────────────────────────────────────────────────────────

#[tokio::test]
async fn resolve_tags_only_no_bm25() {
    let docs = vec![
        make_doc("C-BILLED", &["billing"], &[]),
        make_doc("C-OTHER", &["other"], &[]),
    ];
    let resolver = make_resolver(docs);

    let result = resolver.resolve(&[], &["billing".to_string()], "").await;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id, "C-BILLED");
}

// ── Lines 55-56: query-only path (no explicit IDs, no tags) ──────────────────

#[tokio::test]
async fn resolve_query_only_path() {
    let docs = vec![
        make_doc("C-KWMATCH", &[], &[]),
        make_doc("C-OTHER", &[], &[]),
    ];
    let resolver = make_resolver(docs);

    // No explicit_ids, no tags, only query — exercises `else if !query.is_empty()` branch
    let result = resolver.resolve(&[], &[], "C-KWMATCH").await;
    assert!(result.iter().any(|d| d.id == "C-KWMATCH"));
}

// ── Line 58: all-empty path → empty result ────────────────────────────────────

#[tokio::test]
async fn resolve_all_empty_returns_empty() {
    let docs = vec![make_doc("C-001", &["billing"], &[])];
    let resolver = make_resolver(docs);

    // All empty → `else { vec![] }` branch
    let result = resolver.resolve(&[], &[], "").await;
    assert!(result.is_empty(), "all-empty inputs must return empty");
}

// ── Line 62: ids.is_empty() guard after resolution ───────────────────────────

#[tokio::test]
async fn resolve_explicit_ids_not_in_index_returns_empty() {
    let docs = vec![make_doc("C-EXISTS", &[], &[])];
    let resolver = make_resolver(docs);

    // Explicit id that doesn't exist in index → find_by_ids returns [] → ids empty → vec![]
    let result = resolver
        .resolve(&["C-NONEXISTENT".to_string()], &[], "")
        .await;
    assert!(
        result.is_empty(),
        "explicit id not in index must return empty"
    );
}
