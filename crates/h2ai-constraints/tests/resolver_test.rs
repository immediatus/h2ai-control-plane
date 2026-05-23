use async_trait::async_trait;
use h2ai_constraints::index::ConstraintIndex;
use h2ai_constraints::resolver::ConstraintResolver;
use h2ai_constraints::source::ConstraintError;
use h2ai_constraints::store::ConstraintStore;
use h2ai_constraints::types::{
    CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
};
use std::sync::Arc;

// ── Minimal mock implementations ─────────────────────────────────────────────

struct MockIndex {
    docs: Vec<ConstraintDoc>,
}

#[async_trait]
impl ConstraintIndex for MockIndex {
    async fn find_by_ids(&self, ids: &[String]) -> Vec<String> {
        ids.iter()
            .filter(|id| self.docs.iter().any(|d| &d.id == *id))
            .cloned()
            .collect()
    }

    async fn find_by_tags(&self, tags: &[String]) -> Vec<String> {
        self.docs
            .iter()
            .filter(|d| {
                tags.iter()
                    .any(|t| d.domains.contains(t) || d.mandatory_for_tags.contains(t))
            })
            .map(|d| d.id.clone())
            .collect()
    }

    async fn search(&self, query: &str, top_k: usize) -> Vec<String> {
        // Simple keyword match for tests
        self.docs
            .iter()
            .filter(|d| d.description.contains(query) || d.id.contains(query))
            .take(top_k)
            .map(|d| d.id.clone())
            .collect()
    }
}

struct MockStore {
    docs: Vec<ConstraintDoc>,
}

#[async_trait]
impl ConstraintStore for MockStore {
    async fn load(&self, id: &str) -> Result<ConstraintDoc, ConstraintError> {
        self.docs
            .iter()
            .find(|d| d.id == id)
            .cloned()
            .ok_or_else(|| ConstraintError::NotFound(id.to_string()))
    }
}

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
    let index = Arc::new(MockIndex { docs: docs.clone() });
    let store = Arc::new(MockStore { docs });
    ConstraintResolver::new(index, store)
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
