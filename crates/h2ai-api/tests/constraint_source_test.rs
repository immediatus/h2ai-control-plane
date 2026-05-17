use h2ai_constraints::source::FsConstraintStore;
use h2ai_constraints::types::{ConstraintMeta, ConstraintSeverity, PredicateKind};
use h2ai_constraints::wiki::WikiCache;
use std::fs;
use tempfile::TempDir;

#[test]
fn wiki_cache_resolve_returns_matching_ids() {
    let mut cache = WikiCache::default();
    cache
        .context_map
        .insert("eu_data".into(), vec!["GDPR-001".into()]);
    cache.metas.insert(
        "GDPR-001".into(),
        ConstraintMeta {
            id: "GDPR-001".into(),
            summary: "Minimize personal data.".into(),
            severity: ConstraintSeverity::Hard { threshold: 0.8 },
            predicate_kind: PredicateKind::LlmJudge,
            domains: vec!["eu_data".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            payload_version: "v1".into(),
            inline_predicate: None,
            source: None,
            last_updated_ms: None,
        },
    );

    let resolved = cache.resolve(&["eu_data".to_string()], &[]);
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].id, "GDPR-001");
    assert_eq!(resolved[0].predicate_kind, PredicateKind::LlmJudge);
}

#[tokio::test]
async fn fs_constraint_resolver_loads_and_resolves_by_id() {
    use h2ai_constraints::resolver::ConstraintResolver;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("ADR-001.yaml"),
        "id: ADR-001\ntitle: Cite Source\nseverity: hard\ncriteria:\n  pass: Cites a source reference\n  fail: No source cited\n",
    )
    .unwrap();

    let (index, store) = FsConstraintStore::load(dir.path()).await.unwrap();
    let resolver = ConstraintResolver::new(Arc::new(index), Arc::new(store));
    let docs = resolver.resolve(&["ADR-001".to_string()], &[], "").await;

    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].id, "ADR-001");
}
