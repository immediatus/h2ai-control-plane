use h2ai_constraints::source::{ConstraintSource, FsConstraintSource};
use h2ai_constraints::types::{ConstraintMeta, ConstraintSeverity, PredicateKind};
use h2ai_constraints::wiki::WikiCache;
use std::fs;
use tempfile::TempDir;

#[test]
fn nats_wiki_source_resolve_uses_wiki_cache() {
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
async fn reconstruct_docs_from_static_metas() {
    use h2ai_api::constraint_source::reconstruct_docs;

    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("ADR-001.yaml"),
        "id: ADR-001\ntitle: Cite Source\nseverity: hard\ncriteria:\n  pass: Cites a source reference\n  fail: No source cited\n",
    )
    .unwrap();
    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source
        .resolve_context(&[], &["ADR-001".to_string()], "")
        .await;

    let docs = reconstruct_docs(metas, &source).await;
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].id, "ADR-001");
}
