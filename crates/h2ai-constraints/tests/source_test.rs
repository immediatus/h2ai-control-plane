use h2ai_constraints::source::{ConstraintSource, FsConstraintSource};
use h2ai_constraints::types::ConstraintPredicate;
use std::fs;
use tempfile::TempDir;

fn write_yaml(dir: &TempDir, name: &str, content: &str) {
    fs::write(dir.path().join(format!("{name}.yaml")), content).unwrap();
}

fn simple_yaml(id: &str, pass: &str) -> String {
    format!(
        "id: {id}\ntitle: {id}\nseverity: hard\ncriteria:\n  pass: {pass}\n  fail: Does not satisfy\n"
    )
}

#[tokio::test]
async fn fs_source_resolve_by_explicit_id() {
    let dir = TempDir::new().unwrap();
    write_yaml(
        &dir,
        "ADR-001",
        &simple_yaml("ADR-001", "Cites a source reference"),
    );

    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source
        .resolve_context(&[], &["ADR-001".to_string()], "")
        .await;
    assert_eq!(metas.len(), 1);
    assert_eq!(metas[0].id, "ADR-001");
}

#[tokio::test]
async fn fs_source_resolve_by_tag() {
    let dir = TempDir::new().unwrap();
    write_yaml(
        &dir,
        "GDPR-001",
        "id: GDPR-001\ntitle: Data Minimization\nseverity: hard\ndomains:\n  - eu_data\ncriteria:\n  pass: Minimizes personal data\n  fail: Over-collects\n",
    );

    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source
        .resolve_context(&["eu_data".to_string()], &[], "")
        .await;
    assert_eq!(metas.len(), 1);
    assert_eq!(metas[0].id, "GDPR-001");
}

#[tokio::test]
async fn fs_source_empty_tags_and_ids_returns_all_docs() {
    let dir = TempDir::new().unwrap();
    write_yaml(&dir, "ADR-001", &simple_yaml("ADR-001", "Rule one"));
    write_yaml(&dir, "ADR-002", &simple_yaml("ADR-002", "Rule two"));

    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source.resolve_context(&[], &[], "").await;
    assert_eq!(
        metas.len(),
        2,
        "all docs returned when no tags and no explicit ids"
    );
}

#[tokio::test]
async fn fs_source_tags_with_no_domain_metadata_falls_back_to_all_docs() {
    let dir = TempDir::new().unwrap();
    // File has no domains — context_map will be empty
    write_yaml(&dir, "ADR-001", &simple_yaml("ADR-001", "Rule one"));

    let source = FsConstraintSource::load(dir.path()).unwrap();
    // Tags provided but no domain metadata — must fall back to all docs
    let metas = source
        .resolve_context(&["eu_data".to_string()], &[], "")
        .await;
    assert_eq!(
        metas.len(),
        1,
        "falls back to all docs when context_map has no entry for tag"
    );
}

#[tokio::test]
async fn fs_source_tags_and_bm25_union() {
    // tag matches C-TAG; BM25 on "stateless service" matches C-SEM; both should be returned
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("C-TAG.yaml"),
        "id: C-TAG\ntitle: Budget Idempotency\nseverity: hard\ndomains:\n  - billing\ncriteria:\n  pass: Budget idempotency atomicity check\n  fail: No idempotency\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("C-SEM.yaml"),
        "id: C-SEM\ntitle: Stateless Service\nseverity: hard\ncriteria:\n  pass: Stateless service request handling without sticky session storage\n  fail: Uses sessions\n",
    )
    .unwrap();

    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source
        .resolve_context(
            &["billing".to_string()],
            &[],
            "stateless service request handling",
        )
        .await;
    let ids: std::collections::HashSet<&str> = metas.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains("C-TAG"),
        "tag-matched constraint must be present"
    );
    assert!(
        ids.contains("C-SEM"),
        "BM25-matched constraint must be present alongside tag results"
    );
}

#[tokio::test]
async fn fs_source_load_payload_llm_judge() {
    let dir = TempDir::new().unwrap();
    write_yaml(
        &dir,
        "ADR-002",
        "id: ADR-002\ntitle: Auth Token\nseverity: hard\ncriteria:\n  pass: Uses JWT authentication token\n  fail: Uses sessions\n",
    );

    let source = FsConstraintSource::load(dir.path()).unwrap();
    let payload = source.load_payload("ADR-002", "v1").await.unwrap();
    assert_eq!(payload.id, "ADR-002");
    assert!(matches!(
        payload.predicate,
        ConstraintPredicate::LlmJudge { .. }
    ));
}

#[tokio::test]
async fn fs_source_unknown_id_returns_empty() {
    let dir = TempDir::new().unwrap();
    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source
        .resolve_context(&[], &["NONEXISTENT".to_string()], "")
        .await;
    assert!(metas.is_empty());
}

#[tokio::test]
async fn fs_source_load_payload_missing_returns_error() {
    let dir = TempDir::new().unwrap();
    let source = FsConstraintSource::load(dir.path()).unwrap();
    let result = source.load_payload("GHOST", "v1").await;
    assert!(result.is_err());
}
