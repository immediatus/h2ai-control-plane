use h2ai_constraints::source::{ConstraintSource, FsConstraintSource};
use h2ai_constraints::types::ConstraintPredicate;
use std::fs;
use tempfile::TempDir;

fn write_constraint(dir: &TempDir, name: &str, content: &str) {
    fs::write(dir.path().join(format!("{name}.md")), content).unwrap();
}

#[tokio::test]
async fn fs_source_resolve_by_explicit_id() {
    let dir = TempDir::new().unwrap();
    write_constraint(&dir, "ADR-001", "## Constraints\ncite source reference");

    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source.resolve_context(&[], &["ADR-001".to_string()]).await;
    assert_eq!(metas.len(), 1);
    assert_eq!(metas[0].id, "ADR-001");
}

#[tokio::test]
async fn fs_source_resolve_by_tag() {
    let dir = TempDir::new().unwrap();
    write_constraint(
        &dir,
        "GDPR-001",
        "---\ndomains:\n  - eu_data\n---\n\n## Hard Constraints\nminimization",
    );

    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source.resolve_context(&["eu_data".to_string()], &[]).await;
    assert_eq!(metas.len(), 1);
    assert_eq!(metas[0].id, "GDPR-001");
}

#[tokio::test]
async fn fs_source_empty_tags_and_ids_returns_all_docs() {
    let dir = TempDir::new().unwrap();
    write_constraint(&dir, "ADR-001", "## Constraints\nrule one");
    write_constraint(&dir, "ADR-002", "## Constraints\nrule two");

    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source.resolve_context(&[], &[]).await;
    assert_eq!(
        metas.len(),
        2,
        "all docs returned when no tags and no explicit ids"
    );
}

#[tokio::test]
async fn fs_source_tags_with_no_domain_metadata_falls_back_to_all_docs() {
    let dir = TempDir::new().unwrap();
    // These files have NO frontmatter — context_map will be empty
    write_constraint(&dir, "ADR-001", "## Constraints\nrule one");

    let source = FsConstraintSource::load(dir.path()).unwrap();
    // Tags provided but no domain metadata in files — must fall back to all docs
    let metas = source.resolve_context(&["eu_data".to_string()], &[]).await;
    assert_eq!(
        metas.len(),
        1,
        "falls back to all docs when context_map has no entry for tag"
    );
}

#[tokio::test]
async fn fs_source_load_payload_static() {
    let dir = TempDir::new().unwrap();
    write_constraint(&dir, "ADR-002", "## Hard Constraints\nauthentication token");

    let source = FsConstraintSource::load(dir.path()).unwrap();
    let payload = source.load_payload("ADR-002", "v1").await.unwrap();
    assert_eq!(payload.id, "ADR-002");
    assert!(matches!(
        payload.predicate,
        ConstraintPredicate::VocabularyPresence { .. }
    ));
}

#[tokio::test]
async fn fs_source_unknown_id_returns_empty() {
    let dir = TempDir::new().unwrap();
    let source = FsConstraintSource::load(dir.path()).unwrap();
    let metas = source
        .resolve_context(&[], &["NONEXISTENT".to_string()])
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
