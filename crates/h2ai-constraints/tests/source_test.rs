use h2ai_constraints::resolver::ConstraintResolver;
use h2ai_constraints::source::FsConstraintStore;
use h2ai_constraints::types::ConstraintPredicate;
use std::fs;
use std::sync::Arc;
use tempfile::TempDir;

fn write_yaml(dir: &TempDir, name: &str, content: &str) {
    fs::write(dir.path().join(format!("{name}.yaml")), content).unwrap();
}

fn simple_yaml(id: &str, pass: &str) -> String {
    format!(
        "id: {id}\ntitle: {id}\nseverity: hard\ncriteria:\n  pass: {pass}\n  fail: Does not satisfy\n"
    )
}

async fn make_resolver(dir: &TempDir) -> ConstraintResolver {
    let (index, store) = FsConstraintStore::load(dir.path()).await.unwrap();
    ConstraintResolver::new(Arc::new(index), Arc::new(store))
}

#[tokio::test]
async fn fs_source_resolve_by_explicit_id() {
    let dir = TempDir::new().unwrap();
    write_yaml(
        &dir,
        "ADR-001",
        &simple_yaml("ADR-001", "Cites a source reference"),
    );

    let resolver = make_resolver(&dir).await;
    let docs = resolver.resolve(&["ADR-001".to_string()], &[], "").await;
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].id, "ADR-001");
}

#[tokio::test]
async fn fs_source_resolve_by_tag() {
    let dir = TempDir::new().unwrap();
    write_yaml(
        &dir,
        "GDPR-001",
        "id: GDPR-001\ntitle: Data Minimization\nseverity: hard\ndomains:\n  - eu_data\ncriteria:\n  pass: Minimizes personal data\n  fail: Over-collects\n",
    );

    let resolver = make_resolver(&dir).await;
    let docs = resolver.resolve(&[], &["eu_data".to_string()], "").await;
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].id, "GDPR-001");
}

#[tokio::test]
async fn fs_source_empty_filters_returns_empty() {
    let dir = TempDir::new().unwrap();
    write_yaml(&dir, "ADR-001", &simple_yaml("ADR-001", "Rule one"));
    write_yaml(&dir, "ADR-002", &simple_yaml("ADR-002", "Rule two"));

    let resolver = make_resolver(&dir).await;
    let docs = resolver.resolve(&[], &[], "").await;
    assert!(
        docs.is_empty(),
        "no filters → resolver returns nothing (caller must provide criteria)"
    );
}

#[tokio::test]
async fn fs_source_unknown_tag_returns_empty() {
    let dir = TempDir::new().unwrap();
    write_yaml(&dir, "ADR-001", &simple_yaml("ADR-001", "Rule one"));

    let resolver = make_resolver(&dir).await;
    let docs = resolver.resolve(&[], &["eu_data".to_string()], "").await;
    assert!(
        docs.is_empty(),
        "tag with no domain metadata match returns empty"
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

    let resolver = make_resolver(&dir).await;
    let docs = resolver
        .resolve(
            &[],
            &["billing".to_string()],
            "stateless service request handling",
        )
        .await;
    let ids: std::collections::HashSet<&str> = docs.iter().map(|d| d.id.as_str()).collect();
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

    let resolver = make_resolver(&dir).await;
    let docs = resolver.resolve(&["ADR-002".to_string()], &[], "").await;
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].id, "ADR-002");
    assert!(matches!(
        docs[0].predicate,
        ConstraintPredicate::LlmJudge { .. }
    ));
}

#[tokio::test]
async fn fs_source_unknown_id_returns_empty() {
    let dir = TempDir::new().unwrap();
    let resolver = make_resolver(&dir).await;
    let docs = resolver
        .resolve(&["NONEXISTENT".to_string()], &[], "")
        .await;
    assert!(docs.is_empty());
}

#[tokio::test]
async fn fs_source_checks_embedded_in_rubric() {
    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("C-CHECKS.yaml"),
        "id: C-CHECKS\ntitle: RTB Timeout\nseverity: hard\ncriteria:\n  pass: All four behaviors present.\n  fail: Raises global deadline.\n  checks:\n    - Does the proposal track latency per-DSP locally?\n    - Is T_global kept at 100ms or below?\n",
    )
    .unwrap();

    let resolver = make_resolver(&dir).await;
    let docs = resolver.resolve(&["C-CHECKS".to_string()], &[], "").await;
    assert_eq!(docs.len(), 1);
    let rubric = match &docs[0].predicate {
        ConstraintPredicate::LlmJudge { rubric } => rubric.clone(),
        _ => panic!("expected LlmJudge predicate"),
    };
    assert!(
        rubric.contains("Binary compliance checks"),
        "rubric must embed the binary checks section; got: {rubric}"
    );
    assert!(
        rubric.contains("Track latency per-DSP locally")
            || rubric.contains("track latency per-DSP locally"),
        "first check question must appear in rubric"
    );
    assert!(
        rubric.contains("Score = number of checks marked PRESENT"),
        "arithmetic scoring instruction must appear in rubric"
    );
}

#[tokio::test]
async fn fs_source_no_checks_rubric_unaffected() {
    let dir = TempDir::new().unwrap();
    write_yaml(
        &dir,
        "C-PLAIN",
        &simple_yaml("C-PLAIN", "Must be stateless"),
    );

    let resolver = make_resolver(&dir).await;
    let docs = resolver.resolve(&["C-PLAIN".to_string()], &[], "").await;
    assert_eq!(docs.len(), 1);
    let rubric = match &docs[0].predicate {
        ConstraintPredicate::LlmJudge { rubric } => rubric.clone(),
        _ => panic!("expected LlmJudge predicate"),
    };
    assert!(
        !rubric.contains("Binary compliance checks"),
        "rubric without checks must not contain binary checks section"
    );
}
