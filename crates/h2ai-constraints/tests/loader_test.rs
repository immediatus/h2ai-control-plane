use h2ai_constraints::loader::load_corpus;
use h2ai_constraints::types::ConstraintPredicate;
use std::fs;

#[test]
fn load_corpus_missing_dir_returns_empty() {
    let result = load_corpus("/tmp/h2ai-test-nonexistent-dir-xyz");
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn load_corpus_loads_yaml_files() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("ADR-001.yaml"),
        "id: ADR-001\ntitle: Stateless Auth\nseverity: hard\ncriteria:\n  pass: Uses stateless JWT\n  fail: Uses sessions\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("ADR-002.yaml"),
        "id: ADR-002\ntitle: gRPC Internal\nseverity: hard\ncriteria:\n  pass: Uses gRPC internally\n  fail: Uses REST internally\n",
    )
    .unwrap();

    let corpus = load_corpus(dir.path()).unwrap();
    assert_eq!(corpus.len(), 2);
    let ids: Vec<_> = corpus.iter().map(|d| d.id.as_str()).collect();
    assert!(ids.contains(&"ADR-001") && ids.contains(&"ADR-002"));
}

#[test]
fn load_corpus_ignores_non_yaml_files() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("ADR-001.yaml"),
        "id: ADR-001\ntitle: T\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\n",
    )
    .unwrap();
    fs::write(dir.path().join("README.md"), "# docs").unwrap();
    fs::write(dir.path().join("notes.txt"), "notes").unwrap();

    let corpus = load_corpus(dir.path()).unwrap();
    assert_eq!(corpus.len(), 1, "only YAML files must be loaded");
    assert_eq!(corpus[0].id, "ADR-001");
}

#[test]
fn load_corpus_yaml_produces_composite_predicate_ending_in_llm_judge() {
    use h2ai_constraints::types::CompositeOp;
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("C-001.yaml"),
        "id: C-001\ntitle: Budget Pacing\nseverity: hard\ncriteria:\n  pass: Idempotent atomic debit\n  fail: No idempotency\n",
    )
    .unwrap();

    let corpus = load_corpus(dir.path()).unwrap();
    assert_eq!(corpus.len(), 1);
    match &corpus[0].predicate {
        ConstraintPredicate::Composite { op, children } => {
            assert_eq!(*op, CompositeOp::And);
            assert!(
                children
                    .last()
                    .map(|c| matches!(c, ConstraintPredicate::LlmJudge { .. }))
                    .unwrap_or(false),
                "YAML constraint Composite must end with LlmJudge"
            );
        }
        other => panic!("YAML constraint must produce Composite predicate; got {other:?}"),
    }
}

#[test]
fn load_corpus_preserves_domains_and_tags() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("GDPR-001.yaml"),
        "id: GDPR-001\ntitle: Data Minimization\nseverity: hard\ndomains:\n  - eu_data\n  - compliance\nmandatory_for_tags:\n  - audit\ncriteria:\n  pass: Minimizes data\n  fail: Over-collects\n",
    )
    .unwrap();

    let corpus = load_corpus(dir.path()).unwrap();
    assert_eq!(corpus[0].domains, vec!["eu_data", "compliance"]);
    assert_eq!(corpus[0].mandatory_for_tags, vec!["audit"]);
}

#[test]
fn load_corpus_deduplicates_same_id() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("A-C-001.yaml"),
        "id: C-001\ntitle: First\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("B-C-001.yaml"),
        "id: C-001\ntitle: Duplicate\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\n",
    )
    .unwrap();

    let corpus = load_corpus(dir.path()).unwrap();
    assert_eq!(corpus.len(), 1, "duplicate IDs must be deduplicated");
}
