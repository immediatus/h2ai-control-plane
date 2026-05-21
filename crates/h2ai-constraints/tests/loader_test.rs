use h2ai_constraints::loader::{load_corpus, YamlDirSource};
use h2ai_constraints::source::ConstraintSource;
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
                    .is_some_and(|c| matches!(c, ConstraintPredicate::LlmJudge { .. })),
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

// ── Lines 50-51: deprecated predicates: array warning ────────────────────────

#[test]
fn yaml_dir_source_warns_on_deprecated_predicates_array() {
    // File uses `predicates:` (legacy) — should still load but trigger the tracing warn path
    let dir = tempfile::tempdir().unwrap();
    let yaml = "id: LEGACY-001\ntitle: Legacy\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\npredicates:\n  - type: semantic_presence\n    concept: idempotency\n";
    fs::write(dir.path().join("legacy.yaml"), yaml).unwrap();

    let source = YamlDirSource::new(dir.path());
    let specs = source
        .load_all()
        .expect("should load despite deprecated field");
    // The spec is loaded successfully (no key collision without semantic: present)
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].id, "LEGACY-001");
}

// ── Lines 62-63: Err from into_semantic_spec (semantic: + predicates: collision) ──

#[test]
fn yaml_dir_source_skips_constraint_with_semantic_and_predicates_collision() {
    let dir = tempfile::tempdir().unwrap();
    // Both semantic: and predicates: are present → into_semantic_spec returns Err
    let yaml = "id: COLLISION-001\ntitle: Collision\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\nsemantic:\n  requirements:\n    - concept: idempotency\npredicates:\n  - type: semantic_presence\n    concept: idempotency\n";
    fs::write(dir.path().join("collision.yaml"), yaml).unwrap();

    let source = YamlDirSource::new(dir.path());
    let specs = source.load_all().expect("load_all must not error on skip");
    // The conflicting constraint must be silently skipped
    assert!(specs.is_empty(), "collision constraint must be skipped");
}

// ── Lines 67-69: YAML parse error path ───────────────────────────────────────

#[test]
fn yaml_dir_source_skips_invalid_yaml() {
    let dir = tempfile::tempdir().unwrap();
    // Malformed YAML that cannot be deserialized
    fs::write(dir.path().join("bad.yaml"), "{ not: valid: yaml: [}\n").unwrap();

    let source = YamlDirSource::new(dir.path());
    let specs = source
        .load_all()
        .expect("load_all must not panic on bad YAML");
    assert!(
        specs.is_empty(),
        "invalid YAML files must be silently skipped"
    );
}

// ── Lines 67-69: load_corpus with bad YAML ───────────────────────────────────

#[test]
fn load_corpus_skips_bad_yaml_files() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("bad.yaml"), "{ invalid yaml [ }\n").unwrap();
    fs::write(
        dir.path().join("good.yaml"),
        "id: GOOD-001\ntitle: Good\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\n",
    )
    .unwrap();

    let corpus = load_corpus(dir.path()).unwrap();
    assert_eq!(
        corpus.len(),
        1,
        "bad YAML file must be skipped by load_corpus"
    );
    assert_eq!(corpus[0].id, "GOOD-001");
}

// ── Line 106: load_corpus parse_yaml_constraint returning None ────────────────

#[test]
fn load_corpus_skips_collision_yaml() {
    // parse_yaml_constraint returns None when both semantic: and predicates: are present
    let dir = tempfile::tempdir().unwrap();
    let collision_yaml = "id: COLL-001\ntitle: Collision\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\nsemantic:\n  requirements:\n    - concept: idempotency\npredicates:\n  - type: semantic_presence\n    concept: idempotency\n";
    fs::write(dir.path().join("collision.yaml"), collision_yaml).unwrap();

    let corpus = load_corpus(dir.path()).unwrap();
    assert!(
        corpus.is_empty(),
        "collision constraint must be skipped by load_corpus"
    );
}

// ── Line 106: load_corpus duplicate ID deduplication ─────────────────────────

#[test]
fn load_corpus_duplicate_id_second_file_skipped() {
    let dir = tempfile::tempdir().unwrap();
    // Two files with same constraint id — first alphabetically wins
    fs::write(
        dir.path().join("aaa-first.yaml"),
        "id: DUP-001\ntitle: First version\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("zzz-second.yaml"),
        "id: DUP-001\ntitle: Second version\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\n",
    )
    .unwrap();

    let corpus = load_corpus(dir.path()).unwrap();
    assert_eq!(
        corpus.len(),
        1,
        "duplicate IDs in load_corpus must be deduplicated"
    );
    assert_eq!(
        corpus[0].description, "First version",
        "first alphabetical file must win"
    );
}
