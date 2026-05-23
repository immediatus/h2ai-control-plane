use h2ai_constraints::types::{ConstraintPredicate, ConstraintSeverity};
use h2ai_constraints::yaml::{parse_yaml_constraint, ConstraintYaml};
use std::path::Path;

// ── Lines 209-211: default_severity() — serde default when severity: absent ──

#[test]
fn yaml_missing_severity_defaults_to_hard() {
    let yaml = "id: C-NO-SEV\ntitle: No Severity\ncriteria:\n  pass: ok\n  fail: bad\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let doc = parsed.into_constraint_doc();
    assert!(
        matches!(doc.severity, ConstraintSeverity::Hard { .. }),
        "absent severity must default to Hard"
    );
}

// ── Lines 300-303: Advisory severity in into_constraint_doc ──────────────────

#[test]
fn yaml_advisory_severity_produces_advisory_constraint() {
    let yaml =
        "id: C-ADV\ntitle: Advisory\nseverity: advisory\ncriteria:\n  pass: ok\n  fail: bad\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let doc = parsed.into_constraint_doc();
    assert_eq!(doc.severity, ConstraintSeverity::Advisory);
}

// ── Lines 311-315: both semantic: and predicates: → warn + vec![] ─────────────

#[test]
fn yaml_semantic_and_predicates_collision_produces_single_llm_judge() {
    // When both are present, structural_predicates becomes vec![]
    // Only LlmJudge ends up in the Composite
    let yaml = "id: C-COLL\ntitle: Collision\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\nsemantic:\n  requirements:\n    - concept: idempotency\npredicates:\n  - type: semantic_presence\n    concept: idempotency\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let doc = parsed.into_constraint_doc();
    match &doc.predicate {
        ConstraintPredicate::Composite { children, .. } => {
            // Only LlmJudge; structural predicates were cleared due to collision
            assert_eq!(
                children.len(),
                1,
                "collision must produce only LlmJudge child"
            );
            assert!(matches!(&children[0], ConstraintPredicate::LlmJudge { .. }));
        }
        other => panic!("expected Composite, got {other:?}"),
    }
}

// ── Lines 352-357: unknown numeric_check op defaults to Le ───────────────────

#[test]
fn yaml_unknown_numeric_op_defaults_to_le() {
    let yaml = "id: C-NUMOP\ntitle: Unknown Op\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\nnumeric_checks:\n  - pattern: 'score[:\\s]+(\\d+)'\n    op: unknown_op\n    value: 100\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let doc = parsed.into_constraint_doc();
    // Should produce a NumericThreshold with Le op
    match &doc.predicate {
        ConstraintPredicate::Composite { children, .. } => {
            let nt = children
                .iter()
                .find(|c| matches!(c, ConstraintPredicate::NumericThreshold { .. }));
            assert!(nt.is_some(), "must have a NumericThreshold child");
            if let Some(ConstraintPredicate::NumericThreshold { op, .. }) = nt {
                assert!(
                    matches!(op, h2ai_constraints::types::NumericOp::Le),
                    "unknown op must default to Le"
                );
            }
        }
        other => panic!("expected Composite, got {other:?}"),
    }
}

// ── Lines 397-401: into_semantic_spec collision error ────────────────────────

#[test]
fn yaml_into_semantic_spec_returns_err_on_collision() {
    let yaml = "id: C-SPEC-COLL\ntitle: Spec Collision\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\nsemantic:\n  requirements:\n    - concept: idempotency\npredicates:\n  - type: semantic_presence\n    concept: idempotency\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let result = parsed.into_semantic_spec();
    assert!(result.is_err(), "collision must return Err");
    let msg = result.unwrap_err();
    assert!(
        msg.contains("C-SPEC-COLL"),
        "error message must contain constraint id"
    );
}

// ── Lines 413-416: Soft and Advisory severity in into_semantic_spec ──────────

#[test]
fn yaml_into_semantic_spec_advisory_severity() {
    let yaml = "id: C-ADV-SPEC\ntitle: Advisory Spec\nseverity: advisory\ncriteria:\n  pass: ok\n  fail: bad\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let spec = parsed.into_semantic_spec().expect("must succeed");
    assert_eq!(spec.severity, ConstraintSeverity::Advisory);
}

#[test]
fn yaml_into_semantic_spec_soft_severity() {
    // Covers the Soft arm (lines 413-415) in into_semantic_spec
    let yaml = "id: C-SOFT-SPEC\ntitle: Soft Spec\nseverity: soft\nthreshold: 0.6\ncriteria:\n  pass: ok\n  fail: bad\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let spec = parsed.into_semantic_spec().expect("must succeed");
    match spec.severity {
        ConstraintSeverity::Soft { weight } => {
            assert!((weight - 0.6).abs() < 1e-9);
        }
        other => panic!("expected Soft severity, got {other:?}"),
    }
}

// ── Lines 521-527: map_legacy_predicates — semantic_exclusion type ────────────

#[test]
fn yaml_legacy_predicates_semantic_exclusion_maps_correctly() {
    let yaml = "id: C-EXCL\ntitle: Exclusion\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\npredicates:\n  - type: semantic_exclusion\n    pattern: direct DB write\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let doc = parsed.into_constraint_doc();
    match &doc.predicate {
        ConstraintPredicate::Composite { children, .. } => {
            let has_exclusion = children.iter().any(|c| {
                matches!(c, ConstraintPredicate::SemanticExclusion { pattern, .. } if pattern == "direct DB write")
            });
            assert!(
                has_exclusion,
                "semantic_exclusion legacy predicate must be mapped"
            );
        }
        other => panic!("expected Composite, got {other:?}"),
    }
}

// ── Line 529: map_legacy_predicates — unknown type is silently ignored ─────────

#[test]
fn yaml_legacy_predicates_unknown_type_ignored() {
    let yaml = "id: C-UNK\ntitle: Unknown Type\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\npredicates:\n  - type: unknown_predicate_type\n    concept: something\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let doc = parsed.into_constraint_doc();
    // Unknown type → section stays empty → only LlmJudge in Composite
    match &doc.predicate {
        ConstraintPredicate::Composite { children, .. } => {
            assert_eq!(
                children.len(),
                1,
                "unknown predicate type must produce only LlmJudge"
            );
            assert!(matches!(&children[0], ConstraintPredicate::LlmJudge { .. }));
        }
        other => panic!("expected Composite, got {other:?}"),
    }
}

// ── Line 542: parse_yaml_constraint with collision → None ─────────────────────

#[test]
fn parse_yaml_constraint_returns_none_on_semantic_predicates_collision() {
    let yaml = "id: C-PARSE-COLL\ntitle: Parse Collision\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\nsemantic:\n  requirements:\n    - concept: idempotency\npredicates:\n  - type: semantic_presence\n    concept: idempotency\n";
    let path = Path::new("test.yaml");
    let result = parse_yaml_constraint(path, yaml);
    assert!(
        result.is_none(),
        "parse_yaml_constraint must return None on collision"
    );
}

// ── Lines 550-556: parse_yaml_constraint with bad YAML → None ─────────────────

#[test]
fn parse_yaml_constraint_returns_none_on_bad_yaml() {
    let bad_yaml = "{ not: valid yaml [ }\n";
    let path = Path::new("bad.yaml");
    let result = parse_yaml_constraint(path, bad_yaml);
    assert!(
        result.is_none(),
        "parse_yaml_constraint must return None on bad YAML"
    );
}

// ── Additional: Soft severity with threshold in into_constraint_doc ───────────

#[test]
fn yaml_soft_severity_with_threshold() {
    let yaml = "id: C-SOFT\ntitle: Soft\nseverity: soft\nthreshold: 0.7\ncriteria:\n  pass: ok\n  fail: bad\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let doc = parsed.into_constraint_doc();
    match doc.severity {
        ConstraintSeverity::Soft { weight } => {
            assert!((weight - 0.7).abs() < 1e-9);
        }
        other => panic!("expected Soft, got {other:?}"),
    }
}

// ── Lines 234-235: build_rubric includes domain context ──────────────────────

#[test]
fn yaml_build_rubric_appends_domains() {
    let yaml = "id: C-DOM\ntitle: Domain Constraint\nseverity: hard\ndomains:\n  - payments\n  - compliance\ncriteria:\n  pass: ok\n  fail: bad\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let rubric = parsed.build_rubric();
    assert!(
        rubric.contains("Domain: payments, compliance"),
        "rubric must include domain context"
    );
}

// ── Lines 237-239: build_rubric includes remediation_hint ────────────────────

#[test]
fn yaml_build_rubric_appends_remediation_hint() {
    let yaml = "id: C-HINT\ntitle: Hint Constraint\nseverity: hard\nremediation_hint: 'Use exponential backoff'\ncriteria:\n  pass: ok\n  fail: bad\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let rubric = parsed.build_rubric();
    assert!(
        rubric.contains("Remediation hint: Use exponential backoff"),
        "rubric must include remediation hint"
    );
}

// ── Lines 256-260: build_rubric failure_modes with impact field ───────────────

#[test]
fn yaml_build_rubric_failure_modes_with_impact() {
    let yaml = "id: C-FM\ntitle: Failure Modes\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\nfailure_modes:\n  - id: FM-1\n    name: Timeout\n    description: Request times out\n    impact: Revenue loss\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let rubric = parsed.build_rubric();
    assert!(
        rubric.contains("Failure Modes"),
        "rubric must have failure modes section"
    );
    assert!(
        rubric.contains("Impact: Revenue loss"),
        "rubric must include impact"
    );
}

// ── Lines 264-278: build_rubric negative_examples with all optional fields ───

#[test]
fn yaml_build_rubric_negative_examples_with_label_code_rationale() {
    let yaml = "id: C-NEG\ntitle: Negative Examples\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\nnegative_examples:\n  - scenario: 'Blocking sync call'\n    code: 'db.query()'\n    why_wrong: 'Blocks the event loop'\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let rubric = parsed.build_rubric();
    assert!(rubric.contains("Negative Examples"), "must have section");
    assert!(
        rubric.contains("Blocking sync call"),
        "must have scenario label"
    );
    assert!(rubric.contains("db.query()"), "must have code");
    assert!(
        rubric.contains("Why wrong: Blocks the event loop"),
        "must have rationale"
    );
}

#[test]
fn yaml_build_rubric_negative_examples_empty_label_and_rationale_skipped() {
    // Example with no scenario, no code, no why_wrong → label and rationale writes are skipped
    let yaml = "id: C-NEG-EMPTY\ntitle: Negative Empty\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\nnegative_examples:\n  - {}\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let rubric = parsed.build_rubric();
    assert!(
        rubric.contains("Negative Examples"),
        "section must still appear"
    );
}

// ── Lines 280-294: build_rubric positive_examples with all optional fields ───

#[test]
fn yaml_build_rubric_positive_examples_with_label_code_rationale() {
    let yaml = "id: C-POS\ntitle: Positive Examples\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\npositive_examples:\n  - scenario: 'Async handler'\n    code: 'async fn handle()'\n    why_correct: 'Non-blocking'\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let rubric = parsed.build_rubric();
    assert!(rubric.contains("Positive Examples"), "must have section");
    assert!(rubric.contains("Async handler"), "must have scenario label");
    assert!(rubric.contains("async fn handle()"), "must have code");
    assert!(
        rubric.contains("Why correct: Non-blocking"),
        "must have rationale"
    );
}

#[test]
fn yaml_build_rubric_positive_examples_empty_label_skipped() {
    // Positive example with no scenario/why_correct → label and rationale skipped (empty string branch)
    let yaml = "id: C-POS-EMPTY\ntitle: Positive Empty\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\npositive_examples:\n  - code: 'let x = 1;'\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let rubric = parsed.build_rubric();
    assert!(rubric.contains("Positive Examples"), "section must appear");
    assert!(rubric.contains("let x = 1;"), "code must appear");
    assert!(
        !rubric.contains("Scenario:"),
        "empty label must not produce Scenario: line"
    );
}

// ── Lines 307-310: Hard severity with n>=2 checks → threshold = (n-1)/n ──────

#[test]
fn yaml_hard_severity_with_two_binary_checks_computes_fractional_threshold() {
    // 2 checks → threshold = (2-1)/2 = 0.5
    let yaml = "id: C-BIN2\ntitle: Binary Checks\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\n  checks:\n    - Check A\n    - Check B\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let doc = parsed.into_constraint_doc();
    match doc.severity {
        ConstraintSeverity::Hard { threshold } => {
            let expected = 1.0_f64 / 2.0_f64;
            assert!(
                (threshold - expected).abs() < 1e-9,
                "2-check threshold must be 0.5, got {threshold}"
            );
        }
        other => panic!("expected Hard, got {other:?}"),
    }
}

#[test]
fn yaml_hard_severity_with_three_binary_checks_computes_fractional_threshold() {
    // 3 checks → threshold = (3-1)/3 ≈ 0.667
    let yaml = "id: C-BIN3\ntitle: Three Checks\nseverity: hard\ncriteria:\n  pass: ok\n  fail: bad\n  checks:\n    - Check A\n    - Check B\n    - Check C\n";
    let parsed: ConstraintYaml = serde_yaml::from_str(yaml).expect("must parse");
    let doc = parsed.into_constraint_doc();
    match doc.severity {
        ConstraintSeverity::Hard { threshold } => {
            let expected = 2.0_f64 / 3.0_f64;
            assert!(
                (threshold - expected).abs() < 1e-9,
                "3-check threshold must be 2/3, got {threshold}"
            );
        }
        other => panic!("expected Hard, got {other:?}"),
    }
}
