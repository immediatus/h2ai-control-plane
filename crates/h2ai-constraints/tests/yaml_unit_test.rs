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
