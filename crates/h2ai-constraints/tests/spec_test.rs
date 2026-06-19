use h2ai_constraints::spec::SemanticSpec;
use h2ai_constraints::types::{CompositeOp, ConstraintPredicate};

fn minimal_spec(id: &str) -> SemanticSpec {
    SemanticSpec::builder(id)
        .rubric_pass("Proposal is stateless.")
        .rubric_fail("Proposal uses state.")
        .build()
}

#[test]
fn empty_facets_degrades_to_composite_with_single_llm_judge() {
    let doc = minimal_spec("C-000").into_constraint_doc();
    match &doc.predicate {
        ConstraintPredicate::Composite { op, children } => {
            assert_eq!(*op, CompositeOp::And);
            assert_eq!(children.len(), 1, "only LlmJudge child when no facets");
            assert!(matches!(children[0], ConstraintPredicate::LlmJudge { .. }));
        }
        other => panic!("expected Composite, got {other:?}"),
    }
}

#[test]
fn full_spec_produces_ordered_composite_exclusion_requirement_ordering_llmjudge() {
    let doc = SemanticSpec::builder("C-FULL")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .exclude("direct DB write")
        .require("Kafka topic")
        .order("debit", "publish")
        .build()
        .into_constraint_doc();
    match &doc.predicate {
        ConstraintPredicate::Composite { op, children } => {
            assert_eq!(*op, CompositeOp::And);
            assert_eq!(children.len(), 4);
            assert!(matches!(
                &children[0],
                ConstraintPredicate::SemanticExclusion { .. }
            ));
            assert!(matches!(
                &children[1],
                ConstraintPredicate::SemanticPresence { .. }
            ));
            assert!(matches!(
                &children[2],
                ConstraintPredicate::SemanticOrdering { .. }
            ));
            assert!(matches!(&children[3], ConstraintPredicate::LlmJudge { .. }));
        }
        other => panic!("expected Composite, got {other:?}"),
    }
}

#[test]
fn builder_round_trips_exclusion_pattern() {
    let doc = SemanticSpec::builder("C-EX")
        .exclude("separate GET then DECRBY")
        .rubric_pass("P")
        .rubric_fail("F")
        .build()
        .into_constraint_doc();
    if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
        if let ConstraintPredicate::SemanticExclusion { pattern, passes } = &children[0] {
            assert_eq!(pattern, "separate GET then DECRBY");
            assert_eq!(*passes, 3);
        } else {
            panic!("first child must be SemanticExclusion");
        }
    } else {
        panic!("expected Composite");
    }
}

#[test]
fn build_rubric_text_includes_domain_and_hint() {
    let spec = SemanticSpec::builder("C-R")
        .domain("billing")
        .remediation_hint("Use Lua EVAL.")
        .rubric_pass("Atomic.")
        .rubric_fail("Non-atomic.")
        .build();
    let text = spec.build_rubric_text();
    assert!(
        text.contains("Domain: billing"),
        "rubric must include Domain: line"
    );
    assert!(
        text.contains("Remediation hint: Use Lua EVAL."),
        "rubric must include hint"
    );
}

// ── Lines 101-108: checks block ──────────────────────────────────────────────

#[test]
fn build_rubric_text_includes_checks_block() {
    let spec = SemanticSpec::builder("C-CHK")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .rubric_check("Idempotency key present")
        .rubric_check("Atomic debit executed")
        .build();
    let text = spec.build_rubric_text();
    assert!(
        text.contains("Binary compliance checks"),
        "rubric must include checks header"
    );
    assert!(text.contains("1. Idempotency key present"));
    assert!(text.contains("2. Atomic debit executed"));
    assert!(
        text.contains("Score = number of checks marked PRESENT divided by 2"),
        "must include arithmetic scoring instruction"
    );
}

// ── Lines 110-122: failure_modes block ───────────────────────────────────────

#[test]
fn build_rubric_text_includes_failure_modes_without_impact() {
    use h2ai_constraints::spec::Example;
    let spec = SemanticSpec::builder("C-FM1")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .failure_mode("FM-01", "Missing key", "No idempotency key present")
        .build();
    let text = spec.build_rubric_text();
    assert!(text.contains("--- Failure Modes ---"));
    assert!(text.contains("FM-01 (Missing key): No idempotency key present"));
    // no impact string — must not contain " Impact:"
    assert!(!text.contains(" Impact:"));
    // suppress unused warning
    let _ = Example::default();
}

#[test]
fn build_rubric_text_includes_failure_modes_with_impact() {
    use h2ai_constraints::spec::{FailureMode, QualityRubric};
    let mut spec = SemanticSpec::builder("C-FM2")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .failure_mode("FM-02", "Double debit", "Charged twice for one event")
        .build();
    // inject impact via direct field mutation
    spec.rubric.failure_modes[0].impact = Some("High: financial loss".into());
    let text = spec.build_rubric_text();
    assert!(text.contains(" Impact: High: financial loss"));
    // suppress unused warning
    let _ = QualityRubric::default();
    let _ = FailureMode::default();
}

// ── Lines 124-136: negative_examples block ───────────────────────────────────

#[test]
fn build_rubric_text_negative_example_with_label_code_rationale() {
    use h2ai_constraints::spec::Example;
    let ex = Example {
        label: "Race condition scenario".into(),
        code: Some("GET balance\nif balance > 0: DECRBY".into()),
        rationale: "Non-atomic check-then-act".into(),
    };
    let spec = SemanticSpec::builder("C-NEG1")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .negative_example(ex)
        .build();
    let text = spec.build_rubric_text();
    assert!(text.contains("--- Negative Examples (DO NOT generate) ---"));
    assert!(text.contains("Scenario: Race condition scenario"));
    assert!(text.contains("GET balance"));
    assert!(text.contains("Why wrong: Non-atomic check-then-act"));
}

#[test]
fn build_rubric_text_negative_example_empty_label_and_rationale() {
    use h2ai_constraints::spec::Example;
    // label.is_empty() → no Scenario line; rationale.is_empty() → no Why wrong line
    let ex = Example {
        label: String::new(),
        code: None,
        rationale: String::new(),
    };
    let spec = SemanticSpec::builder("C-NEG2")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .negative_example(ex)
        .build();
    let text = spec.build_rubric_text();
    assert!(text.contains("--- Negative Examples (DO NOT generate) ---"));
    assert!(
        !text.contains("Scenario:"),
        "empty label must not emit Scenario line"
    );
    assert!(
        !text.contains("Why wrong:"),
        "empty rationale must not emit Why wrong line"
    );
}

// ── Lines 138-151: positive_examples block ───────────────────────────────────

#[test]
fn build_rubric_text_positive_example_with_label_code_rationale() {
    use h2ai_constraints::spec::Example;
    let ex = Example {
        label: "Lua EVAL scenario".into(),
        code: Some("EVAL script 1 balance_key amount".into()),
        rationale: "Atomic check-and-deduct in Lua".into(),
    };
    let spec = SemanticSpec::builder("C-POS1")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .positive_example(ex)
        .build();
    let text = spec.build_rubric_text();
    assert!(text.contains("--- Positive Examples (generate patterns like these) ---"));
    assert!(text.contains("Scenario: Lua EVAL scenario"));
    assert!(text.contains("EVAL script 1 balance_key amount"));
    assert!(text.contains("Why correct: Atomic check-and-deduct in Lua"));
}

#[test]
fn build_rubric_text_positive_example_empty_label_and_rationale() {
    use h2ai_constraints::spec::Example;
    let ex = Example {
        label: String::new(),
        code: None,
        rationale: String::new(),
    };
    let spec = SemanticSpec::builder("C-POS2")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .positive_example(ex)
        .build();
    let text = spec.build_rubric_text();
    assert!(text.contains("--- Positive Examples (generate patterns like these) ---"));
    assert!(
        !text.contains("Scenario:"),
        "empty label must not emit Scenario line"
    );
    assert!(
        !text.contains("Why correct:"),
        "empty rationale must not emit Why correct line"
    );
}

// ── Lines 284-307: builder methods ───────────────────────────────────────────

#[test]
fn builder_failure_mode_populates_rubric() {
    let spec = SemanticSpec::builder("C-BFM")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .failure_mode("FM-X", "FailName", "FailDesc")
        .build();
    assert_eq!(spec.rubric.failure_modes.len(), 1);
    assert_eq!(spec.rubric.failure_modes[0].id, "FM-X");
    assert_eq!(spec.rubric.failure_modes[0].name, "FailName");
    assert_eq!(spec.rubric.failure_modes[0].description, "FailDesc");
    assert!(spec.rubric.failure_modes[0].impact.is_none());
}

#[test]
fn builder_negative_example_appends() {
    use h2ai_constraints::spec::Example;
    let ex = Example {
        label: "L".into(),
        code: None,
        rationale: "R".into(),
    };
    let spec = SemanticSpec::builder("C-BNE")
        .rubric_pass("P.")
        .rubric_fail("F.")
        .negative_example(ex)
        .build();
    assert_eq!(spec.rubric.negative_examples.len(), 1);
    assert_eq!(spec.rubric.negative_examples[0].label, "L");
}

#[test]
fn builder_positive_example_appends() {
    use h2ai_constraints::spec::Example;
    let ex = Example {
        label: "Good".into(),
        code: Some("code".into()),
        rationale: "Why".into(),
    };
    let spec = SemanticSpec::builder("C-BPE")
        .rubric_pass("P.")
        .rubric_fail("F.")
        .positive_example(ex)
        .build();
    assert_eq!(spec.rubric.positive_examples.len(), 1);
    assert_eq!(spec.rubric.positive_examples[0].label, "Good");
}

#[test]
fn builder_mandatory_for_tag_appends() {
    let spec = SemanticSpec::builder("C-MFT")
        .rubric_pass("P.")
        .rubric_fail("F.")
        .mandatory_for_tag("billing")
        .mandatory_for_tag("audit")
        .build();
    assert_eq!(spec.mandatory_for_tags, vec!["billing", "audit"]);
}

// ── spec.rs:25-27: default_spec_version via serde ────────────────────────────

#[test]
fn semantic_spec_default_version_via_serde() {
    // Serialize a spec, remove "version", deserialize → default_spec_version() called
    let spec = SemanticSpec::builder("C-DEFVER")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .build();
    let mut json: serde_json::Value = serde_json::to_value(&spec).unwrap();
    json.as_object_mut().unwrap().remove("version");
    let deser: SemanticSpec = serde_json::from_value(json).unwrap();
    assert_eq!(deser.version, 1, "default_spec_version must return 1");
}

// ── spec.rs:198: pass_criteria = None when rubric.pass is empty ──────────────

#[test]
fn into_constraint_doc_empty_pass_rubric_yields_none_pass_criteria() {
    // Builder initializes rubric.pass to "" via QualityRubric::default().
    // Building without rubric_pass() leaves pass empty → pass_criteria = None.
    let spec = SemanticSpec::builder("C-NOPASS")
        .rubric_fail("Fail.")
        .build();
    let doc = spec.into_constraint_doc();
    assert!(doc.pass_criteria.is_none());
}

#[test]
fn builder_related_to_appends() {
    let spec = SemanticSpec::builder("C-REL")
        .rubric_pass("P.")
        .rubric_fail("F.")
        .related_to("C-001")
        .related_to("C-002")
        .build();
    assert_eq!(spec.related_to, vec!["C-001", "C-002"]);
}

// ── Lines 220-231: source_file, severity_hard, severity_soft ────────────────

#[test]
fn builder_source_file_sets_field() {
    let spec = SemanticSpec::builder("C-SF")
        .source_file("constraints/c-sf.yaml")
        .rubric_pass("P.")
        .rubric_fail("F.")
        .build();
    assert_eq!(spec.source_file, "constraints/c-sf.yaml");
}

#[test]
fn builder_severity_hard_sets_threshold() {
    use h2ai_constraints::types::ConstraintSeverity;
    let spec = SemanticSpec::builder("C-SH")
        .severity_hard(0.8)
        .rubric_pass("P.")
        .rubric_fail("F.")
        .build();
    assert_eq!(spec.severity, ConstraintSeverity::Hard { threshold: 0.8 });
}

#[test]
fn builder_severity_soft_sets_weight() {
    use h2ai_constraints::types::ConstraintSeverity;
    let spec = SemanticSpec::builder("C-SS")
        .severity_soft(0.5)
        .rubric_pass("P.")
        .rubric_fail("F.")
        .build();
    assert_eq!(spec.severity, ConstraintSeverity::Soft { weight: 0.5 });
}

// ── Lines 266-269: rubric_partial ────────────────────────────────────────────

#[test]
fn builder_rubric_partial_sets_partial_text() {
    let spec = SemanticSpec::builder("C-PART")
        .rubric_pass("Pass.")
        .rubric_partial("Partially compliant.")
        .rubric_fail("Fail.")
        .build();
    assert_eq!(
        spec.rubric.partial,
        Some("Partially compliant.".to_string())
    );
    // Also verify rubric text uses the custom partial
    let text = spec.build_rubric_text();
    assert!(text.contains("Partially compliant."));
}
