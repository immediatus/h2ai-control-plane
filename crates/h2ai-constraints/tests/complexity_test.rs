use h2ai_constraints::complexity::compute_corpus_complexity;
use h2ai_constraints::types::{
    ConstraintDoc, ConstraintPredicate, ConstraintSeverity, VocabularyMode,
};

fn make_doc(id: &str, severity: ConstraintSeverity, pred: ConstraintPredicate) -> ConstraintDoc {
    ConstraintDoc {
        id: id.into(),
        source_file: "test".into(),
        description: "test constraint".into(),
        severity,
        predicate: pred,
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }
}

#[test]
fn empty_corpus_returns_unit_tcc() {
    let meta = compute_corpus_complexity(&[]);
    assert_eq!(meta.n_constraints, 0);
    assert!((meta.tcc_structural - 1.0).abs() < 1e-9);
    assert!((meta.static_coverage - 1.0).abs() < 1e-9);
}

#[test]
fn all_hard_static_corpus_has_low_tcc() {
    let n_predicate_variants: f64 = 12.0;
    // K_TYPE = 1.0 (default coefficient)
    let k_type = 1.0_f64;
    let corpus = vec![
        make_doc(
            "c1",
            ConstraintSeverity::Hard { threshold: 0.9 },
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AllOf,
                terms: vec!["auth".into()],
            },
        ),
        make_doc(
            "c2",
            ConstraintSeverity::Hard { threshold: 0.9 },
            ConstraintPredicate::RegexMatch {
                pattern: ".*".into(),
                must_match: true,
            },
        ),
    ];
    let meta = compute_corpus_complexity(&corpus);
    assert_eq!(meta.n_constraints, 2);
    assert!((meta.soft_fraction).abs() < 1e-9);
    assert!((meta.heavy_fraction).abs() < 1e-9);
    // 2 distinct types (VocabularyPresence, RegexMatch) out of 12
    // tcc = 1.0 + 0 + K_TYPE * (2/12) + 0 = 1.0 + 1.0 * 0.1667 ≈ 1.167
    let expected = k_type.mul_add(2.0 / n_predicate_variants, 1.0);
    assert!(
        (meta.tcc_structural - expected).abs() < 1e-6,
        "tcc={} expected={}",
        meta.tcc_structural,
        expected
    );
}

#[test]
fn heavy_constraint_increases_heavy_fraction() {
    let corpus = vec![
        make_doc(
            "c1",
            ConstraintSeverity::Hard { threshold: 0.9 },
            ConstraintPredicate::OracleExecution {
                test_runner_uri: "http://localhost".into(),
                test_suite: "suite".into(),
                timeout_secs: 30,
            },
        ),
        make_doc(
            "c2",
            ConstraintSeverity::Soft { weight: 0.5 },
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AnyOf,
                terms: vec!["x".into()],
            },
        ),
    ];
    let meta = compute_corpus_complexity(&corpus);
    assert!((meta.heavy_fraction - 0.5).abs() < 1e-9);
    assert!((meta.soft_fraction - 0.5).abs() < 1e-9);
    assert!((meta.static_coverage - 0.5).abs() < 1e-9);
}

// ── Line 89: ConstraintTier::Light branch ────────────────────────────────────

#[test]
fn light_tier_constraint_not_counted_in_static_or_heavy() {
    // LlmJudge is Light tier — excluded from static_coverage and heavy_fraction
    let corpus = vec![
        make_doc(
            "c-light",
            ConstraintSeverity::Hard { threshold: 0.9 },
            ConstraintPredicate::LlmJudge {
                rubric: "evaluate this rubric".into(),
            },
        ),
        make_doc(
            "c-static",
            ConstraintSeverity::Hard { threshold: 0.9 },
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AllOf,
                terms: vec!["stateless".into()],
            },
        ),
    ];
    let meta = compute_corpus_complexity(&corpus);
    // 1 Static, 1 Light, 0 Heavy → static_coverage = 0.5, heavy_fraction = 0.0
    assert!(
        (meta.static_coverage - 0.5).abs() < 1e-9,
        "static_coverage must be 0.5 with one Light constraint"
    );
    assert!((meta.heavy_fraction).abs() < 1e-9, "no Heavy constraints");
}

// ── Lines 115-125: predicate_variant_name coverage (all variant arms) ─────────

#[test]
#[allow(clippy::too_many_lines)]
fn all_predicate_variants_contribute_to_type_diversity() {
    use h2ai_constraints::types::{CompositeOp, NumericOp};
    // Build a corpus with one of each variant so predicate_variant_name hits every arm.
    let corpus = vec![
        make_doc(
            "c-vp",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AllOf,
                terms: vec!["token".into()],
            },
        ),
        make_doc(
            "c-nk",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::NegativeKeyword {
                terms: vec!["password".into()],
            },
        ),
        make_doc(
            "c-rm",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::RegexMatch {
                pattern: r"\bUUID\b".into(),
                must_match: true,
            },
        ),
        make_doc(
            "c-nt",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::NumericThreshold {
                field_pattern: r"latency:\s+(\d+)".into(),
                op: NumericOp::Lt,
                value: 200.0,
            },
        ),
        make_doc(
            "c-oe",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::OracleExecution {
                test_runner_uri: "http://localhost/run".into(),
                test_suite: "suite.py".into(),
                timeout_secs: 30,
            },
        ),
        make_doc(
            "c-js",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::JsonSchema {
                schema: serde_json::json!({"type": "object"}),
            },
        ),
        make_doc(
            "c-lr",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::LengthRange {
                min_chars: Some(10),
                max_chars: None,
            },
        ),
        make_doc(
            "c-sp",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::SemanticPresence {
                concept: "idempotency".into(),
                passes: 3,
            },
        ),
        make_doc(
            "c-so",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::SemanticOrdering {
                first: "debit".into(),
                then: "publish".into(),
                passes: 3,
            },
        ),
        make_doc(
            "c-se",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::SemanticExclusion {
                pattern: "direct DB write".into(),
                passes: 3,
            },
        ),
        make_doc(
            "c-lj",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::LlmJudge {
                rubric: "evaluate compliance".into(),
            },
        ),
        make_doc(
            "c-comp",
            ConstraintSeverity::Hard { threshold: 0.5 },
            ConstraintPredicate::Composite {
                op: CompositeOp::And,
                children: vec![ConstraintPredicate::LlmJudge {
                    rubric: "inner".into(),
                }],
            },
        ),
    ];
    let meta = compute_corpus_complexity(&corpus);
    // All 12 variants present → type_diversity = 1.0
    assert!(
        (meta.type_diversity - 1.0).abs() < 1e-9,
        "all 12 variants must yield type_diversity=1.0, got {}",
        meta.type_diversity
    );
}

#[test]
fn corpus_hash_is_stable_across_insertion_order() {
    let a = make_doc(
        "a",
        ConstraintSeverity::Advisory,
        ConstraintPredicate::LengthRange {
            min_chars: None,
            max_chars: Some(100),
        },
    );
    let b = make_doc(
        "b",
        ConstraintSeverity::Advisory,
        ConstraintPredicate::LengthRange {
            min_chars: None,
            max_chars: Some(200),
        },
    );
    let h1 = compute_corpus_complexity(&[a.clone(), b.clone()]).corpus_hash;
    let h2 = compute_corpus_complexity(&[b, a]).corpus_hash;
    assert_eq!(h1, h2, "hash must be order-independent");
}
