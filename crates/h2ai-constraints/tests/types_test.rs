use h2ai_constraints::types::{
    aggregate_compliance_score, beta_credible_interval, count_check_verdicts,
    fractional_compliance_score, ComplianceResult, ConstraintDoc, ConstraintPredicate,
    ConstraintSeverity, VocabularyMode,
};

#[test]
fn constraint_doc_vocabulary_from_vocabulary_presence() {
    let doc = ConstraintDoc {
        id: "GDPR-001".into(),
        source_file: "gdpr.md".into(),
        description: "Data minimization".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.8 },
        predicate: ConstraintPredicate::VocabularyPresence {
            mode: h2ai_constraints::types::VocabularyMode::AllOf,
            terms: vec!["personal".into(), "data".into(), "minimization".into()],
        },
        remediation_hint: Some("Include data minimization language.".into()),
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let vocab = doc.vocabulary();
    assert!(vocab.contains("personal"));
    assert!(vocab.contains("data"));
    assert!(vocab.contains("minimization"));
}

#[test]
fn compliance_result_hard_fail_gives_zero_score() {
    let r = ComplianceResult {
        constraint_id: "ADR-001".into(),
        score: 0.3,
        severity: ConstraintSeverity::Hard { threshold: 0.8 },
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: vec![],
    };
    assert!(!r.hard_passes());
    assert!((r.score - 0.3).abs() < 1e-9);
}

#[test]
fn compliance_result_hard_pass_when_score_meets_threshold() {
    let r = ComplianceResult {
        constraint_id: "ADR-001".into(),
        score: 0.9,
        severity: ConstraintSeverity::Hard { threshold: 0.8 },
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: vec![],
    };
    assert!(r.hard_passes());
}

#[test]
fn compliance_result_soft_always_passes_hard_gate() {
    let r = ComplianceResult {
        constraint_id: "ADR-002".into(),
        score: 0.0,
        severity: ConstraintSeverity::Soft { weight: 1.0 },
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: vec![],
    };
    assert!(r.hard_passes());
}

#[test]
fn aggregate_compliance_score_weighted_average_of_soft() {
    let results = vec![
        ComplianceResult {
            constraint_id: "s1".into(),
            score: 0.8,
            severity: ConstraintSeverity::Soft { weight: 2.0 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
        ComplianceResult {
            constraint_id: "s2".into(),
            score: 0.4,
            severity: ConstraintSeverity::Soft { weight: 1.0 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
    ];
    // (2.0*0.8 + 1.0*0.4) / 3.0 = 2.0/3.0
    let score = aggregate_compliance_score(&results);
    assert!((score - 2.0 / 3.0).abs() < 1e-9, "got {score}");
}

#[test]
fn positive_vocabulary_excludes_negative_keyword_terms() {
    let doc = ConstraintDoc {
        id: "ADR-006".into(),
        source_file: "adr-006.md".into(),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.8 },
        predicate: ConstraintPredicate::Composite {
            op: h2ai_constraints::types::CompositeOp::And,
            children: vec![
                ConstraintPredicate::VocabularyPresence {
                    mode: VocabularyMode::AllOf,
                    terms: vec!["zgc".into(), "java".into()],
                },
                ConstraintPredicate::NegativeKeyword {
                    terms: vec!["g1gc".into()],
                },
            ],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let pos = doc.positive_vocabulary();
    let neg = doc.negative_vocabulary();
    let all = doc.vocabulary();

    assert!(pos.contains("zgc"), "zgc must be in positive_vocabulary");
    assert!(pos.contains("java"), "java must be in positive_vocabulary");
    assert!(
        !pos.contains("g1gc"),
        "g1gc must NOT be in positive_vocabulary"
    );

    assert!(neg.contains("g1gc"), "g1gc must be in negative_vocabulary");
    assert!(
        !neg.contains("zgc"),
        "zgc must NOT be in negative_vocabulary"
    );

    assert!(
        all.contains("zgc"),
        "vocabulary() must include positive terms"
    );
    assert!(
        all.contains("g1gc"),
        "vocabulary() must include negative terms"
    );
}

#[test]
fn negative_vocabulary_from_none_of_mode() {
    let doc = ConstraintDoc {
        id: "ADR-002".into(),
        source_file: "adr-002.md".into(),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.8 },
        predicate: ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::NoneOf,
            terms: vec!["rest".into(), "http".into()],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let neg = doc.negative_vocabulary();
    assert!(
        neg.contains("rest"),
        "NoneOf terms must appear in negative_vocabulary"
    );
    assert!(
        neg.contains("http"),
        "NoneOf terms must appear in negative_vocabulary"
    );
    assert!(
        doc.positive_vocabulary().is_empty(),
        "NoneOf must not contribute to positive_vocabulary"
    );
}

#[test]
fn aggregate_compliance_score_one_when_no_soft_constraints() {
    let results = vec![ComplianceResult {
        constraint_id: "h1".into(),
        score: 0.9,
        severity: ConstraintSeverity::Hard { threshold: 0.8 },
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: vec![],
    }];
    assert!((aggregate_compliance_score(&results) - 1.0).abs() < 1e-9);
}

// ── Lines 42-44: default_oracle_timeout_secs via serde ───────────────────────

#[test]
fn oracle_execution_default_timeout_via_serde() {
    // Deserialize without timeout_secs — serde must call default_oracle_timeout_secs()
    let json =
        r#"{"OracleExecution":{"test_runner_uri":"http://localhost/run","test_suite":"suite.py"}}"#;
    let pred: ConstraintPredicate = serde_json::from_str(json).expect("must deserialize");
    match pred {
        ConstraintPredicate::OracleExecution { timeout_secs, .. } => {
            assert_eq!(timeout_secs, 30, "default timeout must be 30");
        }
        other => panic!("expected OracleExecution, got {other:?}"),
    }
}

// ── Lines 114-116: default_binary_passes via serde ───────────────────────────

#[test]
fn semantic_presence_default_passes_via_serde() {
    let json = r#"{"SemanticPresence":{"concept":"idempotency key"}}"#;
    let pred: ConstraintPredicate = serde_json::from_str(json).expect("must deserialize");
    match pred {
        ConstraintPredicate::SemanticPresence { passes, .. } => {
            assert_eq!(passes, 3, "default passes must be 3");
        }
        other => panic!("expected SemanticPresence, got {other:?}"),
    }
}

#[test]
fn semantic_ordering_default_passes_via_serde() {
    let json = r#"{"SemanticOrdering":{"first":"debit","then":"publish"}}"#;
    let pred: ConstraintPredicate = serde_json::from_str(json).expect("must deserialize");
    match pred {
        ConstraintPredicate::SemanticOrdering { passes, .. } => {
            assert_eq!(passes, 3);
        }
        other => panic!("expected SemanticOrdering, got {other:?}"),
    }
}

#[test]
fn semantic_exclusion_default_passes_via_serde() {
    let json = r#"{"SemanticExclusion":{"pattern":"direct DB write"}}"#;
    let pred: ConstraintPredicate = serde_json::from_str(json).expect("must deserialize");
    match pred {
        ConstraintPredicate::SemanticExclusion { passes, .. } => {
            assert_eq!(passes, 3);
        }
        other => panic!("expected SemanticExclusion, got {other:?}"),
    }
}

// ── Line 291: aggregate_compliance_score with total_weight == 0.0 ─────────────

#[test]
fn aggregate_compliance_score_returns_one_when_all_weights_zero() {
    // Soft constraint with weight=0.0 → total_weight=0.0 → early return 1.0
    let results = vec![ComplianceResult {
        constraint_id: "s-zero".into(),
        score: 0.5,
        severity: ConstraintSeverity::Soft { weight: 0.0 },
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: vec![],
    }];
    assert!(
        (aggregate_compliance_score(&results) - 1.0).abs() < 1e-9,
        "zero-weight soft must return 1.0"
    );
}

#[test]
fn test_constraint_doc_binary_checks_populated_from_yaml() {
    use h2ai_constraints::yaml::ConstraintYaml;
    let yaml_str = r#"
id: TEST-001
title: "Test constraint"
severity: hard
criteria:
  pass: "passes"
  fail: "fails"
  checks:
    - "Check A is present"
    - "Check B is present"
"#;
    let cy: ConstraintYaml = serde_yaml::from_str(yaml_str).unwrap();
    let doc = cy.into_constraint_doc();
    assert_eq!(
        doc.binary_checks,
        vec!["Check A is present", "Check B is present"]
    );
}

#[test]
fn test_constraint_doc_binary_checks_empty_when_no_checks() {
    use h2ai_constraints::yaml::ConstraintYaml;
    let yaml_str = r#"
id: TEST-002
title: "No checks constraint"
severity: hard
criteria:
  pass: "passes"
  fail: "fails"
"#;
    let cy: ConstraintYaml = serde_yaml::from_str(yaml_str).unwrap();
    let doc = cy.into_constraint_doc();
    assert!(doc.binary_checks.is_empty());
}

// ── fractional_compliance_score ───────────────────────────────────────────────

#[test]
fn fractional_compliance_score_averages_all_hard_constraint_scores() {
    let results = vec![
        ComplianceResult {
            constraint_id: "HLE-1".into(),
            score: 0.33,
            severity: ConstraintSeverity::Hard { threshold: 0.8 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
        ComplianceResult {
            constraint_id: "HLE-2".into(),
            score: 0.67,
            severity: ConstraintSeverity::Hard { threshold: 0.8 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
    ];
    let score = fractional_compliance_score(&results);
    assert!((score - 0.5).abs() < 1e-9, "expected avg 0.5, got {score}");
}

#[test]
fn fractional_compliance_score_returns_zero_when_empty() {
    assert!(
        (fractional_compliance_score(&[]) - 0.0).abs() < 1e-9,
        "empty results must return 0.0"
    );
}

#[test]
fn fractional_compliance_score_includes_hard_and_soft() {
    let results = vec![
        ComplianceResult {
            constraint_id: "h1".into(),
            score: 0.2,
            severity: ConstraintSeverity::Hard { threshold: 0.8 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
        ComplianceResult {
            constraint_id: "s1".into(),
            score: 0.8,
            severity: ConstraintSeverity::Soft { weight: 1.0 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
    ];
    let score = fractional_compliance_score(&results);
    assert!(
        (score - 0.5).abs() < 1e-9,
        "expected avg 0.5 over hard+soft, got {score}"
    );
}

#[test]
fn fractional_compliance_score_all_zero_stays_zero() {
    let results = vec![
        ComplianceResult {
            constraint_id: "h1".into(),
            score: 0.0,
            severity: ConstraintSeverity::Hard { threshold: 0.8 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
        ComplianceResult {
            constraint_id: "h2".into(),
            score: 0.0,
            severity: ConstraintSeverity::Hard { threshold: 0.8 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
    ];
    assert!(
        (fractional_compliance_score(&results) - 0.0).abs() < 1e-9,
        "all-zero results must return 0.0"
    );
}

// ── Lines 328-333: hard_passes_scaled ───────────────────────────────────────

#[test]
fn compliance_result_hard_passes_scaled_with_relaxed_threshold() {
    // threshold=0.8, scale=0.9 → effective threshold 0.72; score 0.75 passes
    let r = ComplianceResult {
        constraint_id: "ADR-SCL".into(),
        score: 0.75,
        severity: ConstraintSeverity::Hard { threshold: 0.8 },
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: vec![],
    };
    assert!(r.hard_passes_scaled(0.9), "0.75 >= 0.8*0.9=0.72 must pass");
    assert!(
        !r.hard_passes_scaled(1.0),
        "0.75 < 0.8*1.0=0.80 must fail at full scale"
    );
}

#[test]
fn compliance_result_soft_hard_passes_scaled_always_true() {
    // Soft severity: hard_passes_scaled always returns true regardless of scale
    let r = ComplianceResult {
        constraint_id: "S-SCL".into(),
        score: 0.0,
        severity: ConstraintSeverity::Soft { weight: 1.0 },
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: vec![],
    };
    assert!(r.hard_passes_scaled(0.5));
}

// ── Line 365: ConstraintMeta::from_doc with empty description ────────────────

#[test]
fn constraint_meta_from_doc_empty_description_uses_fallback() {
    use h2ai_constraints::types::{ConstraintMeta, PredicateKind};
    let doc = ConstraintDoc {
        id: "C-EMPTY-DESC".into(),
        source_file: "c.yaml".into(),
        description: String::new(), // empty!
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::LengthRange {
            min_chars: Some(10),
            max_chars: None,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let meta = ConstraintMeta::from_doc(&doc);
    assert!(
        meta.summary.contains("C-EMPTY-DESC"),
        "fallback summary must contain the constraint id, got: {}",
        meta.summary
    );
    assert_eq!(meta.predicate_kind, PredicateKind::Static);
}

// ─── beta_credible_interval ───────────────────────────────────────────────────

#[test]
fn beta_ci_no_evidence_returns_full_uncertainty() {
    let (lo, hi) = beta_credible_interval(0, 0);
    assert_eq!(lo, 0.0);
    assert_eq!(hi, 1.0);
}

#[test]
fn beta_ci_all_pass_upper_bound_near_one() {
    let (lo, hi) = beta_credible_interval(10, 10);
    assert!(lo > 0.7, "lower bound for 10/10 should be > 0.7, got {lo}");
    assert!(hi <= 1.0, "upper bound must be ≤ 1.0, got {hi}");
}

#[test]
fn beta_ci_all_fail_lower_bound_near_zero() {
    let (lo, hi) = beta_credible_interval(0, 10);
    assert_eq!(lo, 0.0, "lower bound for 0/10 must be 0.0");
    assert!(hi < 0.3, "upper bound for 0/10 should be < 0.3, got {hi}");
}

#[test]
fn beta_ci_half_pass_is_symmetric_around_half() {
    let (lo, hi) = beta_credible_interval(5, 10);
    let center = (lo + hi) / 2.0;
    assert!(
        (center - 0.5).abs() < 0.02,
        "center for 5/10 should be ≈ 0.5, got {center}"
    );
}

#[test]
fn beta_ci_bounds_are_ordered_and_in_unit_interval() {
    for (k, n) in [(0, 1), (1, 1), (3, 6), (7, 8), (0, 100), (100, 100)] {
        let (lo, hi) = beta_credible_interval(k, n);
        assert!(lo <= hi, "lo={lo} must be ≤ hi={hi} for k={k} n={n}");
        assert!(
            (0.0..=1.0).contains(&lo),
            "lo={lo} out of [0,1] for k={k} n={n}"
        );
        assert!(
            (0.0..=1.0).contains(&hi),
            "hi={hi} out of [0,1] for k={k} n={n}"
        );
    }
}

// ─── count_check_verdicts ─────────────────────────────────────────────────────

fn make_result(verdicts: Vec<bool>) -> ComplianceResult {
    ComplianceResult {
        constraint_id: "C-001".into(),
        score: verdicts.iter().filter(|&&v| v).count() as f64 / verdicts.len().max(1) as f64,
        severity: ConstraintSeverity::Soft { weight: 1.0 },
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: verdicts,
        criteria_pass: None,
        check_reasons: vec![],
    }
}

#[test]
fn count_verdicts_empty_results() {
    assert_eq!(count_check_verdicts(&[]), (0, 0));
}

#[test]
fn count_verdicts_no_binary_checks() {
    let r = make_result(vec![]);
    assert_eq!(count_check_verdicts(&[r]), (0, 0));
}

#[test]
fn count_verdicts_mixed() {
    let r1 = make_result(vec![true, false, true]);
    let r2 = make_result(vec![false, false]);
    assert_eq!(count_check_verdicts(&[r1, r2]), (2, 5));
}

#[test]
fn count_verdicts_all_pass() {
    let r = make_result(vec![true, true, true]);
    assert_eq!(count_check_verdicts(&[r]), (3, 3));
}
