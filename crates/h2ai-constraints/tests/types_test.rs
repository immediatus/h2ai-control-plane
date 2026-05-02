use h2ai_constraints::types::{
    aggregate_compliance_score, ComplianceResult, ConstraintDoc, ConstraintPredicate,
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
        },
        ComplianceResult {
            constraint_id: "s2".into(),
            score: 0.4,
            severity: ConstraintSeverity::Soft { weight: 1.0 },
            remediation_hint: None,
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
    }];
    assert!((aggregate_compliance_score(&results) - 1.0).abs() < 1e-9);
}
