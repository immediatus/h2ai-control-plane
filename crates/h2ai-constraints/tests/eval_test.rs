use h2ai_constraints::eval::eval_sync;
use h2ai_constraints::types::{CompositeOp, ConstraintPredicate, VocabularyMode};

#[test]
fn vocabulary_presence_all_of_full_match() {
    let pred = ConstraintPredicate::VocabularyPresence {
        mode: VocabularyMode::AllOf,
        terms: vec!["data".into(), "minimization".into()],
    };
    let score = eval_sync(&pred, "we apply data minimization principles");
    assert!(
        (score - 1.0).abs() < 1e-9,
        "full AllOf match must be 1.0, got {score}"
    );
}

#[test]
fn vocabulary_presence_all_of_partial_match() {
    let pred = ConstraintPredicate::VocabularyPresence {
        mode: VocabularyMode::AllOf,
        terms: vec!["data".into(), "minimization".into(), "privacy".into()],
    };
    // only "data" and "minimization" present — 2/3
    let score = eval_sync(&pred, "we apply data minimization principles");
    assert!(
        (score - 2.0 / 3.0).abs() < 1e-9,
        "partial AllOf: 2/3, got {score}"
    );
}

#[test]
fn vocabulary_presence_any_of_single_hit() {
    let pred = ConstraintPredicate::VocabularyPresence {
        mode: VocabularyMode::AnyOf,
        terms: vec!["gdpr".into(), "privacy".into(), "ccpa".into()],
    };
    let score = eval_sync(&pred, "this complies with gdpr requirements");
    assert!(
        (score - 1.0).abs() < 1e-9,
        "AnyOf with one hit must be 1.0, got {score}"
    );
}

#[test]
fn vocabulary_presence_any_of_no_hit() {
    let pred = ConstraintPredicate::VocabularyPresence {
        mode: VocabularyMode::AnyOf,
        terms: vec!["gdpr".into(), "privacy".into(), "ccpa".into()],
    };
    let score = eval_sync(&pred, "the system uses local caching");
    assert!(
        (score - 0.0).abs() < 1e-9,
        "AnyOf with no hit must be 0.0, got {score}"
    );
}

#[test]
fn vocabulary_presence_none_of_no_forbidden_terms() {
    let pred = ConstraintPredicate::VocabularyPresence {
        mode: VocabularyMode::NoneOf,
        terms: vec!["pii".into(), "password".into()],
    };
    let score = eval_sync(&pred, "the system stores aggregate statistics only");
    assert!(
        (score - 1.0).abs() < 1e-9,
        "NoneOf with no forbidden terms must be 1.0, got {score}"
    );
}

#[test]
fn vocabulary_presence_none_of_forbidden_term_present() {
    let pred = ConstraintPredicate::VocabularyPresence {
        mode: VocabularyMode::NoneOf,
        terms: vec!["password".into(), "secret".into()],
    };
    let score = eval_sync(&pred, "do not log the password field");
    assert!(
        (score - 0.0).abs() < 1e-9,
        "NoneOf with forbidden term present must be 0.0, got {score}"
    );
}

#[test]
fn negative_keyword_alias_behaves_like_none_of() {
    let pred = ConstraintPredicate::NegativeKeyword {
        terms: vec!["password".into()],
    };
    let score_clean = eval_sync(&pred, "system uses token-based auth");
    let score_dirty = eval_sync(&pred, "do not log the password");
    assert!((score_clean - 1.0).abs() < 1e-9);
    assert!((score_dirty - 0.0).abs() < 1e-9);
}

#[test]
fn composite_and_is_min_of_children() {
    let pred = ConstraintPredicate::Composite {
        op: CompositeOp::And,
        children: vec![
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AnyOf,
                terms: vec!["data".into()],
            },
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::NoneOf,
                terms: vec!["password".into()],
            },
        ],
    };
    let score = eval_sync(&pred, "we store data in encrypted form");
    assert!(
        (score - 1.0).abs() < 1e-9,
        "And of two 1.0 should be 1.0, got {score}"
    );
    let score2 = eval_sync(&pred, "we store data and password");
    assert!(
        (score2 - 0.0).abs() < 1e-9,
        "And with one 0.0 child should be 0.0, got {score2}"
    );
}

#[test]
fn composite_or_is_max_of_children() {
    let pred = ConstraintPredicate::Composite {
        op: CompositeOp::Or,
        children: vec![
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AnyOf,
                terms: vec!["gdpr".into()],
            },
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AnyOf,
                terms: vec!["ccpa".into()],
            },
        ],
    };
    let score = eval_sync(&pred, "we comply with ccpa requirements");
    assert!(
        (score - 1.0).abs() < 1e-9,
        "Or with one hit must be 1.0, got {score}"
    );
    let score_none = eval_sync(&pred, "we use local storage");
    assert!(
        (score_none - 0.0).abs() < 1e-9,
        "Or with no hits must be 0.0, got {score_none}"
    );
}

#[test]
fn composite_not_inverts_score() {
    let pred = ConstraintPredicate::Composite {
        op: CompositeOp::Not,
        children: vec![ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AnyOf,
            terms: vec!["password".into()],
        }],
    };
    let score_with = eval_sync(&pred, "log the password field");
    let score_without = eval_sync(&pred, "log the username field");
    assert!(
        (score_with - 0.0).abs() < 1e-9,
        "Not of 1.0 must be 0.0, got {score_with}"
    );
    assert!(
        (score_without - 1.0).abs() < 1e-9,
        "Not of 0.0 must be 1.0, got {score_without}"
    );
}

#[test]
fn regex_match_must_match_true_passes_when_pattern_found() {
    use h2ai_constraints::types::ConstraintPredicate;
    let pred = ConstraintPredicate::RegexMatch {
        pattern: r"\bUUID\b".into(),
        must_match: true,
    };
    assert!((eval_sync(&pred, "result UUID: abc123") - 1.0).abs() < 1e-9);
}

#[test]
fn regex_match_must_match_false_passes_when_pattern_absent() {
    use h2ai_constraints::types::ConstraintPredicate;
    let pred = ConstraintPredicate::RegexMatch {
        pattern: r"\bpassword\b".into(),
        must_match: false,
    };
    assert!((eval_sync(&pred, "the user logged in successfully") - 1.0).abs() < 1e-9);
    assert!((eval_sync(&pred, "log the password field") - 0.0).abs() < 1e-9);
}

#[test]
fn numeric_threshold_lt_passes_when_value_below() {
    use h2ai_constraints::types::{ConstraintPredicate, NumericOp};
    let pred = ConstraintPredicate::NumericThreshold {
        field_pattern: r"latency[:\s]+(\d+(?:\.\d+)?)".into(),
        op: NumericOp::Lt,
        value: 200.0,
    };
    assert!((eval_sync(&pred, "p99 latency: 150ms response time") - 1.0).abs() < 1e-9);
    assert!((eval_sync(&pred, "p99 latency: 250ms response time") - 0.0).abs() < 1e-9);
}

#[test]
fn numeric_threshold_no_match_returns_zero() {
    use h2ai_constraints::types::{ConstraintPredicate, NumericOp};
    let pred = ConstraintPredicate::NumericThreshold {
        field_pattern: r"latency[:\s]+(\d+)".into(),
        op: NumericOp::Lt,
        value: 200.0,
    };
    assert!((eval_sync(&pred, "response was fast") - 0.0).abs() < 1e-9);
}

#[test]
fn llm_judge_sync_path_passes_through() {
    use h2ai_constraints::types::ConstraintPredicate;
    let pred = ConstraintPredicate::LlmJudge {
        rubric: "Does the response correctly cite the source?".into(),
    };
    assert!((eval_sync(&pred, "anything") - 1.0).abs() < 1e-9);
}

#[test]
fn oracle_execution_sync_path_degrades_to_zero() {
    let pred = ConstraintPredicate::OracleExecution {
        test_runner_uri: "http://localhost:9999/run".into(),
        test_suite: "suite.py".into(),
        timeout_secs: 30,
    };
    assert!(
        (eval_sync(&pred, "any output") - 0.0).abs() < 1e-9,
        "OracleExecution sync path must return 0.0 (safe degradation)"
    );
}

#[test]
fn json_schema_valid_output_passes() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        },
        "required": ["name"]
    });
    let pred = ConstraintPredicate::JsonSchema { schema };
    assert!((eval_sync(&pred, r#"{"name":"Alice"}"#) - 1.0).abs() < 1e-9);
}

#[test]
fn json_schema_missing_required_field_fails() {
    let schema = serde_json::json!({
        "type": "object",
        "required": ["name"]
    });
    let pred = ConstraintPredicate::JsonSchema { schema };
    assert!((eval_sync(&pred, r#"{"age":30}"#) - 0.0).abs() < 1e-9);
}

#[test]
fn json_schema_invalid_json_fails() {
    let schema = serde_json::json!({ "type": "object" });
    let pred = ConstraintPredicate::JsonSchema { schema };
    assert!((eval_sync(&pred, "not json at all") - 0.0).abs() < 1e-9);
}

#[test]
fn length_range_within_bounds_passes() {
    let pred = ConstraintPredicate::LengthRange {
        min_chars: Some(5),
        max_chars: Some(50),
    };
    assert!((eval_sync(&pred, "hello world") - 1.0).abs() < 1e-9);
}

#[test]
fn length_range_below_min_fails() {
    let pred = ConstraintPredicate::LengthRange {
        min_chars: Some(20),
        max_chars: None,
    };
    assert!((eval_sync(&pred, "short") - 0.0).abs() < 1e-9);
}

#[test]
fn length_range_above_max_fails() {
    let pred = ConstraintPredicate::LengthRange {
        min_chars: None,
        max_chars: Some(5),
    };
    assert!((eval_sync(&pred, "this is longer than five chars") - 0.0).abs() < 1e-9);
}

#[test]
fn length_range_no_bounds_always_passes() {
    let pred = ConstraintPredicate::LengthRange {
        min_chars: None,
        max_chars: None,
    };
    assert!((eval_sync(&pred, "") - 1.0).abs() < 1e-9);
    assert!((eval_sync(&pred, "any text at all") - 1.0).abs() < 1e-9);
}
