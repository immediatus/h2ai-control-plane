use chrono::Utc;
use h2ai_adapters::mock::MockAdapter;
use h2ai_constraints::types::aggregate_compliance_score;
use h2ai_orchestrator::verification::{VerificationInput, VerificationPhase};
use h2ai_types::config::{AdapterKind, VerificationConfig};
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::TauValue;

fn make_proposal(task_id: TaskId, text: &str) -> ProposalEvent {
    ProposalEvent {
        task_id,
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.5).unwrap(),
        generation: 0,
        raw_output: text.into(),
        token_cost: 10,
        adapter_kind: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "NONE".into(),
        },
        timestamp: Utc::now(),
    }
}

#[tokio::test]
async fn verification_passes_high_score() {
    // evaluator returns score 0.85 — should pass with default threshold 0.45
    // empty corpus → fallback to __rubric__ Hard constraint; hard_passes at 0.85 >= 0.45;
    // aggregate_compliance_score returns 1.0 (no Soft constraints) → overall = 1.0
    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "My proposal text");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(out.passed.len(), 1, "expected 1 passed proposal");
    assert_eq!(out.failed.len(), 0, "expected 0 failed proposals");
    let (_, results) = &out.passed[0];
    assert_eq!(
        results.len(),
        1,
        "expected 1 compliance result (fallback rubric)"
    );
    assert!(
        (results[0].score - 0.85).abs() < 1e-9,
        "expected raw score 0.85, got {}",
        results[0].score
    );
}

#[tokio::test]
async fn verification_fails_low_score() {
    // evaluator returns score 0.3 — should fail with explicit threshold 0.6
    let evaluator = MockAdapter::new(r#"{"score": 0.3, "reason": "missing constraints"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Incomplete proposal");

    let config = VerificationConfig {
        threshold: 0.6,
        ..Default::default()
    };

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config,
    })
    .await;

    assert_eq!(out.passed.len(), 0, "expected 0 passed proposals");
    assert_eq!(out.failed.len(), 1, "expected 1 failed proposal");
    let (_, results, violations) = &out.failed[0];
    assert_eq!(results.len(), 1, "expected 1 compliance result");
    assert!(
        (results[0].score - 0.3).abs() < 1e-9,
        "expected score 0.3, got {}",
        results[0].score
    );
    assert!(!violations.is_empty(), "expected at least one violation");
    assert_eq!(violations[0].constraint_id, "__rubric__");
}

#[tokio::test]
async fn verification_parallel_multiple_proposals() {
    // 4 proposals all at score 0.85 — verify all 4 are collected via parallel execution
    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let task_id = TaskId::new();
    let proposals = (0..4)
        .map(|i| make_proposal(task_id.clone(), &format!("Proposal {i}")))
        .collect::<Vec<_>>();

    let out = VerificationPhase::run(VerificationInput {
        proposals,
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(out.passed.len(), 4, "expected all 4 proposals to pass");
    assert_eq!(out.failed.len(), 0, "expected 0 failed proposals");
    for (_, results) in &out.passed {
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].score - 0.85).abs() < 1e-9,
            "expected score 0.85, got {}",
            results[0].score
        );
    }
}

#[tokio::test]
async fn verification_parse_error_fails_safe() {
    // evaluator returns non-JSON — should score 0.0 and FAIL (fail-safe)
    let evaluator = MockAdapter::new("not valid json".into());
    let proposal = make_proposal(TaskId::new(), "Some proposal");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(out.passed.len(), 0, "parse error must NOT pass proposals");
    assert_eq!(out.failed.len(), 1, "parse error must fail proposals");
    let (_, results, _violations) = &out.failed[0];
    assert_eq!(results.len(), 1);
    assert!(
        (results[0].score - 0.0).abs() < 1e-9,
        "expected fail-safe score 0.0, got {}",
        results[0].score
    );
}

#[tokio::test]
async fn verification_evaluator_error_fails_safe() {
    // evaluator returns empty string (unparseable) — should score 0.0
    let evaluator = MockAdapter::new(String::new());
    let proposal = make_proposal(TaskId::new(), "Some proposal");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(out.passed.len(), 0, "empty evaluator output must not pass");
    assert_eq!(out.failed.len(), 1);
    let (_, results, _violations) = &out.failed[0];
    assert!(
        (results[0].score).abs() < 1e-9,
        "expected 0.0, got {}",
        results[0].score
    );
}

#[tokio::test]
async fn verification_score_exactly_at_threshold_passes() {
    // score >= threshold is the pass condition — equality must pass.
    // Hard threshold in __rubric__ fallback is 0.45; we set both config threshold and
    // ensure the LLM score equals the hard threshold so it just passes.
    let threshold = 0.45;
    let evaluator =
        MockAdapter::new(format!(r#"{{"score": {threshold}, "reason": "at threshold"}}"#).into());
    let proposal = make_proposal(TaskId::new(), "Proposal at threshold");
    let config = VerificationConfig {
        threshold,
        ..Default::default()
    };
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config,
    })
    .await;
    assert_eq!(out.passed.len(), 1, "score exactly at threshold must pass");
    assert_eq!(out.failed.len(), 0);
}

#[tokio::test]
async fn verification_empty_proposals_returns_empty_output() {
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;
    assert!(out.passed.is_empty());
    assert!(out.failed.is_empty());
}

#[tokio::test]
async fn verification_score_clamped_above_one() {
    // LLM returns score > 1.0 — must be clamped to 1.0 and still pass.
    let evaluator = MockAdapter::new(r#"{"score": 2.5, "reason": "way above"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Test proposal");
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;
    assert_eq!(out.passed.len(), 1, "clamped score must still pass");
    let (_, results) = &out.passed[0];
    assert!(
        (results[0].score - 1.0).abs() < 1e-9,
        "score 2.5 must be clamped to 1.0, got {}",
        results[0].score
    );
}

#[tokio::test]
async fn verification_all_proposals_fail_below_threshold() {
    // score 0.5 < threshold 0.8 → hard gate fails (Hard threshold 0.45 is actually passed,
    // but overall config threshold 0.8 > aggregate score 1.0? No — aggregate is 1.0 only
    // when no Soft constraints exist, and Hard gate passes. So overall=1.0 >= threshold 0.8
    // would pass. To force failure: use a threshold above 0.5 that the Hard gate logic fails.
    // Actually __rubric__ Hard threshold is hardcoded 0.45, so score 0.5 >= 0.45 → hard_passes.
    // aggregate_compliance_score (no Soft) = 1.0. overall = 1.0 >= 0.8 → PASSES!
    // So we need score < 0.45 (Hard threshold) to fail. Use score 0.3.
    let threshold = 0.5;
    let evaluator = MockAdapter::new(r#"{"score": 0.3, "reason": "below"}"#.into());
    let task_id = TaskId::new();
    let proposals: Vec<_> = (0..3)
        .map(|i| make_proposal(task_id.clone(), &format!("Proposal {i}")))
        .collect();
    let config = VerificationConfig {
        threshold,
        ..Default::default()
    };
    let out = VerificationPhase::run(VerificationInput {
        proposals,
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config,
    })
    .await;
    assert_eq!(out.passed.len(), 0, "all below hard threshold must fail");
    assert_eq!(out.failed.len(), 3);
}

#[tokio::test]
async fn verification_aggregate_score_used_for_passed() {
    // Verify that aggregate_compliance_score is used correctly on passed proposals.
    // With empty corpus (→ Hard __rubric__) and score 0.9, aggregate = 1.0.
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Good proposal");
    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;
    assert_eq!(out.passed.len(), 1);
    let (_, results) = &out.passed[0];
    let agg = aggregate_compliance_score(results);
    // No Soft constraints → aggregate returns 1.0
    assert!(
        (agg - 1.0).abs() < 1e-9,
        "aggregate of Hard-only results should be 1.0, got {agg}"
    );
}

#[tokio::test]
async fn verification_json_schema_predicate_passes_valid_json() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let schema = serde_json::json!({"type": "object", "required": ["result"]});
    let doc = ConstraintDoc {
        id: "schema_check".into(),
        source_file: "test".into(),
        description: "output must be JSON with result field".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::JsonSchema { schema },
        remediation_hint: None,
    };
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "unused"}"#.into());
    let proposal = make_proposal(TaskId::new(), r#"{"result": "ok"}"#);

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(
        out.passed.len(),
        1,
        "valid JSON output must pass JsonSchema"
    );
    assert!((out.passed[0].1[0].score - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn verification_length_range_predicate_rejects_long_output() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "length_check".into(),
        source_file: "test".into(),
        description: "output must be ≤ 10 chars".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::LengthRange {
            min_chars: None,
            max_chars: Some(10),
        },
        remediation_hint: None,
    };
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "unused"}"#.into());
    let proposal = make_proposal(
        TaskId::new(),
        "this is definitely longer than ten characters",
    );

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(out.failed.len(), 1, "output exceeding max_chars must fail");
    assert!((out.failed[0].1[0].score - 0.0).abs() < 1e-9);
}

#[tokio::test]
async fn verification_oracle_execution_unreachable_scores_zero() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
    // Point at a URI that will immediately refuse — oracle_timeout/request_failed → 0.0
    let doc = ConstraintDoc {
        id: "oracle_check".into(),
        source_file: "test".into(),
        description: "run test suite".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::OracleExecution {
            test_runner_uri: "http://127.0.0.1:19999/run".into(),
            test_suite: "suite.py".into(),
            timeout_secs: 1,
        },
        remediation_hint: None,
    };
    let evaluator = MockAdapter::new(r#"{"score": 0.9, "reason": "unused"}"#.into());
    let proposal = make_proposal(TaskId::new(), "some output");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraint_corpus: &[doc],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(
        out.failed.len(),
        1,
        "unreachable oracle must fail the constraint"
    );
    assert!((out.failed[0].1[0].score - 0.0).abs() < 1e-9);
}
