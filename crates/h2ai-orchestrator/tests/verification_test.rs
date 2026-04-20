use h2ai_adapters::mock::MockAdapter;
use h2ai_orchestrator::verification::{VerificationInput, VerificationPhase};
use h2ai_types::config::{AdapterKind, VerificationConfig};
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::TauValue;
use chrono::Utc;

fn make_proposal(task_id: TaskId, text: &str) -> ProposalEvent {
    ProposalEvent {
        task_id,
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.5).unwrap(),
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
    let evaluator = MockAdapter::new(r#"{"score": 0.85, "reason": "good"}"#.into());
    let proposal = make_proposal(TaskId::new(), "My proposal text");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraints: &["ADR-001".to_string()],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(out.passed.len(), 1, "expected 1 passed proposal");
    assert_eq!(out.failed.len(), 0, "expected 0 failed proposals");
    let (_, score) = &out.passed[0];
    assert!(
        (*score - 0.85).abs() < 1e-9,
        "expected score 0.85, got {score}"
    );
}

#[tokio::test]
async fn verification_fails_low_score() {
    // evaluator returns score 0.3 — should fail with explicit threshold 0.6
    let evaluator =
        MockAdapter::new(r#"{"score": 0.3, "reason": "missing constraints"}"#.into());
    let proposal = make_proposal(TaskId::new(), "Incomplete proposal");

    let config = VerificationConfig {
        threshold: 0.6,
        ..Default::default()
    };

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraints: &["ADR-001".to_string()],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config,
    })
    .await;

    assert_eq!(out.passed.len(), 0, "expected 0 passed proposals");
    assert_eq!(out.failed.len(), 1, "expected 1 failed proposal");
    let (_, score, reason) = &out.failed[0];
    assert!(
        (*score - 0.3).abs() < 1e-9,
        "expected score 0.3, got {score}"
    );
    assert_eq!(reason, "missing constraints");
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
        constraints: &["ADR-001".to_string(), "ADR-002".to_string()],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(out.passed.len(), 4, "expected all 4 proposals to pass");
    assert_eq!(out.failed.len(), 0, "expected 0 failed proposals");
    for (_, score) in &out.passed {
        assert!(
            (*score - 0.85).abs() < 1e-9,
            "expected score 0.85, got {score}"
        );
    }
}

#[tokio::test]
async fn verification_fallback_on_non_json() {
    // evaluator returns non-JSON — should fall back to score 0.5, pass with default threshold 0.45
    let evaluator = MockAdapter::new("not valid json".into());
    let proposal = make_proposal(TaskId::new(), "Some proposal");

    let out = VerificationPhase::run(VerificationInput {
        proposals: vec![proposal],
        constraints: &[],
        evaluator: &evaluator as &dyn h2ai_types::adapter::IComputeAdapter,
        config: VerificationConfig::default(),
    })
    .await;

    assert_eq!(out.passed.len(), 1, "expected graceful fallback to pass");
    assert_eq!(out.failed.len(), 0, "expected no failures on parse error");
    let (_, score) = &out.passed[0];
    assert!(
        (*score - 0.5).abs() < 1e-9,
        "expected fallback score 0.5, got {score}"
    );
}
