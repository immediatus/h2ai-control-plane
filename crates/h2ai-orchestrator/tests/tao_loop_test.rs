use h2ai_adapters::mock::MockAdapter;
use h2ai_orchestrator::tao_loop::{TaoInput, TaoLoop};
use h2ai_types::adapter::ComputeRequest;
use h2ai_types::config::TaoConfig;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::TauValue;

#[tokio::test]
async fn tao_loop_passes_on_first_turn_when_output_matches_pattern() {
    let adapter = MockAdapter::new("APPROVED: stateless JWT auth".into());
    let task_id = TaskId::new();
    let req = ComputeRequest {
        system_context: "You are a reviewer.".into(),
        task: "Propose auth strategy".into(),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 256,
    };
    let cfg = TaoConfig {
        max_turns: 3,
        verify_pattern: Some("APPROVED".into()),
        ..Default::default()
    };
    let result = TaoLoop::run(TaoInput {
        task_id,
        explorer_id: ExplorerId::new(),
        adapter: &adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        initial_request: req,
        config: cfg,
        schema_config: None,
    })
    .await;
    assert!(result.is_ok());
    let proposal = result.unwrap();
    assert_eq!(proposal.tao_turns, 1);
    assert!(proposal.event.raw_output.contains("APPROVED"));
}

#[tokio::test]
async fn tao_loop_exhausts_turns_and_returns_best() {
    let adapter = MockAdapter::new("no match here".into());
    let task_id = TaskId::new();
    let req = ComputeRequest {
        system_context: "context".into(),
        task: "task".into(),
        tau: TauValue::new(0.3).unwrap(),
        max_tokens: 256,
    };
    let cfg = TaoConfig {
        max_turns: 2,
        verify_pattern: Some("MUST_CONTAIN_THIS".into()),
        ..Default::default()
    };
    let result = TaoLoop::run(TaoInput {
        task_id,
        explorer_id: ExplorerId::new(),
        adapter: &adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        initial_request: req,
        config: cfg,
        schema_config: None,
    })
    .await;
    assert!(result.is_ok());
    let proposal = result.unwrap();
    assert_eq!(proposal.tao_turns, 2);
}

#[tokio::test]
async fn tao_memory_accumulates_on_failed_turns() {
    use h2ai_types::adapter::IComputeAdapter;

    // Pattern requires "FINAL" but adapter always returns "draft" — loop runs max_turns
    let adapter = MockAdapter::new("draft response".into());
    let result = TaoLoop::run(TaoInput {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        adapter: &adapter as &dyn IComputeAdapter,
        initial_request: ComputeRequest {
            system_context: "base context".into(),
            task: "produce output".into(),
            tau: TauValue::new(0.5).unwrap(),
            max_tokens: 128,
        },
        config: TaoConfig {
            max_turns: 3,
            verify_pattern: Some("FINAL".to_string()),
            ..Default::default()
        },
        schema_config: None,
    })
    .await;

    let proposal = result.expect("TAO loop completes after max_turns");
    // All 3 turns ran and all failed the pattern
    assert_eq!(proposal.tao_turns, 3, "should run all 3 turns");
    assert_eq!(proposal.iterations.len(), 3);
    // All iterations should be marked as failed
    for iter in &proposal.iterations {
        assert!(!iter.passed, "turn {} should have failed pattern", iter.turn);
    }
}
