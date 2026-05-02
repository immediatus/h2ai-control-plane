use h2ai_adapters::mock::MockAdapter;
use h2ai_orchestrator::tao_loop::{TaoInput, TaoLoop, TaoMultiplierEstimator};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::{OutputSchemaConfig, TaoConfig};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::TauValue;

#[allow(dead_code)]
fn make_input<'a>(
    adapter: &'a dyn IComputeAdapter,
    cfg: TaoConfig,
    schema_config: Option<OutputSchemaConfig>,
) -> TaoInput<'a> {
    TaoInput {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        adapter,
        initial_request: ComputeRequest {
            system_context: "ctx".into(),
            task: "task".into(),
            tau: TauValue::new(0.5).unwrap(),
            max_tokens: 128,
        },
        config: cfg,
        schema_config,
        generation: 0,
    }
}

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
        generation: 0,
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
        repetition_threshold: 1.1, // disabled — adapter always returns same string
        ..Default::default()
    };
    let result = TaoLoop::run(TaoInput {
        task_id,
        explorer_id: ExplorerId::new(),
        adapter: &adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        initial_request: req,
        config: cfg,
        schema_config: None,
        generation: 0,
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
            repetition_threshold: 1.1, // disabled — adapter always returns same string
            ..Default::default()
        },
        schema_config: None,
        generation: 0,
    })
    .await;

    let proposal = result.expect("TAO loop completes after max_turns");
    // All 3 turns ran and all failed the pattern
    assert_eq!(proposal.tao_turns, 3, "should run all 3 turns");
    assert_eq!(proposal.iterations.len(), 3);
    // All iterations should be marked as failed
    for iter in &proposal.iterations {
        assert!(
            !iter.passed,
            "turn {} should have failed pattern",
            iter.turn
        );
    }
}

#[tokio::test]
async fn tao_max_turns_zero_returns_error() {
    let adapter = MockAdapter::new("anything".into());
    let cfg = TaoConfig {
        max_turns: 0,
        ..Default::default()
    };
    let result = TaoLoop::run(make_input(&adapter as &dyn IComputeAdapter, cfg, None)).await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("max_turns"),
        "expected max_turns in error, got: {msg}"
    );
}

#[tokio::test]
async fn tao_invalid_verify_pattern_returns_error() {
    let adapter = MockAdapter::new("output".into());
    let cfg = TaoConfig {
        max_turns: 2,
        verify_pattern: Some("[invalid regex(".into()),
        ..Default::default()
    };
    let result = TaoLoop::run(make_input(&adapter as &dyn IComputeAdapter, cfg, None)).await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("verify_pattern") || msg.contains("regex") || msg.contains("invalid"),
        "expected regex error, got: {msg}"
    );
}

#[tokio::test]
async fn tao_repetition_detected_returns_error() {
    // Adapter always returns identical output → similarity == 1.0 ≥ threshold 0.9
    let adapter = MockAdapter::new("identical output tokens".into());
    let cfg = TaoConfig {
        max_turns: 3,
        verify_pattern: Some("NEVER_MATCHES".into()),
        repetition_threshold: 0.9,
        ..Default::default()
    };
    let result = TaoLoop::run(make_input(&adapter as &dyn IComputeAdapter, cfg, None)).await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("repetition") || msg.contains("similarity"),
        "expected repetition error, got: {msg}"
    );
}

#[tokio::test]
async fn tao_schema_validation_fail_counts_as_turn_failure() {
    // Output is valid JSON but does NOT satisfy the schema (missing required "score" field).
    let adapter = MockAdapter::new(r#"{"other": "field"}"#.into());
    let schema_cfg = OutputSchemaConfig {
        schema_json: r#"{"type":"object","required":["score"]}"#.into(),
    };
    let cfg = TaoConfig {
        max_turns: 2,
        verify_pattern: None,      // no pattern — only schema controls
        repetition_threshold: 1.1, // disable repetition guard
        ..Default::default()
    };
    let result = TaoLoop::run(make_input(
        &adapter as &dyn IComputeAdapter,
        cfg,
        Some(schema_cfg),
    ))
    .await;
    // Loop exhausts turns and returns the last proposal (schema failure is non-fatal)
    let proposal = result.expect("schema failure is non-fatal — loop returns after max_turns");
    assert_eq!(proposal.tao_turns, 2);
    assert!(
        proposal.iterations.iter().all(|i| !i.passed),
        "all turns should fail schema validation"
    );
}

#[tokio::test]
async fn tao_max_turns_one_with_no_pattern_passes_immediately() {
    let adapter = MockAdapter::new("any output".into());
    let cfg = TaoConfig {
        max_turns: 1,
        verify_pattern: None,
        ..Default::default()
    };
    let result = TaoLoop::run(make_input(&adapter as &dyn IComputeAdapter, cfg, None)).await;
    let proposal = result.expect("single turn with no pattern should always pass");
    assert_eq!(proposal.tao_turns, 1);
    assert!(proposal.iterations[0].passed);
}

// ── TaoMultiplierEstimator ────────────────────────────────────────────────────

#[test]
fn tao_multiplier_estimator_prior_before_20_samples() {
    let estimator = TaoMultiplierEstimator::new_with_alpha(0.05);
    assert!(
        (estimator.multiplier() - 0.6).abs() < 1e-9,
        "prior should be 0.6, got {}",
        estimator.multiplier()
    );
}

#[test]
fn tao_multiplier_estimator_converges_after_20_samples() {
    let mut estimator = TaoMultiplierEstimator::new_with_alpha(0.05);
    for _ in 0..20 {
        estimator.update(0.5, 0.4); // ratio = 0.8
    }
    let m = estimator.multiplier();
    assert!(
        (m - 0.8).abs() < 1e-6,
        "after 20 identical samples of 0.8, multiplier should be 0.8, got {m}"
    );
}

#[test]
fn tao_multiplier_estimator_uses_prior_at_exactly_19_samples() {
    let mut estimator = TaoMultiplierEstimator::new_with_alpha(0.05);
    for _ in 0..19 {
        estimator.update(0.5, 0.4);
    }
    assert!(
        (estimator.multiplier() - 0.6).abs() < 1e-9,
        "at 19 samples prior should still hold, got {}",
        estimator.multiplier()
    );
}

#[test]
fn tao_multiplier_estimator_ignores_zero_q_before() {
    let mut estimator = TaoMultiplierEstimator::new_with_alpha(0.05);
    estimator.update(0.0, 0.5);
    assert_eq!(estimator.sample_count(), 0);
    assert!(
        (estimator.multiplier() - 0.6).abs() < 1e-9,
        "zero q_before should be ignored"
    );
}

#[test]
fn tao_multiplier_estimator_sample_count_increments() {
    let mut estimator = TaoMultiplierEstimator::new_with_alpha(0.05);
    estimator.update(0.5, 0.6);
    assert_eq!(estimator.sample_count(), 1);
    estimator.update(0.5, 0.6);
    assert_eq!(estimator.sample_count(), 2);
}

#[test]
fn tao_multiplier_estimator_ema_tracks_drift() {
    let mut estimator = TaoMultiplierEstimator::new_with_alpha(0.1);
    for _ in 0..20 {
        estimator.update(1.0, 0.6);
    }
    assert!((estimator.multiplier() - 0.6).abs() < 1e-6);
    for _ in 0..50 {
        estimator.update(1.0, 0.9);
    }
    let m = estimator.multiplier();
    assert!(
        m > 0.88,
        "EMA should drift toward 0.9 after 50 samples, got {m}"
    );
}

#[test]
fn tao_multiplier_estimator_serde_roundtrip() {
    let mut estimator = TaoMultiplierEstimator::new_with_alpha(0.05);
    for i in 0..25 {
        estimator.update(0.5, 0.3 + i as f64 * 0.01);
    }
    let json = serde_json::to_string(&estimator).unwrap();
    assert!(!json.contains("alpha"), "alpha should not be serialized");
    assert!(
        !json.contains("warmup_sum"),
        "warmup_sum should not be serialized"
    );
    let restored: TaoMultiplierEstimator = serde_json::from_str(&json).unwrap();
    let restored = restored.with_alpha(0.05);
    assert_eq!(restored.sample_count(), estimator.sample_count());
    assert!((restored.multiplier() - estimator.multiplier()).abs() < 1e-9);
}

#[test]
fn tao_multiplier_estimator_negative_q_before_skipped() {
    let mut estimator = TaoMultiplierEstimator::new_with_alpha(0.05);
    estimator.update(-0.1, 0.5);
    assert_eq!(estimator.sample_count(), 0);
}
