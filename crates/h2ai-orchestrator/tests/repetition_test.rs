use h2ai_orchestrator::repetition::similarity;

#[test]
fn identical_strings_score_one() {
    let s = "the quick brown fox jumps over the lazy dog";
    assert!(
        (similarity(s, s) - 1.0).abs() < 1e-9,
        "identical strings must score 1.0"
    );
}

#[test]
fn empty_strings_score_zero() {
    assert!(
        similarity("", "").abs() < 1e-9,
        "two empty strings must score 0.0 (jaccard of empty sets)"
    );
}

#[test]
fn disjoint_strings_score_zero() {
    assert!(
        similarity("alpha beta gamma", "delta epsilon zeta").abs() < 1e-9,
        "disjoint token sets must score 0.0"
    );
}

#[test]
fn partial_overlap_scores_between_zero_and_one() {
    let a = "the quick brown fox";
    let b = "the quick red fox";
    let s = similarity(a, b);
    assert!(
        s > 0.0 && s < 1.0,
        "partial overlap should score between 0 and 1, got {s}"
    );
}

#[test]
fn word_order_does_not_affect_similarity() {
    let a = "alpha beta gamma";
    let b = "gamma alpha beta";
    assert!(
        (similarity(a, b) - 1.0).abs() < 1e-9,
        "jaccard is order-independent: same tokens = 1.0, got {}",
        similarity(a, b)
    );
}

use h2ai_adapters::mock::MockAdapter;
use h2ai_orchestrator::tao_loop::{TaoInput, TaoLoop};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::TaoConfig;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::TauValue;

#[tokio::test]
async fn tao_loop_detects_repetition_and_returns_err() {
    // Adapter always returns the same output. Pattern requires "APPROVED" which
    // never appears. Turn 1 fails → sets last_output. Turn 2 fails with identical
    // output → similarity 1.0 ≥ 0.92 threshold → Err.
    let adapter = MockAdapter::new("identical output every time".into());
    let result = TaoLoop::run(TaoInput {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        adapter: &adapter as &dyn IComputeAdapter,
        initial_request: ComputeRequest {
            system_context: "ctx".into(),
            task: "task".into(),
            tau: TauValue::new(0.5).unwrap(),
            max_tokens: 128,
        },
        config: TaoConfig {
            max_turns: 3,
            verify_pattern: Some("APPROVED".into()),
            repetition_threshold: 0.92,
            ..Default::default()
        },
        schema_config: None,
        generation: 0,
    })
    .await;

    assert!(result.is_err(), "expected Err on repetition, got Ok");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("repetition"),
        "error message should mention 'repetition', got: {msg}"
    );
}

#[tokio::test]
async fn tao_loop_allows_similar_but_passing_output() {
    // Adapter returns a passing output. Passes on turn 1 → Ok before any repetition check.
    let adapter = MockAdapter::new("APPROVED: good answer".into());
    let result = TaoLoop::run(TaoInput {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        adapter: &adapter as &dyn IComputeAdapter,
        initial_request: ComputeRequest {
            system_context: "ctx".into(),
            task: "task".into(),
            tau: TauValue::new(0.5).unwrap(),
            max_tokens: 128,
        },
        config: TaoConfig {
            max_turns: 3,
            verify_pattern: Some("APPROVED".into()),
            repetition_threshold: 0.92,
            ..Default::default()
        },
        schema_config: None,
        generation: 0,
    })
    .await;

    assert!(result.is_ok(), "passing output on turn 1 should succeed");
    assert_eq!(result.unwrap().tao_turns, 1);
}

#[tokio::test]
async fn tao_loop_disabled_repetition_exhausts_turns() {
    // repetition_threshold > 1.0 → detector disabled. Stuck adapter runs all turns.
    let adapter = MockAdapter::new("stuck output".into());
    let result = TaoLoop::run(TaoInput {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        adapter: &adapter as &dyn IComputeAdapter,
        initial_request: ComputeRequest {
            system_context: "ctx".into(),
            task: "task".into(),
            tau: TauValue::new(0.5).unwrap(),
            max_tokens: 128,
        },
        config: TaoConfig {
            max_turns: 3,
            verify_pattern: Some("MUST_NOT_MATCH".into()),
            repetition_threshold: 1.1, // disabled
            ..Default::default()
        },
        schema_config: None,
        generation: 0,
    })
    .await;

    assert!(
        result.is_ok(),
        "with repetition disabled, should exhaust turns"
    );
    assert_eq!(result.unwrap().tao_turns, 3);
}
