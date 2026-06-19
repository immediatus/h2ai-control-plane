//! Pure-function tests for `phases::generation`.
//!
//! Only tests `generation_outcome` — the sole pure function in that module.
//! The `run()` function requires a real LLM adapter + store and is not tested here.

#![allow(
    clippy::float_cmp,
    clippy::missing_panics_doc,
    clippy::must_use_candidate
)]

use chrono::Utc;
use h2ai_orchestrator::phases::generation::{generation_outcome, GenerationPhaseResult};
use h2ai_types::config::AdapterKind;
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::TauValue;

fn make_proposal() -> ProposalEvent {
    ProposalEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.5).unwrap(),
        generation: 0,
        raw_output: "proposal text".to_string(),
        token_cost: 100,
        adapter_kind: AdapterKind::CloudGeneric {
            endpoint: "mock://test".into(),
            api_key_env: "NONE".into(),
            model: None,
            provider: Default::default(),
        },
        timestamp: Utc::now(),
    }
}

// ── AllTimedOut branch ────────────────────────────────────────────────────────

#[test]
fn all_timed_out_when_completed_empty_with_timeouts() {
    let result = generation_outcome(vec![], 3);
    assert!(
        matches!(result, GenerationPhaseResult::AllTimedOut),
        "empty completed + positive timed_out_count must be AllTimedOut"
    );
}

#[test]
fn all_timed_out_when_completed_empty_zero_timeouts() {
    // Both counts zero → no output at all → AllTimedOut
    let result = generation_outcome(vec![], 0);
    assert!(
        matches!(result, GenerationPhaseResult::AllTimedOut),
        "empty completed + zero timed_out_count must still be AllTimedOut"
    );
}

// ── Full branch ───────────────────────────────────────────────────────────────

#[test]
fn full_when_some_completed_and_zero_timeouts() {
    let p = make_proposal();
    let result = generation_outcome(vec![p], 0);
    assert!(
        matches!(result, GenerationPhaseResult::Full(_)),
        "non-empty completed + zero timeouts must be Full"
    );
}

#[test]
fn full_carries_the_proposals() {
    let p1 = make_proposal();
    let p2 = make_proposal();
    let result = generation_outcome(vec![p1, p2], 0);
    if let GenerationPhaseResult::Full(proposals) = result {
        assert_eq!(proposals.len(), 2, "Full must carry all proposals");
    } else {
        panic!("expected Full variant");
    }
}

// ── Partial branch ────────────────────────────────────────────────────────────

#[test]
fn partial_when_some_completed_and_some_timed_out() {
    let p = make_proposal();
    let result = generation_outcome(vec![p], 2);
    assert!(
        matches!(result, GenerationPhaseResult::Partial(_)),
        "non-empty completed + positive timeouts must be Partial"
    );
}

#[test]
fn partial_carries_only_completed_proposals() {
    let p1 = make_proposal();
    let p2 = make_proposal();
    let result = generation_outcome(vec![p1, p2], 1);
    if let GenerationPhaseResult::Partial(proposals) = result {
        assert_eq!(
            proposals.len(),
            2,
            "Partial must carry the completed proposals"
        );
    } else {
        panic!("expected Partial variant");
    }
}

#[test]
fn partial_with_single_timeout_and_multiple_completed() {
    let proposals: Vec<ProposalEvent> = (0..4).map(|_| make_proposal()).collect();
    let result = generation_outcome(proposals, 1);
    assert!(
        matches!(result, GenerationPhaseResult::Partial(_)),
        "one timeout with multiple completed must be Partial, not Full"
    );
}
