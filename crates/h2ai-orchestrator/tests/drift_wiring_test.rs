use h2ai_orchestrator::engine::consensus_agreement_rate_from_events;
use h2ai_types::events::VerificationScoredEvent;
use h2ai_types::identity::{ExplorerId, TaskId};

fn make_event(passed: bool) -> VerificationScoredEvent {
    VerificationScoredEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        score: if passed { 0.8 } else { 0.3 },
        reason: String::new(),
        passed,
        cache_hit: false,
        passed_checks: None,
        total_checks: None,
        score_lower: None,
        score_upper: None,
        timestamp: chrono::Utc::now(),
    }
}

#[test]
fn rate_is_1_when_all_pass() {
    let events = vec![make_event(true), make_event(true), make_event(true)];
    assert!((consensus_agreement_rate_from_events(&events) - 1.0).abs() < 1e-9);
}

#[test]
fn rate_is_0_when_all_fail() {
    let events = vec![make_event(false), make_event(false)];
    assert!((consensus_agreement_rate_from_events(&events) - 0.0).abs() < 1e-9);
}

#[test]
fn rate_is_half_when_split() {
    let events = vec![make_event(true), make_event(false)];
    assert!((consensus_agreement_rate_from_events(&events) - 0.5).abs() < 1e-9);
}

#[test]
fn rate_is_1_when_no_events() {
    assert!((consensus_agreement_rate_from_events(&[]) - 1.0).abs() < 1e-9);
}
