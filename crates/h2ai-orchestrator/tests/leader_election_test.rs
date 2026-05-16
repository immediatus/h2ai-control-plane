//! Integration tests for epistemic leader election behavior.

use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_orchestrator::mape_k::{MapeKController, PipelineOutcome, PipelineWaveResult, WaveEvents};
use h2ai_orchestrator::phases::ExitReason;
use h2ai_types::events::VerificationScoredEvent;
use h2ai_types::identity::{ExplorerId, TaskId};

fn make_verification_event(explorer_id: ExplorerId, score: f64) -> VerificationScoredEvent {
    VerificationScoredEvent {
        task_id: TaskId::new(),
        explorer_id,
        score,
        reason: String::from("test reason"),
        passed: score >= 0.45,
        cache_hit: false,
        timestamp: Utc::now(),
    }
}

fn failed_wave_with_scores(scores: Vec<(ExplorerId, f64)>) -> PipelineWaveResult {
    let verification_events = scores
        .iter()
        .map(|(id, s)| make_verification_event(id.clone(), *s))
        .collect();
    let wave_proposal_texts = scores
        .iter()
        .map(|(id, _)| (id.clone(), format!("proposal by {id}")))
        .collect();
    let events = WaveEvents {
        verification_events,
        wave_proposal_texts,
        ..WaveEvents::default()
    };
    PipelineWaveResult {
        outcome: PipelineOutcome::EarlyExit(ExitReason::ZeroSurvival {
            failure_mode: None,
            coherence: h2ai_orchestrator::coherence::CoherenceState::default(),
            n_eff_cosine: None,
            filter_ratio: 0.0,
            tau_values: vec![],
        }),
        events,
    }
}

#[test]
fn leader_none_before_first_failed_wave() {
    let ctrl = MapeKController::new_for_test(H2AIConfig::default());
    assert!(ctrl.leader.is_none());
}

#[test]
fn prepare_leader_election_returns_none_when_no_verification_scores() {
    let ctrl = MapeKController::new_for_test(H2AIConfig::default());
    let cfg = H2AIConfig {
        leader_enabled: true,
        ..H2AIConfig::default()
    };
    let plan = ctrl.prepare_leader_election(&cfg);
    assert!(plan.is_none());
}

#[test]
fn prepare_leader_election_picks_highest_score_as_leader() {
    let mut ctrl = MapeKController::new_for_test(H2AIConfig::default());
    let id_a = ExplorerId::new();
    let id_b = ExplorerId::new();
    let wave = failed_wave_with_scores(vec![(id_a.clone(), 0.3), (id_b.clone(), 0.7)]);
    ctrl.observe(&wave);

    let cfg = H2AIConfig {
        leader_enabled: true,
        ..H2AIConfig::default()
    };
    let plan = ctrl.prepare_leader_election(&cfg).unwrap();
    assert_eq!(plan.leader_explorer_id, id_b);
    assert_eq!(plan.runner_up_explorer_id, Some(id_a));
}

#[test]
fn apply_leader_result_populates_leader_state() {
    let mut ctrl = MapeKController::new_for_test(H2AIConfig::default());
    let id_a = ExplorerId::new();
    let wave = failed_wave_with_scores(vec![(id_a.clone(), 0.6)]);
    ctrl.observe(&wave);

    let cfg = H2AIConfig {
        leader_enabled: true,
        ..H2AIConfig::default()
    };
    let plan = ctrl.prepare_leader_election(&cfg).unwrap();
    ctrl.apply_leader_result(plan, "Why did this fail?".into(), 1, 0, &cfg);

    let ls = ctrl.leader.as_ref().unwrap();
    assert_eq!(ls.term, 1);
    assert_eq!(ls.socratic_question, "Why did this fail?");
    assert_eq!(ls.credibility_score, 1.0);
    assert_eq!(ls.belief_buffer.len(), 1);
}

#[test]
fn leader_events_buffered_after_apply() {
    let mut ctrl = MapeKController::new_for_test(H2AIConfig::default());
    let id_a = ExplorerId::new();
    let wave = failed_wave_with_scores(vec![(id_a.clone(), 0.6)]);
    ctrl.observe(&wave);

    let cfg = H2AIConfig {
        leader_enabled: true,
        ..H2AIConfig::default()
    };
    let plan = ctrl.prepare_leader_election(&cfg).unwrap();
    ctrl.apply_leader_result(plan, "Socratic question?".into(), 1, 0, &cfg);

    let (elected_evs, diag_evs) = ctrl.take_leader_events();
    assert_eq!(elected_evs.len(), 1);
    assert_eq!(diag_evs.len(), 1);
    assert_eq!(diag_evs[0].question, "Socratic question?");
}

#[test]
fn stagnation_count_increments_on_flat_confidence() {
    let mut ctrl = MapeKController::new_for_test(H2AIConfig::default());
    let id_a = ExplorerId::new();
    let cfg = H2AIConfig {
        leader_enabled: true,
        ..H2AIConfig::default()
    };

    // Wave 1
    let wave = failed_wave_with_scores(vec![(id_a.clone(), 0.5)]);
    ctrl.observe(&wave);
    let plan = ctrl.prepare_leader_election(&cfg).unwrap();
    ctrl.apply_leader_result(plan, "Q1?".into(), 1, 0, &cfg);

    // Wave 2 — same score (no improvement)
    let wave2 = failed_wave_with_scores(vec![(id_a.clone(), 0.5)]);
    ctrl.observe(&wave2);
    let plan2 = ctrl.prepare_leader_election(&cfg).unwrap();
    ctrl.apply_leader_result(plan2, "Q2?".into(), 1, 0, &cfg);

    let ls = ctrl.leader.as_ref().unwrap();
    assert_eq!(ls.stagnation_count, 2);
}
