#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
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
        passed_checks: None,
        total_checks: None,
        score_lower: None,
        score_upper: None,
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

#[tokio::test]
async fn prepare_leader_election_picks_highest_score_as_leader() {
    let mut ctrl = MapeKController::new_for_test(H2AIConfig::default());
    let id_a = ExplorerId::new();
    let id_b = ExplorerId::new();
    let wave = failed_wave_with_scores(vec![(id_a.clone(), 0.3), (id_b.clone(), 0.7)]);
    ctrl.observe(&wave, 0).await;

    let cfg = H2AIConfig {
        leader_enabled: true,
        ..H2AIConfig::default()
    };
    let plan = ctrl.prepare_leader_election(&cfg).unwrap();
    assert_eq!(plan.leader_explorer_id, id_b);
    assert_eq!(plan.runner_up_explorer_id, Some(id_a));
}

#[tokio::test]
async fn apply_leader_result_populates_leader_state() {
    let mut ctrl = MapeKController::new_for_test(H2AIConfig::default());
    let id_a = ExplorerId::new();
    let wave = failed_wave_with_scores(vec![(id_a.clone(), 0.6)]);
    ctrl.observe(&wave, 0).await;

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

#[tokio::test]
async fn leader_events_buffered_after_apply() {
    let mut ctrl = MapeKController::new_for_test(H2AIConfig::default());
    let id_a = ExplorerId::new();
    let wave = failed_wave_with_scores(vec![(id_a.clone(), 0.6)]);
    ctrl.observe(&wave, 0).await;

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

#[tokio::test]
async fn stagnation_count_increments_on_flat_confidence() {
    let mut ctrl = MapeKController::new_for_test(H2AIConfig::default());
    let id_a = ExplorerId::new();
    let cfg = H2AIConfig {
        leader_enabled: true,
        ..H2AIConfig::default()
    };

    // Wave 1
    let wave = failed_wave_with_scores(vec![(id_a.clone(), 0.5)]);
    ctrl.observe(&wave, 0).await;
    let plan = ctrl.prepare_leader_election(&cfg).unwrap();
    ctrl.apply_leader_result(plan, "Q1?".into(), 1, 0, &cfg);

    // Wave 2 — same score (no improvement)
    let wave2 = failed_wave_with_scores(vec![(id_a.clone(), 0.5)]);
    ctrl.observe(&wave2, 1).await;
    let plan2 = ctrl.prepare_leader_election(&cfg).unwrap();
    ctrl.apply_leader_result(plan2, "Q2?".into(), 1, 0, &cfg);

    let ls = ctrl.leader.as_ref().unwrap();
    assert_eq!(ls.stagnation_count, 2);
}
