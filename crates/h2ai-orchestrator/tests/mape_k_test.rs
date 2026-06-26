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
use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_orchestrator::engine::EngineError;
use h2ai_orchestrator::mape_k::{
    MapeKController, MapeKDecision, PipelineOutcome, PipelineWaveResult, WaveEvents,
};
use h2ai_orchestrator::phases::ExitReason;
use h2ai_types::events::VerificationScoredEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::MultiplicationConditionFailure;

fn default_controller() -> MapeKController {
    MapeKController::new_for_test(H2AIConfig::default())
}

fn empty_wave(outcome: PipelineOutcome) -> PipelineWaveResult {
    PipelineWaveResult {
        outcome,
        events: WaveEvents::default(),
    }
}

fn make_verification_event() -> VerificationScoredEvent {
    VerificationScoredEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        score: 0.8,
        reason: "ok".into(),
        passed: true,
        cache_hit: false,
        passed_checks: None,
        total_checks: None,
        score_lower: None,
        score_upper: None,
        per_check_verdicts: vec![],
        timestamp: Utc::now(),
    }
}

#[tokio::test]
async fn decide_retry_or_fail_on_multiplication_failed() {
    let mut ctrl = default_controller();
    let wave = empty_wave(PipelineOutcome::EarlyExit(
        ExitReason::MultiplicationFailed {
            msg: "test".into(),
            tau_values: vec![0.3, 0.5, 0.7],
            failure: MultiplicationConditionFailure::InsufficientCompetence {
                actual: 0.1,
                required: 0.6,
            },
        },
    ));
    ctrl.observe(&wave, 0).await;
    let decision = ctrl.decide(
        PipelineOutcome::EarlyExit(ExitReason::MultiplicationFailed {
            msg: "test".into(),
            tau_values: vec![0.3, 0.5, 0.7],
            failure: MultiplicationConditionFailure::InsufficientCompetence {
                actual: 0.1,
                required: 0.6,
            },
        }),
        0,
        1.0,
    );
    // Either Retry (policy chose a new topology) or Fail (retries exhausted)
    assert!(matches!(
        decision,
        MapeKDecision::Retry | MapeKDecision::Fail(..)
    ));
}

/// Verify that observe() aggregates verification events across waves.
///
/// Strategy: observe two waves each containing one verification event, then
/// call decide(OracleBlocked) which returns Fail with `partial_verification_events`.
/// Assert the partial events contain exactly the 2 events that were observed.
#[tokio::test]
async fn observe_aggregates_verification_events_across_waves() {
    let mut ctrl = default_controller();

    let mut wave1_events = WaveEvents::default();
    wave1_events
        .verification_events
        .push(make_verification_event());
    ctrl.observe(
        &PipelineWaveResult {
            outcome: PipelineOutcome::EarlyExit(ExitReason::OracleBlocked),
            events: wave1_events,
        },
        0,
    )
    .await;

    let mut wave2_events = WaveEvents::default();
    wave2_events
        .verification_events
        .push(make_verification_event());
    ctrl.observe(
        &PipelineWaveResult {
            outcome: PipelineOutcome::EarlyExit(ExitReason::OracleBlocked),
            events: wave2_events,
        },
        1,
    )
    .await;

    // OracleBlocked always returns Fail with all accumulated verification events.
    let decision = ctrl.decide(
        PipelineOutcome::EarlyExit(ExitReason::OracleBlocked),
        0,
        1.0,
    );
    if let MapeKDecision::Fail(EngineError::MaxRetriesExhausted, ctx) = decision {
        assert_eq!(
            ctx.verification_events.len(),
            2,
            "expected 2 accumulated verification events across 2 waves"
        );
    } else {
        panic!("expected Fail(MaxRetriesExhausted) from OracleBlocked");
    }
}

#[test]
fn wave_events_default_has_none_conflict_rate() {
    let events = WaveEvents::default();
    assert!(events.conflict_rate.is_none());
}

#[test]
fn leader_state_is_none_on_new_controller() {
    let ctrl = default_controller();
    assert!(ctrl.leader.is_none());
}

#[tokio::test]
async fn decide_fail_on_oracle_blocked() {
    let mut ctrl = default_controller();
    let wave = PipelineWaveResult {
        outcome: PipelineOutcome::EarlyExit(ExitReason::OracleBlocked),
        events: WaveEvents::default(),
    };
    ctrl.observe(&wave, 0).await;
    let decision = ctrl.decide(
        PipelineOutcome::EarlyExit(ExitReason::OracleBlocked),
        0,
        1.0,
    );
    assert!(matches!(decision, MapeKDecision::Fail(..)));
}

/// `last_wave_n_eff` starts at 1.0 and is updated after a ZeroSurvival wave
/// that carries an n_eff_cosine value.
#[test]
fn last_wave_n_eff_initialises_to_one() {
    let ctrl = default_controller();
    assert_eq!(ctrl.last_wave_n_eff(), 1.0);
}

#[tokio::test]
async fn last_wave_n_eff_updates_after_zero_survival() {
    use h2ai_orchestrator::coherence::CoherenceState;
    let mut ctrl = default_controller();
    let wave = empty_wave(PipelineOutcome::EarlyExit(ExitReason::ZeroSurvival {
        failure_mode: None,
        coherence: CoherenceState::default(),
        n_eff_cosine: Some(0.3),
        filter_ratio: 1.0,
        tau_values: vec![],
        partial_verification_events: vec![],
    }));
    ctrl.observe(&wave, 0).await;
    // Trigger decide so handle_exit_reason runs and sets last_wave_n_eff.
    let _ = ctrl.decide(
        PipelineOutcome::EarlyExit(ExitReason::ZeroSurvival {
            failure_mode: None,
            coherence: CoherenceState::default(),
            n_eff_cosine: Some(0.3),
            filter_ratio: 1.0,
            tau_values: vec![],
            partial_verification_events: vec![],
        }),
        0,
        1.0,
    );
    assert_eq!(ctrl.last_wave_n_eff(), 0.3);
}

#[tokio::test]
async fn last_wave_n_eff_defaults_one_when_none() {
    use h2ai_orchestrator::coherence::CoherenceState;
    let mut ctrl = default_controller();
    let wave = empty_wave(PipelineOutcome::EarlyExit(ExitReason::ZeroSurvival {
        failure_mode: None,
        coherence: CoherenceState::default(),
        n_eff_cosine: None,
        filter_ratio: 1.0,
        tau_values: vec![],
        partial_verification_events: vec![],
    }));
    ctrl.observe(&wave, 0).await;
    let _ = ctrl.decide(
        PipelineOutcome::EarlyExit(ExitReason::ZeroSurvival {
            failure_mode: None,
            coherence: CoherenceState::default(),
            n_eff_cosine: None,
            filter_ratio: 1.0,
            tau_values: vec![],
            partial_verification_events: vec![],
        }),
        0,
        1.0,
    );
    assert_eq!(ctrl.last_wave_n_eff(), 1.0);
}

#[test]
fn mape_k_decision_complexity_overflow_variant_exists() {
    let d = MapeKDecision::ComplexityOverflow {
        probe_score: 5,
        rationale: "test".into(),
        graft_first: false,
    };
    match d {
        MapeKDecision::ComplexityOverflow {
            probe_score,
            graft_first,
            ..
        } => {
            assert_eq!(probe_score, 5);
            assert!(!graft_first);
        }
        _ => panic!("unexpected variant"),
    }
}

#[test]
fn test_inject_wave_continue() {
    let mut ctrl = default_controller();
    assert!(ctrl.params().retry_context.is_none());

    ctrl.inject_wave_continue(Some("Lua".into()), Some("Prefer atomic ops".into()));
    assert_eq!(
        ctrl.params().retry_context.as_deref(),
        Some("Lua\nMANDATE OVERRIDE: Prefer atomic ops")
    );

    ctrl.inject_wave_continue(Some("Python".into()), None);
    assert_eq!(
        ctrl.params().retry_context.as_deref(),
        Some("Lua\nMANDATE OVERRIDE: Prefer atomic ops\nPython")
    );
}
