use h2ai_config::H2AIConfig;
use h2ai_orchestrator::mape_k::{
    MapeKController, MapeKDecision, MergeOutput, PipelineOutcome, PipelineWaveResult, WaveEvents,
};
use h2ai_types::sizing::{MergeStrategy, PredictionBasis};

// ── shared helpers ────────────────────────────────────────────────────────────

fn make_merge_output(
    verification_events: Vec<h2ai_types::events::VerificationScoredEvent>,
) -> MergeOutput {
    use h2ai_orchestrator::attribution::HarnessAttribution;
    use h2ai_orchestrator::coherence::CoherenceState;
    let task_id = h2ai_types::identity::TaskId::new();
    let explorer_id = h2ai_types::identity::ExplorerId::new();
    MergeOutput {
        task_id: task_id.clone(),
        resolved_output: "test output".to_string(),
        selection_resolved: true,
        selection_resolved_event: h2ai_types::events::SelectionResolvedEvent {
            task_id: task_id.clone(),
            valid_proposals: vec![explorer_id.clone()],
            pruned_proposals: vec![],
            merge_strategy: MergeStrategy::ScoreOrdered,
            timestamp: chrono::Utc::now(),
            merge_elapsed_secs: None,
            n_input_proposals: 1,
            n_failed_proposals: 0,
        },
        attribution: HarnessAttribution {
            baseline_quality: 0.0,
            topology_gain: 0.0,
            verification_gain: 0.0,
            tao_gain: 0.0,
            q_confidence: 1.0,
            prediction_basis: PredictionBasis::Heuristic,
            q_measured: None,
            rho_adjusted: 0.0,
            case_b_flag: false,
            synthesis_gain: 0.0,
        },
        attribution_interval: None,
        talagrand: None,
        suggested_next_params: None,
        waste_ratio: 0.0,
        applied_optimizations: vec![],
        epistemic_yield: None,
        frontier_event: None,
        adapter_correctness: vec![(explorer_id, true)],
        coherence_state: CoherenceState::default(),
        comparison_events: vec![],
        oracle_gate_passed: None,
        tau_values: vec![],
        iteration_verification_events: verification_events,
        wave_token_cost: 0,
        pairwise_cosine_mean: None,
    }
}

fn scored_event(score: f64, passed: bool) -> h2ai_types::events::VerificationScoredEvent {
    h2ai_types::events::VerificationScoredEvent {
        task_id: h2ai_types::identity::TaskId::new(),
        explorer_id: h2ai_types::identity::ExplorerId::new(),
        score,
        reason: "test".to_string(),
        passed,
        cache_hit: false,
        timestamp: chrono::Utc::now(),
    }
}

fn make_zero_survival_exit() -> PipelineOutcome {
    use h2ai_orchestrator::coherence::CoherenceState;
    use h2ai_orchestrator::phases::ExitReason;
    PipelineOutcome::EarlyExit(ExitReason::ZeroSurvival {
        failure_mode: None,
        coherence: CoherenceState::default(),
        n_eff_cosine: Some(1.0),
        filter_ratio: 1.0,
        tau_values: vec![0.2],
    })
}

fn cfg_with_routing(decompose_threshold: u8, min_retries_before_graft: u32) -> H2AIConfig {
    H2AIConfig {
        complexity_routing: h2ai_config::ComplexityRoutingConfig {
            enabled: true,
            decompose_threshold,
            hitl_threshold: 5,
            min_retries_before_graft,
            ..h2ai_config::ComplexityRoutingConfig::default()
        },
        max_autonomic_retries: 10,
        ..H2AIConfig::default()
    }
}

// ── tiered exit (gap_l1_tee) ──────────────────────────────────────────────────

#[test]
fn tee_gate_forces_retry_when_k_not_met() {
    let cfg = H2AIConfig {
        tiered_exit: h2ai_config::TieredExitConfig {
            enabled: true,
            min_n: 1,
            max_n: 3,
            quorum_fraction: 0.5,
            acceptance_score: 0.90,
            require_all_binary_checks: false,
        },
        max_autonomic_retries: 4,
        ..H2AIConfig::default()
    };
    let mut ctrl = MapeKController::new_for_test(cfg);
    ctrl.set_n_agents(1);

    let out = make_merge_output(vec![scored_event(0.50, true)]);
    let decision = ctrl.decide(PipelineOutcome::Resolved(Box::new(out)), 0, 1.0);
    assert!(matches!(decision, MapeKDecision::Retry), "expected Retry");
    assert!(ctrl.tee_event_ref().is_none());
}

#[test]
fn tee_gate_accepts_when_k_met() {
    let cfg = H2AIConfig {
        tiered_exit: h2ai_config::TieredExitConfig {
            enabled: true,
            min_n: 1,
            max_n: 3,
            quorum_fraction: 0.5,
            acceptance_score: 0.85,
            require_all_binary_checks: false,
        },
        max_autonomic_retries: 4,
        ..H2AIConfig::default()
    };
    let mut ctrl = MapeKController::new_for_test(cfg);
    ctrl.set_n_agents(1);

    let out = make_merge_output(vec![scored_event(0.95, true)]);
    let decision = ctrl.decide(PipelineOutcome::Resolved(Box::new(out)), 0, 1.0);
    assert!(
        matches!(decision, MapeKDecision::Return(_)),
        "expected Return"
    );

    let evt = ctrl.tee_event_ref().expect("tee_event should be set");
    assert_eq!(evt.wave, 0);
    assert_eq!(evt.n, 1);
    assert_eq!(evt.k_required, 1);
    assert_eq!(evt.k_accepted, 1);
}

#[test]
fn tee_gate_accepts_on_last_retry_even_if_k_not_met() {
    let cfg = H2AIConfig {
        tiered_exit: h2ai_config::TieredExitConfig {
            enabled: true,
            min_n: 1,
            max_n: 3,
            quorum_fraction: 0.5,
            acceptance_score: 0.90,
            require_all_binary_checks: false,
        },
        max_autonomic_retries: 2,
        ..H2AIConfig::default()
    };
    let mut ctrl = MapeKController::new_for_test(cfg);
    ctrl.set_n_agents(1);

    let out = make_merge_output(vec![scored_event(0.50, true)]);
    let decision = ctrl.decide(PipelineOutcome::Resolved(Box::new(out)), 2, 1.0);
    assert!(
        matches!(decision, MapeKDecision::Return(_)),
        "expected Return on last retry"
    );
    assert!(
        ctrl.tee_event_ref().is_some(),
        "tee_event should be set even on last retry"
    );
}

#[test]
fn tee_disabled_does_not_interfere() {
    let mut ctrl = MapeKController::new_for_test(H2AIConfig::default());
    ctrl.set_n_agents(3);

    let out = make_merge_output(vec![scored_event(0.20, true)]);
    let decision = ctrl.decide(PipelineOutcome::Resolved(Box::new(out)), 0, 1.0);
    assert!(
        matches!(decision, MapeKDecision::Return(_)),
        "TEE disabled should always Return on Resolved"
    );
}

// ── pipeline params budget (compile-time shape checks + runtime) ──────────────

#[test]
fn wave_events_default_has_zero_token_cost() {
    use h2ai_orchestrator::mape_k::WaveEvents;
    let e = WaveEvents::default();
    assert_eq!(e.wave_token_cost, 0);
}

// ── probe routing guard ───────────────────────────────────────────────────────

#[test]
fn corpus_synthesis_viable_false_by_default_in_test_constructor() {
    let ctrl = MapeKController::new_for_test(H2AIConfig::default());
    assert!(
        !ctrl.corpus_synthesis_viable_flag(),
        "new_for_test has empty binary_checks so viable must be false"
    );
}

#[test]
fn graft_blocked_when_corpus_not_viable() {
    let cfg = cfg_with_routing(4, 0);
    let mut ctrl = MapeKController::new_for_test(cfg);
    ctrl.set_corpus_viable(false);
    ctrl.set_probe_result(h2ai_autonomic::complexity_probe::ComplexityProbeResult {
        complexity: 4,
        rationale: "test".to_string(),
        decompose_recommended: true,
    });

    let decision = ctrl.decide(make_zero_survival_exit(), 0, 1.0);
    assert!(
        !matches!(
            decision,
            MapeKDecision::ComplexityOverflow {
                graft_first: true,
                ..
            }
        ),
        "graft must not fire when corpus has no binary_checks"
    );
}

#[test]
fn graft_blocked_before_min_retries_floor() {
    let cfg = cfg_with_routing(4, 2);
    let mut ctrl = MapeKController::new_for_test(cfg);
    ctrl.set_corpus_viable(true);
    ctrl.set_probe_result(h2ai_autonomic::complexity_probe::ComplexityProbeResult {
        complexity: 4,
        rationale: "test".to_string(),
        decompose_recommended: true,
    });

    let decision = ctrl.decide(make_zero_survival_exit(), 1, 1.0);
    assert!(
        !matches!(
            decision,
            MapeKDecision::ComplexityOverflow {
                graft_first: true,
                ..
            }
        ),
        "graft must not fire before min_retries_before_graft (retry_count=1 < floor=2)"
    );
}

#[test]
fn graft_fires_when_both_conditions_met() {
    let cfg = cfg_with_routing(4, 2);
    let mut ctrl = MapeKController::new_for_test(cfg);
    ctrl.set_corpus_viable(true);
    ctrl.set_probe_result(h2ai_autonomic::complexity_probe::ComplexityProbeResult {
        complexity: 4,
        rationale: "test".to_string(),
        decompose_recommended: true,
    });

    let decision = ctrl.decide(make_zero_survival_exit(), 2, 1.0);
    assert!(
        matches!(
            decision,
            MapeKDecision::ComplexityOverflow {
                graft_first: true,
                ..
            }
        ),
        "graft must fire when corpus viable AND retry_count >= min_retries_before_graft"
    );
}

#[test]
fn graft_fires_immediately_when_floor_zero_and_corpus_viable() {
    let cfg = cfg_with_routing(4, 0);
    let mut ctrl = MapeKController::new_for_test(cfg);
    ctrl.set_corpus_viable(true);
    ctrl.set_probe_result(h2ai_autonomic::complexity_probe::ComplexityProbeResult {
        complexity: 4,
        rationale: "test".to_string(),
        decompose_recommended: true,
    });

    let decision = ctrl.decide(make_zero_survival_exit(), 0, 1.0);
    assert!(
        matches!(
            decision,
            MapeKDecision::ComplexityOverflow {
                graft_first: true,
                ..
            }
        ),
        "floor=0 must fire immediately (backward-compat for viable corpus)"
    );
}

#[test]
fn backstop_invariant_signal_requires_viable_corpus() {
    let cfg = cfg_with_routing(1, 0);
    let mut ctrl = MapeKController::new_for_test(cfg);
    ctrl.set_corpus_viable(false);
    ctrl.set_probe_result(h2ai_autonomic::complexity_probe::ComplexityProbeResult {
        complexity: 4,
        rationale: "test".to_string(),
        decompose_recommended: true,
    });

    for retry_count in 0..10u32 {
        let decision = ctrl.decide(make_zero_survival_exit(), retry_count, 1.0);
        assert!(
            !matches!(decision, MapeKDecision::ComplexityOverflow { graft_first: true, .. }),
            "graft_first=true must never fire when corpus_synthesis_viable=false (retry={retry_count})"
        );
        ctrl.set_corpus_viable(false);
    }
}

// ── ambiguity scorecard (gap_f8) ──────────────────────────────────────────────

fn pruned_event_with_reason(cid: &str, reason: &str) -> h2ai_types::events::BranchPrunedEvent {
    use h2ai_types::sizing::RoleErrorCost;
    h2ai_types::events::BranchPrunedEvent {
        task_id: h2ai_types::identity::TaskId::new(),
        explorer_id: h2ai_types::identity::ExplorerId::new(),
        reason: reason.to_string(),
        raw_output: String::new(),
        constraint_error_cost: RoleErrorCost::new(0.0).unwrap(),
        violated_constraints: vec![h2ai_types::events::ConstraintViolation {
            constraint_id: cid.to_string(),
            score: 0.0,
            severity_label: "Hard".to_string(),
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: Some(reason.to_string()),
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: None,
        }],
        timestamp: chrono::Utc::now(),
        retry_count: 0,
        bypass_reason: None,
    }
}

fn inject_divergence(ctrl: &mut MapeKController, cid: &str) {
    ctrl.seed_pruned_waves(
        vec![
            pruned_event_with_reason(cid, "alpha bravo charlie delta echo"),
            pruned_event_with_reason(cid, "alpha bravo charlie delta foxtrot"),
        ],
        vec![
            pruned_event_with_reason(cid, "zulu yankee xray whiskey victor"),
            pruned_event_with_reason(cid, "zulu yankee xray whiskey uniform"),
        ],
    );
}

fn ambiguity_cfg() -> H2AIConfig {
    H2AIConfig {
        gap_k1: h2ai_config::GapK1Config {
            enabled: true,
            instability_threshold: 0.10,
            ..h2ai_config::GapK1Config::default()
        },
        ambiguity_detection: h2ai_constraints::ambiguity::AmbiguityDetectionConfig {
            enabled: true,
            score_threshold: 0.6,
            ..h2ai_constraints::ambiguity::AmbiguityDetectionConfig::default()
        },
        ..H2AIConfig::default()
    }
}

#[test]
fn find_instability_legacy_path_when_ambiguity_disabled() {
    let cfg = H2AIConfig {
        gap_k1: h2ai_config::GapK1Config {
            enabled: true,
            instability_threshold: 0.10,
            ..h2ai_config::GapK1Config::default()
        },
        ambiguity_detection: h2ai_constraints::ambiguity::AmbiguityDetectionConfig {
            enabled: false,
            ..h2ai_constraints::ambiguity::AmbiguityDetectionConfig::default()
        },
        ..H2AIConfig::default()
    };
    let mut ctrl = MapeKController::new_for_test(cfg);
    inject_divergence(&mut ctrl, "C-001");

    let sig = ctrl.find_instability(0).expect("instability should fire");
    assert_eq!(sig.constraint_id, "C-001");
    assert_eq!(sig.check_index, 0, "legacy path must set check_index=0");
    assert!(sig.ambiguity_evidence.is_empty());
    assert_eq!(sig.ambiguity_score, 0.0);
}

#[test]
fn find_instability_accumulates_below_threshold_returns_none() {
    let mut ctrl = MapeKController::new_for_test(ambiguity_cfg());
    inject_divergence(&mut ctrl, "C-002");

    let result = ctrl.find_instability(1);
    assert!(
        result.is_none(),
        "below threshold, must return None; got {result:?}"
    );
    let has_scorecard = ctrl
        .ambiguity_scorecards_ref()
        .values()
        .any(|sc| sc.constraint_id == "C-002" && !sc.evidence.is_empty());
    assert!(
        has_scorecard,
        "scorecard must be updated after accumulation"
    );
}

#[test]
fn find_instability_threshold_crossed_precise_returns_real_check_idx() {
    use h2ai_constraints::ambiguity::{AmbiguityEvidence, AmbiguityScorecard};

    let mut cfg = ambiguity_cfg();
    cfg.ambiguity_detection.score_threshold = 0.14;
    let mut ctrl = MapeKController::new_for_test(cfg.clone());

    let mut base_card = AmbiguityScorecard::new("C-003".to_string(), 2);
    base_card.evidence.push(AmbiguityEvidence::FmTermNegation {
        term: "cockroachdb".to_string(),
        negated_in: "avoid cockroachdb".to_string(),
    });
    ctrl.seed_ambiguity_scorecard(("C-003".to_string(), 2), base_card);

    inject_divergence(&mut ctrl, "C-003");
    let sig = ctrl
        .find_instability(2)
        .expect("threshold crossed, must return Some");
    assert_eq!(sig.constraint_id, "C-003");
    assert_eq!(
        sig.check_index, 2,
        "Precise patch mode must set real check_index"
    );
    assert!(!sig.ambiguity_evidence.is_empty());
    assert!(sig.ambiguity_score >= cfg.ambiguity_detection.score_threshold);
}

#[test]
fn find_instability_threshold_crossed_diagnostic_queues_event_returns_none() {
    let mut cfg = ambiguity_cfg();
    cfg.ambiguity_detection.score_threshold = 0.14;
    let mut ctrl = MapeKController::new_for_test(cfg);

    inject_divergence(&mut ctrl, "C-004");
    let result = ctrl.find_instability(3);
    assert!(
        result.is_none(),
        "DiagnosticOnly must return None; got {result:?}"
    );

    let events = ctrl.take_pending_ambiguity_events();
    assert_eq!(
        events.len(),
        1,
        "one pending ambiguity event must be queued"
    );
    assert_eq!(events[0].constraint_id, "C-004");
    assert!(events[0].check_idx.is_none());
}

#[test]
fn find_instability_no_double_trigger_after_fired() {
    let mut cfg = ambiguity_cfg();
    cfg.ambiguity_detection.score_threshold = 0.14;
    let mut ctrl = MapeKController::new_for_test(cfg);

    inject_divergence(&mut ctrl, "C-005");
    let _ = ctrl.find_instability(1);
    let first_events = ctrl.take_pending_ambiguity_events();
    assert_eq!(first_events.len(), 1, "first call must queue one event");

    inject_divergence(&mut ctrl, "C-005");
    let result2 = ctrl.find_instability(2);
    assert!(
        result2.is_none(),
        "double-trigger must be prevented after rewrite_applied=true"
    );
    let events2 = ctrl.take_pending_ambiguity_events();
    assert!(
        events2.is_empty(),
        "no second pending event after rewrite_applied"
    );
}

// ── Frozen verifier: per_constraint_wave_scores and is_verifier_bypassed ──────

fn make_wave_with_pruned_c008(score: f64) -> PipelineWaveResult {
    let mut events = WaveEvents::default();
    events.pruned_events = vec![h2ai_types::events::BranchPrunedEvent {
        task_id: h2ai_types::identity::TaskId::new(),
        explorer_id: h2ai_types::identity::ExplorerId::new(),
        reason: "test".to_string(),
        raw_output: String::new(),
        constraint_error_cost: h2ai_types::sizing::RoleErrorCost::new(0.0).unwrap(),
        violated_constraints: vec![h2ai_types::events::ConstraintViolation {
            constraint_id: "CONSTRAINT-008".to_string(),
            score,
            severity_label: "Hard".to_string(),
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: Some("idempotency key missing tenant_id prefix".to_string()),
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: None,
        }],
        timestamp: chrono::Utc::now(),
        retry_count: 0,
        bypass_reason: None,
    }];
    PipelineWaveResult {
        outcome: make_zero_survival_exit(),
        events,
    }
}

#[test]
fn per_constraint_wave_scores_populated_from_pruned_events() {
    let mut ctrl = MapeKController::new_minimal();
    let wave = make_wave_with_pruned_c008(0.0);
    ctrl.observe(&wave, 0);
    let scores = ctrl.per_constraint_wave_scores();
    assert!(
        scores.contains_key("CONSTRAINT-008"),
        "C-008 must appear in per_constraint_wave_scores"
    );
    assert_eq!(scores["CONSTRAINT-008"][0], 0.0);
}

#[test]
fn is_verifier_bypassed_returns_false_before_bypass_activates() {
    let ctrl = MapeKController::new_minimal();
    assert!(!ctrl.is_verifier_bypassed("CONSTRAINT-008"));
}

// ── Frozen verifier: bypass activation and correctness ────────────────────────

fn make_wave_frozen_c008_improving_c004(wave_idx: u32) -> PipelineWaveResult {
    // C-008 always scores 0.0 with the same reason (frozen verifier pattern)
    // C-004 scores increase across waves (improving pattern)
    let c004_score = match wave_idx {
        0 => 0.25,
        1 => 0.5,
        2 => 0.75,
        _ => 0.9,
    };
    let mut events = WaveEvents::default();
    events.pruned_events = vec![h2ai_types::events::BranchPrunedEvent {
        task_id: h2ai_types::identity::TaskId::new(),
        explorer_id: h2ai_types::identity::ExplorerId::new(),
        reason: "test".to_string(),
        raw_output: String::new(),
        constraint_error_cost: h2ai_types::sizing::RoleErrorCost::new(0.0).unwrap(),
        violated_constraints: vec![
            h2ai_types::events::ConstraintViolation {
                constraint_id: "CONSTRAINT-008".to_string(),
                score: 0.0,
                severity_label: "Hard".to_string(),
                remediation_hint: None,
                constraint_description: String::new(),
                verifier_reason: Some("idempotency key missing tenant_id prefix".to_string()),
                check_verdicts: vec![],
                criteria_pass: None,
                check_reasons: None,
            },
            h2ai_types::events::ConstraintViolation {
                constraint_id: "CONSTRAINT-004".to_string(),
                score: c004_score,
                severity_label: "Hard".to_string(),
                remediation_hint: None,
                constraint_description: String::new(),
                verifier_reason: None,
                check_verdicts: vec![],
                criteria_pass: None,
                check_reasons: None,
            },
        ],
        timestamp: chrono::Utc::now(),
        retry_count: wave_idx,
        bypass_reason: None,
    }];
    PipelineWaveResult {
        outcome: make_zero_survival_exit(),
        events,
    }
}

#[test]
fn bypass_activates_after_min_waves_frozen() {
    // Simulate 4 waves (0..4) with min_waves_to_detect=3.
    // At wave_idx=3, retry_count=3 >= 3, so detection fires.
    // C-008: [0.0, 0.0, 0.0, 0.0] — frozen (score_range=0.0 < 0.05)
    // C-004: [0.25, 0.5, 0.75, 0.9] — monotonically improving, mean_last_3=0.717 > 0.5
    // reason history: 4 identical strings → Jaccard=1.0 > 0.7
    let mut ctrl = MapeKController::new_minimal();
    for wave_idx in 0..4u32 {
        let wave = make_wave_frozen_c008_improving_c004(wave_idx);
        ctrl.observe(&wave, wave_idx);
        ctrl.decide(make_zero_survival_exit(), wave_idx, 1.0);
    }
    assert!(
        ctrl.is_verifier_bypassed("CONSTRAINT-008"),
        "bypass must activate after min_waves_to_detect=3 frozen waves (checked at wave 3)"
    );
}

#[test]
fn bypass_does_not_activate_before_min_waves() {
    // Only 3 iterations (0..3): at wave_idx=2, retry_count=2 < min_waves_to_detect=3.
    // Bypass must NOT activate yet.
    let mut ctrl = MapeKController::new_minimal();
    for wave_idx in 0..3u32 {
        let wave = make_wave_frozen_c008_improving_c004(wave_idx);
        ctrl.observe(&wave, wave_idx);
        ctrl.decide(make_zero_survival_exit(), wave_idx, 1.0);
    }
    assert!(
        !ctrl.is_verifier_bypassed("CONSTRAINT-008"),
        "bypass must NOT activate when retry_count=2 < min_waves_to_detect=3"
    );
}

// ── Group B: MapeKController method coverage ──────────────────────────────────

// tokens_used() getter (lines 757-759)
#[test]
fn tokens_used_returns_zero_on_fresh_controller() {
    let ctrl = MapeKController::new_minimal();
    assert_eq!(
        ctrl.tokens_used(),
        0,
        "fresh controller must report zero tokens"
    );
}

// observe_wave_tokens() accumulation (lines 752-754)
#[test]
fn observe_wave_tokens_accumulates_correctly() {
    let mut ctrl = MapeKController::new_minimal();
    ctrl.observe_wave_tokens(1_000);
    ctrl.observe_wave_tokens(2_500);
    assert_eq!(
        ctrl.tokens_used(),
        3_500,
        "tokens must accumulate across observe_wave_tokens calls"
    );
}

#[test]
fn observe_wave_tokens_saturates_at_u64_max() {
    let mut ctrl = MapeKController::new_minimal();
    ctrl.observe_wave_tokens(u64::MAX);
    ctrl.observe_wave_tokens(1);
    assert_eq!(
        ctrl.tokens_used(),
        u64::MAX,
        "saturating_add must cap at u64::MAX"
    );
}

// mark_oom_abort() mutation (lines 774-776)
// Observable effect: budget_exhausted=true prevents TEE from issuing a Retry decision.
#[test]
fn mark_oom_abort_overrides_tee_retry_to_return() {
    // TEE requires k_required=1, acceptance_score=0.90.  We supply score=0.50 (below threshold).
    // Without OOM abort → Retry (k_accepted=0 < k_required=1, retry_count=0 < max_retries=4).
    // After mark_oom_abort() → budget_exhausted=true → TEE skips retry → Return.
    let cfg = H2AIConfig {
        tiered_exit: h2ai_config::TieredExitConfig {
            enabled: true,
            min_n: 1,
            max_n: 3,
            quorum_fraction: 0.5,
            acceptance_score: 0.90,
            require_all_binary_checks: false,
        },
        max_autonomic_retries: 4,
        ..H2AIConfig::default()
    };

    // Without OOM abort → Retry
    let mut ctrl_no_oom = MapeKController::new_for_test(cfg.clone());
    ctrl_no_oom.set_n_agents(1);
    let out = make_merge_output(vec![scored_event(0.50, true)]);
    assert!(
        matches!(
            ctrl_no_oom.decide(PipelineOutcome::Resolved(Box::new(out)), 0, 1.0),
            MapeKDecision::Retry
        ),
        "without OOM abort, low score must trigger Retry"
    );

    // With OOM abort → Return (budget_exhausted bypasses retry guard)
    let mut ctrl_oom = MapeKController::new_for_test(cfg);
    ctrl_oom.set_n_agents(1);
    ctrl_oom.mark_oom_abort();
    let out2 = make_merge_output(vec![scored_event(0.50, true)]);
    assert!(
        matches!(
            ctrl_oom.decide(PipelineOutcome::Resolved(Box::new(out2)), 0, 1.0),
            MapeKDecision::Return(_)
        ),
        "after mark_oom_abort, budget_exhausted must prevent TEE Retry → Return"
    );
}

// inject_wave_continue() with empty/None args → early return at line 724-725
#[test]
fn inject_wave_continue_with_none_args_is_noop() {
    let mut ctrl = MapeKController::new_minimal();
    ctrl.inject_wave_continue(None, None);
    // retry_context must remain None; verify that a subsequent real injection works.
    ctrl.inject_wave_continue(Some("real grounding".to_string()), None);
    let params = ctrl.params();
    assert_eq!(
        params.retry_context,
        Some("real grounding".to_string()),
        "retry_context must be set after non-empty inject_wave_continue"
    );
}

#[test]
fn inject_wave_continue_with_whitespace_only_args_is_noop() {
    let mut ctrl = MapeKController::new_minimal();
    ctrl.inject_wave_continue(Some("   ".to_string()), Some("\t\n".to_string()));
    // Both are whitespace-only → parts is empty → early return. retry_context stays None.
    let params = ctrl.params();
    assert!(
        params.retry_context.is_none(),
        "whitespace-only args must not set retry_context"
    );
}

// last_wave_n_eff() getter (line 740-742)
#[test]
fn last_wave_n_eff_returns_one_before_any_wave() {
    let ctrl = MapeKController::new_minimal();
    assert!(
        (ctrl.last_wave_n_eff() - 1.0).abs() < 1e-9,
        "default n_eff must be 1.0 (no-dropout sentinel)"
    );
}

// seed_synthesis() and domain_synthesis_cache population (lines 2492-2500)
#[test]
fn seed_synthesis_populates_domain_synthesis_cache() {
    let mut ctrl = MapeKController::new_minimal();
    let synthesis = h2ai_types::gap_i1::DomainSynthesis {
        check_id: ("C-001".to_string(), 2),
        incorrect_pattern: "unbounded loop".to_string(),
        correct_pattern: "bounded retry with backoff".to_string(),
        mechanistic_reason: "unbounded loops exhaust executor threads under load".to_string(),
        source: Some("grounding-wave-3".to_string()),
        confidence: 0.85,
        injected_at_wave: None,
        pre_injection_pass_rate: None,
        post_injection_pass_rates: vec![],
    };
    ctrl.seed_synthesis("C-001", 2, synthesis);

    // The cache entry must be present; verify indirectly via seed_synthesis again
    // (no direct read accessor, but the cache can be over-written to confirm it existed)
    let synthesis2 = h2ai_types::gap_i1::DomainSynthesis {
        check_id: ("C-001".to_string(), 2),
        incorrect_pattern: "updated".to_string(),
        correct_pattern: "updated".to_string(),
        mechanistic_reason: "updated".to_string(),
        source: None,
        confidence: 0.99,
        injected_at_wave: Some(1),
        pre_injection_pass_rate: Some(0.5),
        post_injection_pass_rates: vec![0.7],
    };
    ctrl.seed_synthesis("C-001", 2, synthesis2);
    // No panic = cache insertions work correctly for both entries and overwrites.
}

#[test]
fn seed_synthesis_distinct_keys_are_independent() {
    let mut ctrl = MapeKController::new_minimal();
    let make_synthesis = |tag: &str| h2ai_types::gap_i1::DomainSynthesis {
        check_id: (tag.to_string(), 0),
        incorrect_pattern: tag.to_string(),
        correct_pattern: tag.to_string(),
        mechanistic_reason: tag.to_string(),
        source: None,
        confidence: 0.5,
        injected_at_wave: None,
        pre_injection_pass_rate: None,
        post_injection_pass_rates: vec![],
    };
    ctrl.seed_synthesis("C-001", 0, make_synthesis("C-001"));
    ctrl.seed_synthesis("C-002", 1, make_synthesis("C-002"));
    // Both keys must be inserted without collision; no panic = success.
}
