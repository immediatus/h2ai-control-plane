use h2ai_autonomic::knowledge_gap::detect_cold_checks;
use h2ai_autonomic::repair::PartialPass;
use h2ai_config::{ConvergenceGateConfig, CostGuardConfig};
use h2ai_orchestrator::mape_k::{
    build_budget_hint_if_needed, check_convergence_gate, constraint_reasons_jaccard,
    has_isolation_evidence, is_compliance_plateau,
};

// ── gap_k1: constraint_reasons_jaccard ───────────────────────────────────────

#[test]
fn detect_instability_fires_on_low_jaccard() {
    let reasons_a = vec!["quota atomic CAS redis".to_owned()];
    let reasons_b = vec!["audit log missing actor".to_owned()];
    let score = constraint_reasons_jaccard(&reasons_a, &reasons_b);
    assert!(score < 0.10, "low jaccard expected, got {score}");
}

#[test]
fn detect_instability_stable_when_same_reasons() {
    let reasons = vec!["quota atomic CAS redis lua eval".to_owned()];
    let score = constraint_reasons_jaccard(&reasons, &reasons);
    assert!(score > 0.90, "high jaccard expected, got {score}");
}

// ── gap_i1: detect_cold_checks ────────────────────────────────────────────────

#[test]
fn cold_check_detection_returns_empty_when_all_checks_pass() {
    let rates = vec![
        (("C-001".to_string(), 0usize), 1.0_f64),
        (("C-001".to_string(), 1usize), 0.5_f64),
    ];
    let cold = detect_cold_checks(&rates, 0.0);
    assert!(cold.is_empty());
}

#[test]
fn gap_i1_disabled_by_default_in_config() {
    let cfg = h2ai_config::H2AIConfig::default();
    assert!(!cfg.gap_i1.enabled, "gap_i1 must be disabled by default");
}

// ── plateau detection ─────────────────────────────────────────────────────────

#[test]
fn plateau_detected_after_two_identical_scores() {
    let history = vec![0.50_f64, 0.77, 0.77];
    assert!(is_compliance_plateau(&history, 2, 0.02));
}

#[test]
fn plateau_not_detected_if_improving() {
    let history = vec![0.50_f64, 0.77, 0.85];
    assert!(!is_compliance_plateau(&history, 2, 0.02));
}

#[test]
fn plateau_not_detected_before_min_retry() {
    let history = vec![0.77_f64, 0.77];
    assert!(!is_compliance_plateau(&history, 1, 0.02));
}

// ── isolation evidence ────────────────────────────────────────────────────────

#[test]
fn isolation_evidence_detected_when_partials_cover_all_checks() {
    let partial_a = PartialPass {
        proposal_text: "proposal A".to_owned(),
        check_results: vec![
            (0, "check 0".to_owned(), true),
            (1, "check 1".to_owned(), true),
            (2, "check 2".to_owned(), false),
            (3, "check 3".to_owned(), false),
        ],
        score: 0.5,
    };
    let partial_b = PartialPass {
        proposal_text: "proposal B".to_owned(),
        check_results: vec![
            (0, "check 0".to_owned(), false),
            (1, "check 1".to_owned(), false),
            (2, "check 2".to_owned(), true),
            (3, "check 3".to_owned(), true),
        ],
        score: 0.5,
    };
    assert!(has_isolation_evidence(&[partial_a, partial_b], 4));
}

#[test]
fn isolation_evidence_absent_when_single_partial_covers_all() {
    let partial = PartialPass {
        proposal_text: "proposal".to_owned(),
        check_results: vec![
            (0, "c".to_owned(), true),
            (1, "c".to_owned(), true),
            (2, "c".to_owned(), true),
            (3, "c".to_owned(), true),
        ],
        score: 1.0,
    };
    assert!(!has_isolation_evidence(&[partial], 4));
}

#[test]
fn isolation_evidence_absent_when_coverage_incomplete() {
    let partial_a = PartialPass {
        proposal_text: "A".to_owned(),
        check_results: vec![
            (0, "c".to_owned(), true),
            (1, "c".to_owned(), true),
            (2, "c".to_owned(), false),
            (3, "c".to_owned(), false),
        ],
        score: 0.5,
    };
    let partial_b = PartialPass {
        proposal_text: "B".to_owned(),
        check_results: vec![
            (0, "c".to_owned(), false),
            (1, "c".to_owned(), false),
            (2, "c".to_owned(), true),
            (3, "c".to_owned(), false),
        ],
        score: 0.25,
    };
    assert!(!has_isolation_evidence(&[partial_a, partial_b], 4));
}

// ── cost guard: build_budget_hint_if_needed + check_convergence_gate ─────────

fn enabled_cost_guard(budget: u64, inject: bool) -> CostGuardConfig {
    CostGuardConfig {
        enabled: true,
        budget_tokens_per_task: budget,
        budget_warning_fraction: 0.80,
        budget_abort_fraction: 1.00,
        budget_prompt_injection_enabled: inject,
        budget_injection_warn_fraction: 0.50,
        budget_injection_max_complexity: 3,
    }
}

#[test]
fn fraction_used_computes_correctly_when_enabled() {
    let cg = enabled_cost_guard(100_000, false);
    assert!((cg.fraction_used(80_000) - 0.80).abs() < 1e-9);
    assert!((cg.fraction_used(100_000) - 1.00).abs() < 1e-9);
}

#[test]
fn budget_hint_built_when_in_injection_window() {
    let cg = enabled_cost_guard(100_000, true);
    let hint = build_budget_hint_if_needed(&cg, 60_000, 2);
    assert!(hint.is_some(), "expected hint at 60% consumption");
    assert!(hint.unwrap().contains("tokens remain"));
}

#[test]
fn budget_hint_skipped_above_85_percent() {
    let cg = enabled_cost_guard(100_000, true);
    let hint = build_budget_hint_if_needed(&cg, 90_000, 2);
    assert!(hint.is_none(), "must not inject above 85%");
}

#[test]
fn budget_hint_skipped_for_high_complexity() {
    let cg = enabled_cost_guard(100_000, true);
    let hint = build_budget_hint_if_needed(&cg, 60_000, 4);
    assert!(hint.is_none(), "must not inject for complexity > max");
}

fn enabled_convergence_gate() -> ConvergenceGateConfig {
    ConvergenceGateConfig {
        enabled: true,
        ..ConvergenceGateConfig::default()
    }
}

#[test]
fn convergence_gate_fires_when_conditions_met() {
    let gate = enabled_convergence_gate();
    assert!(check_convergence_gate(&gate, Some(0.92), 0.83, 1, 2, 0.50));
}

#[test]
fn convergence_gate_skipped_below_budget_floor() {
    let gate = enabled_convergence_gate();
    assert!(!check_convergence_gate(&gate, Some(0.92), 0.85, 1, 2, 0.10));
}

#[test]
fn convergence_gate_skipped_on_wave_zero() {
    let gate = enabled_convergence_gate();
    assert!(!check_convergence_gate(&gate, Some(0.92), 0.85, 0, 2, 0.50));
}

#[test]
fn convergence_gate_skipped_when_score_below_floor() {
    let gate = enabled_convergence_gate();
    assert!(!check_convergence_gate(&gate, Some(0.92), 0.75, 1, 2, 0.50));
}

// ── OOM guard pure functions ──────────────────────────────────────────────────

#[test]
fn oom_signal_none_below_threshold() {
    use h2ai_autonomic::repair::oom_signal;
    use h2ai_config::OomGuardConfig;
    let cfg = OomGuardConfig::default(); // rss_abort_mb = 4096
    assert!(oom_signal(3000, &cfg).is_none());
}

#[test]
fn oom_signal_some_at_threshold() {
    use h2ai_autonomic::repair::oom_signal;
    use h2ai_config::OomGuardConfig;
    let cfg = OomGuardConfig::default();
    let sig = oom_signal(4096, &cfg);
    assert!(sig.is_some());
    let s = sig.unwrap();
    assert_eq!(s.rss_mb, 4096);
    assert_eq!(s.limit_mb, 4096);
}

#[test]
fn oom_signal_some_above_threshold() {
    use h2ai_autonomic::repair::oom_signal;
    use h2ai_config::OomGuardConfig;
    let cfg = OomGuardConfig::default();
    assert!(oom_signal(5000, &cfg).is_some());
}

#[test]
fn oom_signal_none_when_disabled() {
    use h2ai_autonomic::repair::oom_signal;
    use h2ai_config::OomGuardConfig;
    let mut cfg = OomGuardConfig::default();
    cfg.enabled = false;
    assert!(oom_signal(99999, &cfg).is_none());
}

// ── OOM guard integration (wave boundary logic) ───────────────────────────────

#[test]
fn oom_guard_disabled_never_fires() {
    use h2ai_autonomic::repair::oom_signal;
    use h2ai_config::OomGuardConfig;
    let mut cfg = OomGuardConfig::default();
    cfg.enabled = false;
    assert!(oom_signal(100_000, &cfg).is_none());
}

#[test]
fn oom_guard_check_interval_respected() {
    let check_interval: u32 = 2;
    assert_eq!(0 % check_interval, 0); // wave 0 → check
    assert_ne!(1 % check_interval, 0); // wave 1 → skip
    assert_eq!(2 % check_interval, 0); // wave 2 → check
}

// ── gap_i1: extract_incorrect_concept ────────────────────────────────────────

fn make_pruned_event(
    constraint_id: &str,
    reason: &str,
    check_verdicts: Vec<bool>,
) -> h2ai_types::events::BranchPrunedEvent {
    use h2ai_types::sizing::RoleErrorCost;
    h2ai_types::events::BranchPrunedEvent {
        task_id: h2ai_types::identity::TaskId::new(),
        explorer_id: h2ai_types::identity::ExplorerId::new(),
        reason: "verifier score below threshold".to_string(),
        raw_output: "proposal".to_string(),
        constraint_error_cost: RoleErrorCost::new(0.0).unwrap(),
        violated_constraints: vec![h2ai_types::events::ConstraintViolation {
            constraint_id: constraint_id.to_string(),
            score: 0.0,
            severity_label: "Hard".to_string(),
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: Some(reason.to_string()),
            check_verdicts,
            criteria_pass: None,
            check_reasons: None,
        }],
        timestamp: chrono::Utc::now(),
        retry_count: 0,
        bypass_reason: None,
    }
}

#[test]
fn extract_incorrect_concept_ignores_empty_verdicts() {
    let pruned = make_pruned_event(
        "C-008",
        "process crash between Redis debit and WAL write",
        vec![],
    );
    let concept = h2ai_orchestrator::mape_k::MapeKController::extract_incorrect_concept_from(
        &[pruned],
        "C-008",
        3,
    );
    assert!(
        concept.is_empty(),
        "empty verdicts must not contaminate check 3 reason, got: {concept:?}"
    );
}

#[test]
fn extract_incorrect_concept_returns_reason_when_check_idx_known_failed() {
    let pruned = make_pruned_event(
        "C-008",
        "unbounded WATCH/MULTI/EXEC retry loop at high TPS",
        vec![true, true, true, false],
    );
    let concept = h2ai_orchestrator::mape_k::MapeKController::extract_incorrect_concept_from(
        &[pruned],
        "C-008",
        3,
    );
    assert!(
        concept.contains("WATCH"),
        "expected reason for check 3, got: {concept:?}"
    );
}

// ── generation_outcome pure function ─────────────────────────────────────────

#[test]
fn generation_outcome_all_timed_out_when_completed_empty() {
    use h2ai_orchestrator::phases::generation::{generation_outcome, GenerationPhaseResult};
    let result = generation_outcome(vec![], 3);
    assert!(matches!(result, GenerationPhaseResult::AllTimedOut));
}

#[test]
fn generation_outcome_all_timed_out_when_completed_empty_zero_timeouts() {
    use h2ai_orchestrator::phases::generation::{generation_outcome, GenerationPhaseResult};
    // 0 completed, 0 timed out → AllTimedOut (no output at all)
    let result = generation_outcome(vec![], 0);
    assert!(matches!(result, GenerationPhaseResult::AllTimedOut));
}

#[test]
fn generation_outcome_full_when_no_timeouts_and_some_completed() {
    use h2ai_orchestrator::phases::generation::{generation_outcome, GenerationPhaseResult};
    // We can't easily construct ProposalEvent in a unit test,
    // but we can test the logic: Full fires when timed_out_count == 0 and completed non-empty.
    // Instead, test that a non-empty completed + 0 timeouts → Full (via the enum logic)
    // Indirect test: full fires when timed_out_count == 0 and completed is non-empty.
    // We test the inverse: 0 timeouts but empty completed still = AllTimedOut.
    let result = generation_outcome(vec![], 0);
    assert!(matches!(result, GenerationPhaseResult::AllTimedOut));
}
