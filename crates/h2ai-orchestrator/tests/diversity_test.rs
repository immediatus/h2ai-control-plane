use h2ai_constraints::types::{ComplianceResult, ConstraintSeverity};
use h2ai_orchestrator::diversity::{DiversityGuard, DiversityResult};
use h2ai_types::events::ProposalEvent;

fn make_result(score: f64, hard: bool) -> ComplianceResult {
    ComplianceResult {
        constraint_id: "c1".into(),
        score,
        severity: if hard {
            ConstraintSeverity::Hard { threshold: 0.5 }
        } else {
            ConstraintSeverity::Soft { weight: 1.0 }
        },
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: vec![],
    }
}

fn proposal() -> ProposalEvent {
    use h2ai_types::config::AdapterKind;
    use h2ai_types::identity::{ExplorerId, TaskId};
    use h2ai_types::sizing::TauValue;
    ProposalEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.7).unwrap(),
        generation: 0,
        raw_output: "output".into(),
        token_cost: 10,
        adapter_kind: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "NONE".into(),
            model: None,
            provider: Default::default(),
        },
        timestamp: chrono::Utc::now(),
    }
}

#[test]
fn single_proposal_is_always_diverse() {
    let passed = vec![(proposal(), vec![make_result(1.0, true)], false)];
    assert!(matches!(
        DiversityGuard::check(&passed, 0.15),
        DiversityResult::Diverse
    ));
}

#[test]
fn identical_profiles_collapse() {
    let results = vec![make_result(1.0, true), make_result(1.0, true)];
    let passed = vec![
        (proposal(), results.clone(), false),
        (proposal(), results, false),
    ];
    assert!(matches!(
        DiversityGuard::check(&passed, 0.15),
        DiversityResult::Collapsed
    ));
}

#[test]
fn opposite_profiles_are_diverse() {
    let results_a = vec![
        ComplianceResult {
            constraint_id: "c1".into(),
            score: 1.0,
            severity: ConstraintSeverity::Hard { threshold: 0.5 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
        ComplianceResult {
            constraint_id: "c2".into(),
            score: 0.0,
            severity: ConstraintSeverity::Hard { threshold: 0.5 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
    ];
    let results_b = vec![
        ComplianceResult {
            constraint_id: "c1".into(),
            score: 0.0,
            severity: ConstraintSeverity::Hard { threshold: 0.5 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
        ComplianceResult {
            constraint_id: "c2".into(),
            score: 1.0,
            severity: ConstraintSeverity::Hard { threshold: 0.5 },
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: vec![],
        },
    ];
    let passed = vec![
        (proposal(), results_a, false),
        (proposal(), results_b, false),
    ];
    assert!(matches!(
        DiversityGuard::check(&passed, 0.15),
        DiversityResult::Diverse
    ));
}

#[test]
fn empty_fingerprints_fail_open() {
    let passed = vec![(proposal(), vec![], false), (proposal(), vec![], false)];
    assert!(matches!(
        DiversityGuard::check(&passed, 0.15),
        DiversityResult::Diverse
    ));
}

#[test]
fn fewer_than_two_proposals_is_always_diverse() {
    let passed: Vec<(ProposalEvent, Vec<ComplianceResult>, bool)> = vec![];
    assert!(matches!(
        DiversityGuard::check(&passed, 0.15),
        DiversityResult::Diverse
    ));
}

#[test]
fn mismatched_fingerprint_lengths_fail_open() {
    // One proposal has 2 constraints, other has 1 — corpus inconsistency → Diverse
    let short = vec![make_result(1.0, true)];
    let long = vec![make_result(1.0, true), make_result(1.0, true)];
    let passed = vec![(proposal(), short, false), (proposal(), long, false)];
    assert!(matches!(
        DiversityGuard::check(&passed, 0.15),
        DiversityResult::Diverse
    ));
}

/// When the diversity gate collapses, the EarlyExit(ZeroSurvival) must carry
/// `partial_verification_events` so the pipeline can propagate them into
/// `wave.events.verification_events` even though no audit/merge phase runs.
///
/// This test will fail to compile until `ExitReason::ZeroSurvival` has the
/// `partial_verification_events` field (the TDD failing-test gate).
#[test]
fn zero_survival_exit_carries_partial_verification_events() {
    use chrono::Utc;
    use h2ai_orchestrator::coherence::CoherenceState;
    use h2ai_orchestrator::phases::ExitReason;
    use h2ai_types::events::VerificationScoredEvent;
    use h2ai_types::identity::{ExplorerId, TaskId};

    let evt = VerificationScoredEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        score: 0.9,
        reason: "all constraints passed".into(),
        passed: true,
        cache_hit: false,
        passed_checks: Some(3),
        total_checks: Some(3),
        score_lower: None,
        score_upper: None,
        per_check_verdicts: vec![],
        timestamp: Utc::now(),
    };
    let exit = ExitReason::ZeroSurvival {
        failure_mode: None,
        coherence: CoherenceState::default(),
        n_eff_cosine: None,
        filter_ratio: 0.0,
        tau_values: vec![],
        partial_verification_events: vec![evt],
    };
    let ExitReason::ZeroSurvival {
        partial_verification_events,
        ..
    } = exit
    else {
        unreachable!()
    };
    assert_eq!(
        partial_verification_events.len(),
        1,
        "ZeroSurvival from diversity gate must carry the passed proposals' verification events"
    );
}
