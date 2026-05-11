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
        },
        ComplianceResult {
            constraint_id: "c2".into(),
            score: 0.0,
            severity: ConstraintSeverity::Hard { threshold: 0.5 },
            remediation_hint: None,
        },
    ];
    let results_b = vec![
        ComplianceResult {
            constraint_id: "c1".into(),
            score: 0.0,
            severity: ConstraintSeverity::Hard { threshold: 0.5 },
            remediation_hint: None,
        },
        ComplianceResult {
            constraint_id: "c2".into(),
            score: 1.0,
            severity: ConstraintSeverity::Hard { threshold: 0.5 },
            remediation_hint: None,
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
