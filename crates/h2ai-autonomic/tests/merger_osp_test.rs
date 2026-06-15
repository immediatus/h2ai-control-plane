#![allow(clippy::match_wildcard_for_single_variants)]
use chrono::Utc;
use h2ai_autonomic::merger::{MergeEngine, MergeOutcome};
use h2ai_state::semilattice::ProposalSet;
use h2ai_types::config::AdapterKind;
use h2ai_types::events::{ConstraintViolation, ProposalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::{MergeStrategy, OspConfig, TauValue};

fn tid() -> TaskId {
    TaskId::new()
}
fn eid(_s: &str) -> ExplorerId {
    ExplorerId::new()
}

fn adapter() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://api.test".into(),
        api_key_env: "K".into(),
        model: None,
        provider: Default::default(),
    }
}

fn proposal(explorer_id: ExplorerId, task_id: TaskId, text: &str) -> ProposalEvent {
    ProposalEvent {
        task_id,
        explorer_id,
        tau: TauValue::new(0.4).unwrap(),
        generation: 0,
        raw_output: text.to_string(),
        token_cost: 10,
        adapter_kind: adapter(),
        timestamp: Utc::now(),
    }
}

fn v(cid: &str, hint: &str) -> ConstraintViolation {
    ConstraintViolation {
        constraint_id: cid.to_string(),
        score: 0.0,
        severity_label: "Hard".to_string(),
        remediation_hint: Some(hint.to_string()),
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: None,
    }
}

#[tokio::test]
async fn osp_zero_survival_before_any_selection() {
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(eid("e1"), tid(), "bad"), 0.0);

    let outcome = MergeEngine::resolve(
        tid(),
        set,
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
        Some(&[v("c-001", "hint")]),
        None,
        Some(&OspConfig::default()),
    )
    .await;

    assert!(matches!(outcome, MergeOutcome::ZeroSurvival(_)));
}

#[tokio::test]
async fn osp_single_survivor_returns_valid_only() {
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(eid("e1"), tid(), "good output"), 0.8);
    set.insert_scored(proposal(eid("e2"), tid(), "bad output"), 0.0);

    let outcome = MergeEngine::resolve(
        tid(),
        set,
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
        None,
        None,
        Some(&OspConfig::default()),
    )
    .await;

    match outcome {
        MergeOutcome::Resolved { resolved, .. } => {
            assert_eq!(resolved.resolved_output, "good output");
        }
        _ => panic!("expected Resolved"),
    }
}

#[tokio::test]
async fn osp_clear_leader_picks_top_score() {
    // Δ = 0.9 - 0.2 = 0.7 ≥ 2 * T_v = 0.25 → ClearLeader → pick argmax
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(eid("e1"), tid(), "leader"), 0.9);
    set.insert_scored(proposal(eid("e2"), tid(), "follower"), 0.2);

    let outcome = MergeEngine::resolve(
        tid(),
        set,
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
        None,
        None,
        Some(&OspConfig::default()),
    )
    .await;

    match outcome {
        MergeOutcome::Resolved { resolved, .. } => {
            assert_eq!(resolved.resolved_output, "leader");
        }
        _ => panic!("expected Resolved"),
    }
}

#[tokio::test]
async fn osp_zone3_attached_when_concordant() {
    // N_f=1, τ(1)=1.0, C_k=1.0 → inject
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(eid("e1"), tid(), "valid output"), 0.8);
    let violations = vec![v("c-005", "Validate all external inputs")];

    let outcome = MergeEngine::resolve(
        tid(),
        set,
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
        Some(&violations),
        None,
        Some(&OspConfig::default()),
    )
    .await;

    match outcome {
        MergeOutcome::Resolved { resolved, .. } => {
            let z3 = resolved.zone3_hints.expect("zone3 must be set");
            assert!(z3.contains("c-005"));
            assert!(z3.contains("Validate all external inputs"));
        }
        _ => panic!("expected Resolved"),
    }
}

#[tokio::test]
async fn osp_zone3_absent_when_n_v_5() {
    let mut set = ProposalSet::new();
    for i in 0..5 {
        set.insert_scored(
            proposal(eid(&format!("e{i}")), tid(), &format!("v{i}")),
            0.8,
        );
    }
    let violations = vec![v("c-001", "hint")];

    let outcome = MergeEngine::resolve(
        tid(),
        set,
        vec![],
        MergeStrategy::ConsensusMedian,
        0,
        None,
        Some(&violations),
        None,
        Some(&OspConfig::default()),
    )
    .await;

    match outcome {
        MergeOutcome::Resolved { resolved, .. } => {
            assert!(resolved.zone3_hints.is_none(), "N_v=5 suppresses zone3");
        }
        _ => panic!("expected Resolved"),
    }
}

#[tokio::test]
async fn osp_backward_compat_with_osp_config_none() {
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(eid("e1"), tid(), "output"), 0.8);

    let outcome = MergeEngine::resolve(
        tid(),
        set,
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
        None,
        None,
        None,
    )
    .await;

    assert!(matches!(outcome, MergeOutcome::Resolved { .. }));
}

#[tokio::test]
async fn osp_softmax_nan_free_identical_scores() {
    // A.4: identical scores must not NaN
    let mut set = ProposalSet::new();
    for i in 0..3 {
        set.insert_scored(
            proposal(eid(&format!("e{i}")), tid(), &format!("o{i}")),
            0.5,
        );
    }

    let outcome = MergeEngine::resolve(
        tid(),
        set,
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
        None,
        None,
        Some(&OspConfig::default()),
    )
    .await;

    match outcome {
        MergeOutcome::Resolved { resolved, .. } => {
            assert!(!resolved.resolved_output.is_empty());
        }
        _ => panic!("expected Resolved"),
    }
}

#[tokio::test]
async fn osp_retry_accumulator_updated_on_resolve() {
    // Passes both retry_accumulator and osp_config with violations → exercises lines 249-253.
    use h2ai_autonomic::retry_accumulator::RetryAccumulator;
    let task_id = tid();
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(eid("e1"), tid(), "valid output"), 0.8);
    let violations = vec![v("c-007", "Ensure idempotency")];
    let mut acc = RetryAccumulator::new();

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
        Some(&violations),
        Some(&mut acc),
        Some(&OspConfig::default()),
    )
    .await;

    assert!(matches!(outcome, MergeOutcome::Resolved { .. }));
    assert!(
        acc.rates().contains_key("c-007"),
        "RetryAccumulator must be updated with violation rates"
    );
}

#[tokio::test]
async fn osp_retry_accumulator_not_updated_when_violations_none() {
    // retry_accumulator provided but violations=None → inner if-let not entered.
    use h2ai_autonomic::retry_accumulator::RetryAccumulator;
    let task_id = tid();
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(eid("e1"), tid(), "output"), 0.8);
    let mut acc = RetryAccumulator::new();

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
        None,
        Some(&mut acc),
        Some(&OspConfig::default()),
    )
    .await;

    assert!(matches!(outcome, MergeOutcome::Resolved { .. }));
    assert!(
        acc.rates().is_empty(),
        "no violations means accumulator unchanged"
    );
}

#[tokio::test]
async fn osp_empty_violations_vec_uses_semilattice_n_f() {
    // violations = Some(vec![]) → the `else` branch on line 240 is taken:
    // n_f falls back to n_f_semilattice (no .max(1) bump).
    let task_id = tid();
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(eid("e1"), tid(), "output a"), 0.8);
    set.insert_scored(proposal(eid("e2"), tid(), "output b"), 0.0);
    let empty_violations: Vec<h2ai_types::events::ConstraintViolation> = vec![];

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        vec![],
        MergeStrategy::ScoreOrdered,
        0,
        None,
        Some(&empty_violations),
        None,
        Some(&OspConfig::default()),
    )
    .await;

    assert!(matches!(outcome, MergeOutcome::Resolved { .. }));
}
