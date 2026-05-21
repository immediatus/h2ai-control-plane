use chrono::Utc;
use h2ai_autonomic::retry::{RetryAction, RetryPolicy};
use h2ai_types::config::TopologyKind;
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation, ZeroSurvivalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::RoleErrorCost;

fn zero_survival(task_id: &TaskId, retry_count: u32) -> ZeroSurvivalEvent {
    ZeroSurvivalEvent {
        task_id: task_id.clone(),
        retry_count,
        timestamp: Utc::now(),
        n_eff_cosine_actual: None,
        failure_mode: None,
    }
}

#[test]
fn retry_suggests_hierarchical_tree_after_ensemble_fails() {
    let task_id = TaskId::new();
    let tried = vec![TopologyKind::Ensemble];
    let action = RetryPolicy::decide(&zero_survival(&task_id, 0), &tried, vec![], vec![], None);
    assert!(matches!(
        action,
        RetryAction::Retry(TopologyKind::HierarchicalTree { .. })
    ));
}

#[test]
fn retry_suggests_team_swarm_hybrid_after_tree_fails() {
    let task_id = TaskId::new();
    let tried = vec![
        TopologyKind::Ensemble,
        TopologyKind::HierarchicalTree {
            branching_factor: Some(3),
        },
    ];
    let action = RetryPolicy::decide(&zero_survival(&task_id, 1), &tried, vec![], vec![], None);
    assert!(matches!(
        action,
        RetryAction::Retry(TopologyKind::TeamSwarmHybrid)
    ));
}

#[test]
fn retry_fails_after_all_three_topologies_tried() {
    let task_id = TaskId::new();
    let tried = vec![
        TopologyKind::Ensemble,
        TopologyKind::HierarchicalTree {
            branching_factor: None,
        },
        TopologyKind::TeamSwarmHybrid,
    ];
    let action = RetryPolicy::decide(&zero_survival(&task_id, 2), &tried, vec![], vec![], None);
    assert!(matches!(action, RetryAction::Fail(_)));
}

#[test]
fn retry_fail_action_contains_all_tried_topologies() {
    let task_id = TaskId::new();
    let tried = vec![
        TopologyKind::Ensemble,
        TopologyKind::HierarchicalTree {
            branching_factor: None,
        },
        TopologyKind::TeamSwarmHybrid,
    ];
    let action = RetryPolicy::decide(&zero_survival(&task_id, 2), &tried, vec![], vec![], None);
    if let RetryAction::Fail(event) = action {
        assert_eq!(event.topologies_tried.len(), 3);
    } else {
        panic!("expected Fail");
    }
}

#[test]
fn retry_suggests_ensemble_when_nothing_tried_yet() {
    let task_id = TaskId::new();
    let action = RetryPolicy::decide(&zero_survival(&task_id, 0), &[], vec![], vec![], None);
    assert!(matches!(action, RetryAction::Retry(TopologyKind::Ensemble)));
}

// ── Pruned-reason and constraint-violation tests ───────────────────────────────

fn pruned(reason: &str) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: reason.into(),
        constraint_error_cost: RoleErrorCost::new(0.5).unwrap(),
        violated_constraints: vec![],
        timestamp: Utc::now(),
    }
}

fn zero_event() -> ZeroSurvivalEvent {
    ZeroSurvivalEvent {
        task_id: TaskId::new(),
        retry_count: 0,
        timestamp: Utc::now(),
        n_eff_cosine_actual: None,
        failure_mode: None,
    }
}

#[test]
fn hallucination_reasons_trigger_tau_reduction() {
    let pruned_events = vec![
        pruned("hallucination detected: output fabricated facts"),
        pruned("hallucination detected: invented citations"),
    ];
    let action = RetryPolicy::decide(&zero_event(), &[], pruned_events, vec![], None);
    assert!(
        matches!(action, RetryAction::RetryWithTauReduction { .. }),
        "majority hallucination reasons must trigger tau reduction"
    );
}

#[test]
fn non_hallucination_reasons_use_plain_retry() {
    let pruned_events = vec![
        pruned("violated ADR-001 constraint"),
        pruned("missing required field"),
    ];
    let action = RetryPolicy::decide(&zero_event(), &[], pruned_events, vec![], None);
    assert!(matches!(action, RetryAction::Retry(_)));
}

#[test]
fn empty_pruned_events_uses_plain_retry() {
    let action = RetryPolicy::decide(&zero_event(), &[], vec![], vec![], None);
    assert!(matches!(action, RetryAction::Retry(_)));
}

#[test]
fn tau_reduction_factor_is_in_open_unit_interval() {
    let pruned_events = vec![pruned("hallucination detected")];
    let action = RetryPolicy::decide(&zero_event(), &[], pruned_events, vec![], None);
    if let RetryAction::RetryWithTauReduction { tau_factor, .. } = action {
        assert!(
            tau_factor > 0.0 && tau_factor < 1.0,
            "tau_factor must be in (0,1), got {tau_factor}"
        );
    }
}

#[test]
fn violated_constraints_with_hints_produce_retry_with_hints() {
    let mut event = pruned("constraint violation");
    event.violated_constraints = vec![ConstraintViolation {
        constraint_id: "GDPR-001".into(),
        score: 0.0,
        severity_label: "Hard".into(),
        remediation_hint: Some("Include explicit data minimization language.".into()),
    }];
    let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
    assert!(
        matches!(action, RetryAction::RetryWithHints { .. }),
        "structured Hard violations with hints must produce RetryWithHints"
    );
    if let RetryAction::RetryWithHints { hints, .. } = action {
        assert!(hints.iter().any(|h| h.contains("data minimization")));
    }
}

#[test]
fn violated_constraints_without_hints_fall_back_to_reason_scan() {
    // Violation with no remediation_hint → no hints collected → falls back to reason scan
    let mut event = pruned("hallucination detected: fabricated output");
    event.violated_constraints = vec![ConstraintViolation {
        constraint_id: "GDPR-001".into(),
        score: 0.0,
        severity_label: "Hard".into(),
        remediation_hint: None, // no hint
    }];
    let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
    // Since no hints, falls back to hallucination keyword scan → TauReduction
    assert!(
        matches!(action, RetryAction::RetryWithTauReduction { .. }),
        "no remediation hints + hallucination reason must still trigger tau reduction"
    );
}
