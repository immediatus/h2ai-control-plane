use chrono::Utc;
use h2ai_autonomic::retry::{RetryAction, RetryPolicy};
use h2ai_types::config::TopologyKind;
use h2ai_types::events::ZeroSurvivalEvent;
use h2ai_types::identity::TaskId;

fn zero_survival(task_id: &TaskId, retry_count: u32) -> ZeroSurvivalEvent {
    ZeroSurvivalEvent {
        task_id: task_id.clone(),
        retry_count,
        timestamp: Utc::now(),
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
