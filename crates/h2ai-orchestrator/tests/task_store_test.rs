use h2ai_orchestrator::task_store::{TaskPhase, TaskState, TaskStore};
use h2ai_types::identity::{TaskId, TenantId};

// ── TryFrom<u8> ──────────────────────────────────────────────────────────────

#[test]
fn try_from_u8_all_known_variants() {
    let cases: &[(u8, TaskPhase)] = &[
        (1, TaskPhase::Bootstrap),
        (2, TaskPhase::Provisioning),
        (3, TaskPhase::MultiplicationCheck),
        (4, TaskPhase::ParallelGeneration),
        (5, TaskPhase::AuditorGate),
        (6, TaskPhase::Merging),
        (7, TaskPhase::Resolved),
        (8, TaskPhase::Failed),
        (9, TaskPhase::ComplexityAssessed),
        (10, TaskPhase::AwaitingApproval),
    ];
    for &(byte, ref expected) in cases {
        let got = TaskPhase::try_from(byte).expect("should parse");
        assert_eq!(got, *expected, "byte {byte}");
    }
}

#[test]
fn try_from_u8_unknown_returns_err() {
    assert!(TaskPhase::try_from(0u8).is_err());
    assert!(TaskPhase::try_from(255u8).is_err());
}

// ── try_from_name_str ────────────────────────────────────────────────────────

#[test]
fn try_from_name_str_all_known() {
    let cases: &[(&str, TaskPhase)] = &[
        ("Bootstrap", TaskPhase::Bootstrap),
        ("TopologyProvisioning", TaskPhase::Provisioning),
        ("MultiplicationCheck", TaskPhase::MultiplicationCheck),
        ("ParallelGeneration", TaskPhase::ParallelGeneration),
        ("AuditorGate", TaskPhase::AuditorGate),
        ("Merging", TaskPhase::Merging),
        ("Resolved", TaskPhase::Resolved),
        ("Failed", TaskPhase::Failed),
        ("ComplexityAssessment", TaskPhase::ComplexityAssessed),
        ("AwaitingApproval", TaskPhase::AwaitingApproval),
    ];
    for (name, expected) in cases {
        let got = TaskPhase::try_from_name_str(name).expect("should parse");
        assert_eq!(got, *expected, "name {name}");
    }
}

#[test]
fn try_from_name_str_unknown_returns_none() {
    assert!(TaskPhase::try_from_name_str("Unknown").is_none());
    assert!(TaskPhase::try_from_name_str("").is_none());
}

// ── status_str / name_str for all variants ────────────────────────────────────

#[test]
fn status_str_and_name_str_cover_all_phases() {
    let all = [
        TaskPhase::Bootstrap,
        TaskPhase::ComplexityAssessed,
        TaskPhase::Provisioning,
        TaskPhase::MultiplicationCheck,
        TaskPhase::ParallelGeneration,
        TaskPhase::AuditorGate,
        TaskPhase::Merging,
        TaskPhase::Resolved,
        TaskPhase::Failed,
        TaskPhase::AwaitingApproval,
    ];
    for phase in &all {
        assert!(!phase.status_str().is_empty(), "{phase:?} status_str empty");
        assert!(!phase.name_str().is_empty(), "{phase:?} name_str empty");
    }
    assert_eq!(
        TaskPhase::AwaitingApproval.status_str(),
        "awaiting_approval"
    );
    assert_eq!(TaskPhase::AwaitingApproval.name_str(), "AwaitingApproval");
    assert_eq!(TaskPhase::ComplexityAssessed.status_str(), "assessing");
    assert_eq!(
        TaskPhase::ComplexityAssessed.name_str(),
        "ComplexityAssessment"
    );
}

// ── set_awaiting_approval ─────────────────────────────────────────────────────

#[test]
fn set_awaiting_approval_transitions_phase() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(
        id.clone(),
        TaskState::new(id.clone(), TenantId::default_tenant()),
    );
    store.set_awaiting_approval(&id);
    let state = store.get(&id).unwrap();
    assert_eq!(state.phase, TaskPhase::AwaitingApproval as u8);
    assert_eq!(state.phase_name, "AwaitingApproval");
    assert_eq!(state.status, "awaiting_approval");
}

// ── is_active ─────────────────────────────────────────────────────────────────

#[test]
fn is_active_returns_false_for_unknown_task() {
    let store = TaskStore::new();
    assert!(!store.is_active(&TaskId::new()));
}

#[test]
fn is_active_returns_true_for_in_progress_task() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(
        id.clone(),
        TaskState::new(id.clone(), TenantId::default_tenant()),
    );
    assert!(store.is_active(&id));
}

#[test]
fn is_active_returns_false_after_resolved() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(
        id.clone(),
        TaskState::new(id.clone(), TenantId::default_tenant()),
    );
    store.mark_resolved(&id);
    assert!(!store.is_active(&id));
}

#[test]
fn is_active_returns_false_after_failed() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(
        id.clone(),
        TaskState::new(id.clone(), TenantId::default_tenant()),
    );
    store.mark_failed(&id);
    assert!(!store.is_active(&id));
}

// ── record_validation ─────────────────────────────────────────────────────────

#[test]
fn record_validation_increments_valid() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(
        id.clone(),
        TaskState::new(id.clone(), TenantId::default_tenant()),
    );
    store.record_validation(&id, true);
    let state = store.get(&id).unwrap();
    assert_eq!(state.proposals_valid, 1);
    assert_eq!(state.proposals_pruned, 0);
}

#[test]
fn record_validation_increments_pruned() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(
        id.clone(),
        TaskState::new(id.clone(), TenantId::default_tenant()),
    );
    store.record_validation(&id, false);
    let state = store.get(&id).unwrap();
    assert_eq!(state.proposals_pruned, 1);
    assert_eq!(state.proposals_valid, 0);
}

#[test]
fn insert_and_retrieve() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(
        id.clone(),
        TaskState::new(id.clone(), TenantId::default_tenant()),
    );
    let state = store.get(&id).unwrap();
    assert_eq!(state.phase, TaskPhase::Bootstrap as u8);
    assert_eq!(state.status, "pending");
}

#[test]
fn advance_phase_updates_status() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(
        id.clone(),
        TaskState::new(id.clone(), TenantId::default_tenant()),
    );
    store.set_phase(&id, TaskPhase::ParallelGeneration, 4, 0);
    let state = store.get(&id).unwrap();
    assert_eq!(state.phase, TaskPhase::ParallelGeneration as u8);
    assert_eq!(state.status, "generating");
    assert_eq!(state.explorers_total, 4);
}

#[test]
fn mark_resolved_closes_task() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(
        id.clone(),
        TaskState::new(id.clone(), TenantId::default_tenant()),
    );
    store.mark_resolved(&id);
    let state = store.get(&id).unwrap();
    assert_eq!(state.status, "resolved");
}

#[test]
fn increment_completed_explorer() {
    let store = TaskStore::new();
    let id = TaskId::new();
    let mut initial = TaskState::new(id.clone(), TenantId::default_tenant());
    initial.explorers_total = 4;
    store.insert(id.clone(), initial);
    store.increment_completed(&id);
    store.increment_completed(&id);
    let state = store.get(&id).unwrap();
    assert_eq!(state.explorers_completed, 2);
}

#[test]
fn get_for_tenant_returns_none_for_wrong_tenant() {
    let store = TaskStore::new();
    let task_id = TaskId::new();
    store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), TenantId::from("acme")),
    );
    assert!(store
        .get_for_tenant(&task_id, &TenantId::from("beta"))
        .is_none());
}

#[test]
fn get_for_tenant_returns_state_for_owner() {
    let store = TaskStore::new();
    let task_id = TaskId::new();
    let tenant = TenantId::from("acme");
    store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), tenant.clone()),
    );
    assert!(store.get_for_tenant(&task_id, &tenant).is_some());
}

#[test]
fn get_without_tenant_still_works_for_backward_compat() {
    let store = TaskStore::new();
    let task_id = TaskId::new();
    store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), TenantId::default_tenant()),
    );
    assert!(store.get(&task_id).is_some());
}
