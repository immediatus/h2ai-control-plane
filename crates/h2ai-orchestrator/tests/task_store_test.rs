use h2ai_orchestrator::task_store::{TaskPhase, TaskState, TaskStore};
use h2ai_types::identity::TaskId;

#[test]
fn insert_and_retrieve() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(id.clone(), TaskState::new(id.clone()));
    let state = store.get(&id).unwrap();
    assert_eq!(state.phase, TaskPhase::Bootstrap as u8);
    assert_eq!(state.status, "pending");
}

#[test]
fn advance_phase_updates_status() {
    let store = TaskStore::new();
    let id = TaskId::new();
    store.insert(id.clone(), TaskState::new(id.clone()));
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
    store.insert(id.clone(), TaskState::new(id.clone()));
    store.mark_resolved(&id);
    let state = store.get(&id).unwrap();
    assert_eq!(state.status, "resolved");
}

#[test]
fn increment_completed_explorer() {
    let store = TaskStore::new();
    let id = TaskId::new();
    let mut initial = TaskState::new(id.clone());
    initial.explorers_total = 4;
    store.insert(id.clone(), initial);
    store.increment_completed(&id);
    store.increment_completed(&id);
    let state = store.get(&id).unwrap();
    assert_eq!(state.explorers_completed, 2);
}
