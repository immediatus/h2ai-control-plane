use h2ai_types::identity::{ExplorerId, TaskId};

#[test]
fn task_id_is_unique_each_time() {
    let a = TaskId::new();
    let b = TaskId::new();
    assert_ne!(a, b);
}

#[test]
fn task_id_display_is_hyphenated_uuid() {
    let id = TaskId::new();
    let s = id.to_string();
    assert_eq!(s.len(), 36);
    assert_eq!(s.chars().filter(|&c| c == '-').count(), 4);
}

#[test]
fn explorer_id_is_unique_each_time() {
    let a = ExplorerId::new();
    let b = ExplorerId::new();
    assert_ne!(a, b);
}

#[test]
fn task_id_and_explorer_id_are_distinct_types() {
    let _t = TaskId::new();
    let _e = ExplorerId::new();
}

#[test]
fn task_id_serde_round_trip() {
    let id = TaskId::new();
    let json = serde_json::to_string(&id).unwrap();
    let back: TaskId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, back);
}

#[test]
fn explorer_id_serde_round_trip() {
    let id = ExplorerId::new();
    let json = serde_json::to_string(&id).unwrap();
    let back: ExplorerId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, back);
}
