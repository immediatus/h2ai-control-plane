use h2ai_types::identity::SubtaskId;
use h2ai_types::identity::TaskId;
use h2ai_types::plan::{PlanStatus, Subtask, SubtaskPlan, SubtaskResult};

#[test]
fn subtask_id_display_is_uuid_string() {
    let id = SubtaskId::new();
    assert_eq!(id.to_string().len(), 36); // UUID hyphenated format
}

#[test]
fn subtask_plan_serialises_and_round_trips() {
    let task_id = TaskId::new();
    let a = SubtaskId::new();
    let b = SubtaskId::new();
    let plan = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: task_id.clone(),
        subtasks: vec![
            Subtask {
                id: a.clone(),
                description: "Define data model".into(),
                depends_on: vec![],
                role_hint: None,
            },
            Subtask {
                id: b.clone(),
                description: "Implement API endpoints".into(),
                depends_on: vec![a.clone()],
                role_hint: None,
            },
        ],
        status: PlanStatus::Draft,
        created_at: chrono::Utc::now(),
    };
    let json = serde_json::to_string(&plan).unwrap();
    let back: SubtaskPlan = serde_json::from_str(&json).unwrap();
    assert_eq!(back.subtasks.len(), 2);
    assert_eq!(back.subtasks[1].depends_on[0], a);
    assert_eq!(back.parent_task_id, task_id);
    assert_eq!(back.status, PlanStatus::Draft);
}

#[test]
fn plan_status_unit_variant_json_shape() {
    let json = serde_json::to_string(&PlanStatus::Draft).unwrap();
    assert_eq!(json, r#"{"status":"draft"}"#);
    let back: PlanStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(back, PlanStatus::Draft);
}

#[test]
fn plan_status_rejected_carries_reason() {
    let status = PlanStatus::Rejected {
        reason: "Missing error handling step".into(),
    };
    let json = serde_json::to_string(&status).unwrap();
    let back: PlanStatus = serde_json::from_str(&json).unwrap();
    assert!(
        matches!(back, PlanStatus::Rejected { reason } if reason == "Missing error handling step")
    );
}

#[test]
fn subtask_result_round_trips() {
    let r = SubtaskResult {
        subtask_id: SubtaskId::new(),
        output: "The API design uses REST with JSON payloads.".into(),
        token_cost: 42,
        timestamp: chrono::Utc::now(),
    };
    let json = serde_json::to_string(&r).unwrap();
    let back: SubtaskResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.output, r.output);
    assert_eq!(back.token_cost, 42);
}
