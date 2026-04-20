use chrono::Utc;
use h2ai_adapters::MockAdapter;
use h2ai_planner::reviewer::{PlanReviewer, ReviewOutcome};
use h2ai_types::identity::{SubtaskId, TaskId};
use h2ai_types::physics::TauValue;
use h2ai_types::plan::{PlanStatus, Subtask, SubtaskPlan};

fn two_step_plan() -> SubtaskPlan {
    let a = SubtaskId::new();
    let b = SubtaskId::new();
    SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![
            Subtask {
                id: a.clone(),
                description: "Design schema".into(),
                depends_on: vec![],
                role_hint: None,
            },
            Subtask {
                id: b.clone(),
                description: "Implement endpoints".into(),
                depends_on: vec![a],
                role_hint: None,
            },
        ],
        status: PlanStatus::PendingReview,
        created_at: Utc::now(),
    }
}

#[tokio::test]
async fn reviewer_approves_when_llm_returns_true() {
    let adapter =
        MockAdapter::new(r#"{"approved": true, "reason": "Plan fully covers the task."}"#.into());
    let outcome = PlanReviewer::evaluate(
        &two_step_plan(),
        "Build a REST API for user authentication",
        &adapter,
        TauValue::new(0.2).unwrap(),
    )
    .await
    .unwrap();
    assert!(matches!(outcome, ReviewOutcome::Approved));
}

#[tokio::test]
async fn reviewer_rejects_when_llm_returns_false() {
    let adapter =
        MockAdapter::new(r#"{"approved": false, "reason": "Missing token refresh step."}"#.into());
    let outcome = PlanReviewer::evaluate(
        &two_step_plan(),
        "Build auth with token refresh",
        &adapter,
        TauValue::new(0.2).unwrap(),
    )
    .await
    .unwrap();
    assert!(matches!(outcome, ReviewOutcome::Rejected { reason } if reason.contains("refresh")));
}

#[tokio::test]
async fn reviewer_rejects_empty_plan_without_llm_call() {
    // Adapter returns malformed JSON to prove no call was made.
    let adapter = MockAdapter::new("NOT JSON".into());
    let empty = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![],
        status: PlanStatus::PendingReview,
        created_at: Utc::now(),
    };
    let outcome = PlanReviewer::evaluate(&empty, "anything", &adapter, TauValue::new(0.2).unwrap())
        .await
        .unwrap();
    assert!(matches!(outcome, ReviewOutcome::Rejected { .. }));
}

#[tokio::test]
async fn reviewer_rejects_cyclic_plan_without_llm_call() {
    let adapter = MockAdapter::new("NOT JSON".into());
    let a = SubtaskId::new();
    let b = SubtaskId::new();
    // A depends on B, B depends on A — direct cycle.
    let cyclic = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![
            Subtask {
                id: a.clone(),
                description: "A".into(),
                depends_on: vec![b.clone()],
                role_hint: None,
            },
            Subtask {
                id: b.clone(),
                description: "B".into(),
                depends_on: vec![a.clone()],
                role_hint: None,
            },
        ],
        status: PlanStatus::PendingReview,
        created_at: Utc::now(),
    };
    let outcome =
        PlanReviewer::evaluate(&cyclic, "anything", &adapter, TauValue::new(0.2).unwrap())
            .await
            .unwrap();
    assert!(
        matches!(outcome, ReviewOutcome::Rejected { reason } if reason.contains("cycle") || reason.contains("Cyclic"))
    );
}

#[tokio::test]
async fn reviewer_rejects_self_referential_cycle() {
    // A depends on itself — the simplest cycle.
    let adapter = MockAdapter::new("NOT JSON".into());
    let a = SubtaskId::new();
    let plan = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![Subtask {
            id: a.clone(),
            description: "Self-dependent".into(),
            depends_on: vec![a.clone()],
            role_hint: None,
        }],
        status: PlanStatus::PendingReview,
        created_at: Utc::now(),
    };
    let outcome =
        PlanReviewer::evaluate(&plan, "anything", &adapter, TauValue::new(0.2).unwrap())
            .await
            .unwrap();
    assert!(
        matches!(outcome, ReviewOutcome::Rejected { .. }),
        "self-referential dependency must be rejected"
    );
}

#[tokio::test]
async fn reviewer_rejects_three_node_cycle() {
    // A → B → C → A
    let adapter = MockAdapter::new("NOT JSON".into());
    let a = SubtaskId::new();
    let b = SubtaskId::new();
    let c = SubtaskId::new();
    let plan = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![
            Subtask {
                id: a.clone(),
                description: "A".into(),
                depends_on: vec![c.clone()], // A depends on C
                role_hint: None,
            },
            Subtask {
                id: b.clone(),
                description: "B".into(),
                depends_on: vec![a.clone()], // B depends on A
                role_hint: None,
            },
            Subtask {
                id: c.clone(),
                description: "C".into(),
                depends_on: vec![b.clone()], // C depends on B → cycle A→C→B→A
                role_hint: None,
            },
        ],
        status: PlanStatus::PendingReview,
        created_at: Utc::now(),
    };
    let outcome =
        PlanReviewer::evaluate(&plan, "anything", &adapter, TauValue::new(0.2).unwrap())
            .await
            .unwrap();
    assert!(
        matches!(outcome, ReviewOutcome::Rejected { .. }),
        "3-node cycle must be rejected"
    );
}

#[tokio::test]
async fn reviewer_returns_parse_error_when_approved_field_missing() {
    // LLM omits the "approved" field — strict deserialization must fail.
    use h2ai_planner::decomposer::PlannerError;
    let adapter = MockAdapter::new(r#"{"reason": "looks fine to me"}"#.into());
    let result = PlanReviewer::evaluate(
        &two_step_plan(),
        "anything",
        &adapter,
        TauValue::new(0.2).unwrap(),
    )
    .await;
    assert!(
        matches!(result, Err(PlannerError::ParseError(_))),
        "missing 'approved' field must produce ParseError, got: {result:?}"
    );
}
