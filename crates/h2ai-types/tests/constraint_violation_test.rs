use h2ai_types::events::ConstraintViolation;

#[test]
fn constraint_violation_serde_round_trip() {
    let v = ConstraintViolation {
        constraint_id: "ADR-001".into(),
        score: 0.25,
        severity_label: "Hard".into(),
        remediation_hint: Some("Include 'data minimization' in the response.".into()),
    };
    let json = serde_json::to_string(&v).unwrap();
    let back: ConstraintViolation = serde_json::from_str(&json).unwrap();
    assert_eq!(back.constraint_id, "ADR-001");
    assert!((back.score - 0.25).abs() < 1e-9);
    assert_eq!(back.remediation_hint.unwrap(), "Include 'data minimization' in the response.");
}

#[test]
fn branch_pruned_event_carries_violations() {
    use chrono::Utc;
    use h2ai_types::events::BranchPrunedEvent;
    use h2ai_types::identity::{ExplorerId, TaskId};
    use h2ai_types::physics::RoleErrorCost;

    let e = BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: "constraint violation".into(),
        constraint_error_cost: RoleErrorCost::new(0.75).unwrap(),
        violated_constraints: vec![ConstraintViolation {
            constraint_id: "GDPR-001".into(),
            score: 0.0,
            severity_label: "Hard".into(),
            remediation_hint: Some("Do not include PII.".into()),
        }],
        timestamp: Utc::now(),
    };
    assert_eq!(e.violated_constraints.len(), 1);
    assert_eq!(e.violated_constraints[0].constraint_id, "GDPR-001");
}
