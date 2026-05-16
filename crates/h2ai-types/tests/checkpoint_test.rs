use h2ai_types::checkpoint::{ConstraintSnapshot, TaskCheckpoint};

#[test]
fn constraint_snapshot_roundtrip() {
    let snap = ConstraintSnapshot {
        wiki_revision: 42,
        resolved_ids: vec!["ADR-001".into(), "GDPR-DPA-001".into()],
        evaluated_ids: vec!["ADR-001".into()],
        violation_ids: vec![],
    };
    let json = serde_json::to_string(&snap).unwrap();
    let back: ConstraintSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back.wiki_revision, 42);
    assert_eq!(back.resolved_ids.len(), 2);
    assert!(back.violation_ids.is_empty());
}

#[test]
fn task_checkpoint_constraint_snapshot_defaults_none() {
    let json = r#"{
        "task_id": "abc",
        "phase": "ParallelGeneration",
        "node_id": "host:1234",
        "lease_seq": 1,
        "proposals": [],
        "auditor_survivors": [],
        "resolved_output": null,
        "manifest_json": "{}",
        "object_store_ref": null,
        "created_at_ms": 0,
        "updated_at_ms": 0
    }"#;
    let cp: TaskCheckpoint = serde_json::from_str(json).unwrap();
    assert!(
        cp.constraint_snapshot.is_none(),
        "constraint_snapshot must default to None"
    );
}

#[test]
fn task_checkpoint_j_eff_round_trips() {
    let cp = TaskCheckpoint {
        task_id: "t1".into(),
        phase: "Merging".into(),
        node_id: "host:1".into(),
        lease_seq: 0,
        proposals: vec![],
        auditor_survivors: vec![],
        resolved_output: Some("ok".into()),
        manifest_json: "{}".into(),
        object_store_ref: None,
        created_at_ms: 0,
        updated_at_ms: 0,
        constraint_snapshot: None,
        j_eff: Some(0.72),
    };
    let json = serde_json::to_string(&cp).unwrap();
    let back: TaskCheckpoint = serde_json::from_str(&json).unwrap();
    assert_eq!(back.j_eff, Some(0.72));
}

#[test]
fn task_checkpoint_j_eff_defaults_none_on_old_payload() {
    let json = r#"{
        "task_id": "abc",
        "phase": "Merging",
        "node_id": "host:1234",
        "lease_seq": 1,
        "proposals": [],
        "auditor_survivors": [],
        "resolved_output": null,
        "manifest_json": "{}",
        "object_store_ref": null,
        "created_at_ms": 0,
        "updated_at_ms": 0
    }"#;
    let cp: TaskCheckpoint = serde_json::from_str(json).unwrap();
    assert_eq!(
        cp.j_eff, None,
        "j_eff must default to None for old payloads"
    );
}
