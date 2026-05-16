use h2ai_orchestrator::task_store::TaskPhase;
use h2ai_types::checkpoint::TaskCheckpoint;

#[test]
fn run_from_checkpoint_phase_routing_logic() {
    // Verifies the name-string routing that run_from_checkpoint uses.
    // For Merging: should use saved output directly.
    // For earlier phases: should fall back to run_offline (tested by phase name match).

    let merging = TaskPhase::try_from_name_str("Merging").unwrap();
    assert_eq!(merging.name_str(), "Merging");

    let earlier = TaskPhase::try_from_name_str("ParallelGeneration").unwrap();
    assert_ne!(earlier.name_str(), "Merging");

    // Unknown phase maps to None
    assert!(TaskPhase::try_from_name_str("UnknownPhase").is_none());
}

#[test]
fn checkpoint_serializes_and_deserializes() {
    let c = TaskCheckpoint {
        task_id: "test-task-1".into(),
        phase: "ParallelGeneration".into(),
        node_id: "node-1".into(),
        lease_seq: 42,
        proposals: vec!["proposal A".into(), "proposal B".into()],
        auditor_survivors: vec![0, 1],
        resolved_output: None,
        manifest_json: r#"{"description":"test"}"#.into(),
        object_store_ref: None,
        created_at_ms: 1000,
        updated_at_ms: 2000,
        constraint_snapshot: None,
        j_eff: None,
    };
    let json = serde_json::to_string(&c).unwrap();
    let back: TaskCheckpoint = serde_json::from_str(&json).unwrap();
    assert_eq!(c, back);
}

#[test]
fn checkpoint_with_resolved_output_roundtrips() {
    let c = TaskCheckpoint {
        task_id: "test-task-2".into(),
        phase: "Merging".into(),
        node_id: "node-2".into(),
        lease_seq: 7,
        proposals: vec![],
        auditor_survivors: vec![],
        resolved_output: Some("final answer here".into()),
        manifest_json: "{}".into(),
        object_store_ref: Some("sha256:abcdef123456".into()),
        created_at_ms: 3000,
        updated_at_ms: 4000,
        constraint_snapshot: None,
        j_eff: None,
    };
    let json = serde_json::to_string(&c).unwrap();
    let back: TaskCheckpoint = serde_json::from_str(&json).unwrap();
    assert_eq!(c, back);
}

#[test]
fn phase_name_str_round_trips_via_try_from_name_str() {
    let phases = [
        TaskPhase::ParallelGeneration,
        TaskPhase::AuditorGate,
        TaskPhase::Merging,
        TaskPhase::Resolved,
        TaskPhase::AwaitingApproval,
    ];
    for phase in phases {
        let name = phase.name_str();
        let back = TaskPhase::try_from_name_str(name)
            .unwrap_or_else(|| panic!("failed to round-trip phase: {name}"));
        assert_eq!(back.name_str(), name);
    }
}

#[test]
fn awaiting_approval_phase_has_correct_status() {
    assert_eq!(
        TaskPhase::AwaitingApproval.status_str(),
        "awaiting_approval"
    );
    assert_eq!(TaskPhase::AwaitingApproval.name_str(), "AwaitingApproval");
}

#[test]
fn try_from_name_str_unknown_returns_none() {
    assert!(TaskPhase::try_from_name_str("NonExistent").is_none());
    assert!(TaskPhase::try_from_name_str("").is_none());
}

#[test]
fn checkpoint_payload_compresses_and_decompresses() {
    let c = TaskCheckpoint {
        task_id: "compress-test".into(),
        phase: "Merging".into(),
        node_id: "node-compress".into(),
        lease_seq: 1,
        proposals: vec!["a long proposal string that should compress well".repeat(10)],
        auditor_survivors: vec![0],
        resolved_output: Some("final output".repeat(20)),
        manifest_json: r#"{"description":"compression test"}"#.into(),
        object_store_ref: None,
        created_at_ms: 1000,
        updated_at_ms: 2000,
        constraint_snapshot: None,
        j_eff: None,
    };

    let json = serde_json::to_vec(&c).unwrap();
    let compressed = zstd::encode_all(json.as_slice(), 3).unwrap();
    let decompressed = zstd::decode_all(compressed.as_slice()).unwrap();
    let back: TaskCheckpoint = serde_json::from_slice(&decompressed).unwrap();

    assert_eq!(c, back);
    // Compressed should be significantly smaller (repetitive text compresses well)
    assert!(
        compressed.len() < json.len(),
        "compressed ({}) should be smaller than original ({})",
        compressed.len(),
        json.len()
    );
}
