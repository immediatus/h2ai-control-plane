use h2ai_constraints::versioned::{RepairProvenance, VersionConflictError, VersionedSpec};

#[test]
fn repair_provenance_round_trips() {
    let p = RepairProvenance {
        triggered_by_task: "task-1".into(),
        triggered_at_ms: 1_700_000_000_000,
        instability_score: 0.034,
        original_check_index: 0,
        original_check_text: "old".into(),
        simplified_check_text: "new".into(),
        validation_consistency: 0.95,
    };
    let json = serde_json::to_string(&p).unwrap();
    let back: RepairProvenance = serde_json::from_str(&json).unwrap();
    assert_eq!(back.triggered_by_task, "task-1");
    assert!((back.instability_score - 0.034).abs() < 1e-9);
}

#[test]
fn versioned_spec_default_version_is_one() {
    let spec = VersionedSpec {
        spec: h2ai_constraints::spec::SemanticSpec::default_for_test("C-1"),
        provenance: None,
    };
    assert_eq!(spec.spec.version, 1);
}

#[test]
fn version_conflict_error_display() {
    let e = VersionConflictError {
        constraint_id: "C-1".into(),
        expected: 2,
        actual: 3,
    };
    assert_eq!(e.expected, 2);
    assert_eq!(e.actual, 3);
    // Test the Display output is meaningful
    let msg = e.to_string();
    assert!(
        msg.contains("C-1"),
        "should mention constraint id, got: {msg}"
    );
    assert!(
        msg.contains("2"),
        "should mention expected version, got: {msg}"
    );
    assert!(
        msg.contains("3"),
        "should mention actual version, got: {msg}"
    );
}
