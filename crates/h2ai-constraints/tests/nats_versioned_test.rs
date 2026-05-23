use h2ai_constraints::{
    nats_versioned::NatsVersionedSource,
    source::{ConstraintSource, InMemorySource},
    spec::SemanticSpec,
    versioned::{RepairProvenance, VersionedConstraintSource},
};

fn make_spec(id: &str, check: &str) -> SemanticSpec {
    let mut s = SemanticSpec::default_for_test(id);
    s.rubric.checks = vec![check.to_owned()];
    s.rubric.pass = "good proposal text".into();
    s
}

#[tokio::test]
async fn load_latest_versioned_falls_through_to_inner() {
    let inner = InMemorySource {
        specs: vec![make_spec("C-1", "Does it use atomic ops?")],
    };
    let store = NatsVersionedSource::new_in_memory(inner);
    let vs = store.load_latest_versioned("C-1").await.unwrap();
    assert_eq!(vs.spec.version, 1);
    assert_eq!(vs.spec.id, "C-1");
    assert!(vs.provenance.is_none());
}

#[tokio::test]
async fn create_next_version_increments_version() {
    let inner = InMemorySource {
        specs: vec![make_spec("C-1", "original check")],
    };
    let store = NatsVersionedSource::new_in_memory(inner);

    let prov = RepairProvenance {
        triggered_by_task: "t1".into(),
        triggered_at_ms: 0,
        instability_score: 0.03,
        original_check_index: 0,
        original_check_text: "original check".into(),
        simplified_check_text: "simplified check".into(),
        validation_consistency: 0.92,
    };
    let mut repaired = make_spec("C-1", "simplified check");
    repaired.version = 2;

    let new_version = store
        .create_next_version("C-1", 1, repaired.clone(), prov)
        .await
        .unwrap();
    assert_eq!(new_version, 2);

    let vs = store.load_latest_versioned("C-1").await.unwrap();
    assert_eq!(vs.spec.version, 2);
    assert_eq!(vs.spec.rubric.checks[0], "simplified check");
}

#[tokio::test]
async fn create_next_version_conflict_when_expected_wrong() {
    let inner = InMemorySource {
        specs: vec![make_spec("C-1", "check")],
    };
    let store = NatsVersionedSource::new_in_memory(inner);

    let prov = RepairProvenance {
        triggered_by_task: "t1".into(),
        triggered_at_ms: 0,
        instability_score: 0.03,
        original_check_index: 0,
        original_check_text: "check".into(),
        simplified_check_text: "better check".into(),
        validation_consistency: 0.90,
    };
    let repaired = make_spec("C-1", "better check");

    let err = store
        .create_next_version("C-1", 99, repaired, prov)
        .await
        .unwrap_err();
    assert_eq!(err.expected, 99);
    assert_eq!(err.actual, 1);
}

#[tokio::test]
async fn load_all_returns_updated_spec_after_repair() {
    let inner = InMemorySource {
        specs: vec![make_spec("C-1", "original")],
    };
    let store = NatsVersionedSource::new_in_memory(inner);

    let prov = RepairProvenance {
        triggered_by_task: "t1".into(),
        triggered_at_ms: 0,
        instability_score: 0.03,
        original_check_index: 0,
        original_check_text: "original".into(),
        simplified_check_text: "repaired".into(),
        validation_consistency: 0.95,
    };
    let repaired = make_spec("C-1", "repaired");
    store
        .create_next_version("C-1", 1, repaired, prov)
        .await
        .unwrap();

    let specs = store.load_all().unwrap();
    let c1 = specs.iter().find(|s| s.id == "C-1").unwrap();
    assert_eq!(c1.rubric.checks[0], "repaired");
}
