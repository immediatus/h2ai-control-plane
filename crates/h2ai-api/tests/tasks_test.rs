use h2ai_types::config::ParetoWeights;
use h2ai_types::manifest::TaskManifest;
use serde_json::json;

#[test]
fn pareto_weights_must_sum_to_one() {
    assert!(ParetoWeights::new(0.5, 0.5, 0.5).is_err());
    assert!(ParetoWeights::new(0.2, 0.3, 0.5).is_ok());
}

#[test]
fn task_manifest_deserialises_from_api_shape() {
    let raw = json!({
        "description": "Propose stateless auth",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 4, "tau_min": 0.2, "tau_max": 0.9}
    });
    let m: TaskManifest = serde_json::from_value(raw).unwrap();
    assert_eq!(m.topology.kind, "ensemble");
    assert_eq!(m.explorers.count, 4);
}
