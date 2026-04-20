use h2ai_types::manifest::{
    MergeRequest, MergeResolution, TaskManifest, TaskStatusResponse, TopologyRequest,
};

#[test]
fn task_manifest_roundtrip() {
    let m = TaskManifest {
        description: "auth token rotation".into(),
        pareto_weights: h2ai_types::config::ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: h2ai_types::manifest::ExplorerRequest {
            count: 4,
            tau_min: Some(0.2),
            tau_max: Some(0.9),
            roles: vec![],
            review_gates: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    };
    let json = serde_json::to_string(&m).unwrap();
    let back: TaskManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.description, "auth token rotation");
    assert_eq!(back.explorers.count, 4);
}

#[test]
fn merge_request_select_roundtrip() {
    let req = MergeRequest {
        resolution: MergeResolution::Select,
        selected_proposals: vec!["exp_A".into(), "exp_B".into()],
        synthesis_notes: Some("combined approach".into()),
        final_output: None,
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: MergeRequest = serde_json::from_str(&json).unwrap();
    assert!(matches!(back.resolution, MergeResolution::Select));
    assert_eq!(back.selected_proposals.len(), 2);
}

#[test]
fn task_status_response_roundtrip() {
    let resp = TaskStatusResponse {
        task_id: "task_01".into(),
        status: "running".into(),
        phase: 3,
        phase_name: "ParallelGeneration".into(),
        explorers_completed: 2,
        explorers_total: 4,
        proposals_valid: 2,
        proposals_pruned: 0,
        autonomic_retries: 0,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let back: TaskStatusResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(back.phase, 3);
}
