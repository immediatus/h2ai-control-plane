use h2ai_types::manifest::{
    CotStyle, ExplorerSlotConfig, MergeRequest, MergeResolution, TaskManifest, TaskStatusResponse,
    TopologyRequest,
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
            slot_configs: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
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

#[test]
fn cot_style_instructions_are_non_empty_except_none() {
    assert!(CotStyle::None.instruction().is_empty());
    assert!(!CotStyle::StepByStep.instruction().is_empty());
    assert!(!CotStyle::BackwardChaining.instruction().is_empty());
    assert!(!CotStyle::FirstPrinciples.instruction().is_empty());
    assert!(!CotStyle::DevilsAdvocate.instruction().is_empty());
}

#[test]
fn cot_style_roundtrips_json() {
    for style in [
        CotStyle::None,
        CotStyle::StepByStep,
        CotStyle::BackwardChaining,
        CotStyle::FirstPrinciples,
        CotStyle::DevilsAdvocate,
    ] {
        let json = serde_json::to_string(&style).unwrap();
        let back: CotStyle = serde_json::from_str(&json).unwrap();
        assert_eq!(back, style);
    }
}

#[test]
fn explorer_slot_config_diverse_defaults_has_four_entries() {
    let defaults = ExplorerSlotConfig::diverse_defaults();
    assert_eq!(defaults.len(), 4);
    // All four CoT styles are distinct — no two slots share a strategy
    let styles: Vec<_> = defaults.iter().map(|c| &c.cot_style).collect();
    let unique: std::collections::HashSet<_> = styles
        .iter()
        .map(|s| serde_json::to_string(s).unwrap())
        .collect();
    assert_eq!(
        unique.len(),
        4,
        "all four default slot strategies must be distinct"
    );
}

#[test]
fn manifest_with_slot_configs_roundtrips_json() {
    let m = TaskManifest {
        description: "auth token rotation".into(),
        pareto_weights: h2ai_types::config::ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: h2ai_types::manifest::ExplorerRequest {
            count: 3,
            tau_min: Some(0.2),
            tau_max: Some(0.8),
            roles: vec![],
            review_gates: vec![],
            slot_configs: ExplorerSlotConfig::diverse_defaults(),
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
    };
    let json = serde_json::to_string(&m).unwrap();
    let back: TaskManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.explorers.slot_configs.len(), 4);
    assert_eq!(
        back.explorers.slot_configs[1].cot_style,
        CotStyle::DevilsAdvocate
    );
}

#[test]
fn manifest_without_slot_configs_deserializes_to_empty() {
    // Old API clients that don't send slot_configs should get empty vec (backward compat)
    let json = r#"{
        "description": "test",
        "pareto_weights": {"throughput": 0.5, "containment": 0.3, "diversity": 0.2},
        "topology": {"kind": "auto"},
        "explorers": {"count": 2}
    }"#;
    let m: TaskManifest = serde_json::from_str(json).unwrap();
    assert!(m.explorers.slot_configs.is_empty());
}

#[test]
fn manifest_constraint_tags_defaults_empty() {
    let json = r#"{
        "description": "test",
        "pareto_weights": {"throughput": 0.5, "containment": 0.3, "diversity": 0.2},
        "topology": {"kind": "auto"},
        "explorers": {"count": 2}
    }"#;
    let m: TaskManifest = serde_json::from_str(json).unwrap();
    assert!(
        m.constraint_tags.is_empty(),
        "constraint_tags must default to empty vec"
    );
}

#[test]
fn manifest_constraint_tags_roundtrip() {
    let json = r#"{
        "description": "EU data task",
        "pareto_weights": {"throughput": 0.33, "containment": 0.33, "diversity": 0.34},
        "topology": {"kind": "auto"},
        "explorers": {"count": 3},
        "constraint_tags": ["eu_data", "financial_report"]
    }"#;
    let m: TaskManifest = serde_json::from_str(json).unwrap();
    assert_eq!(m.constraint_tags, vec!["eu_data", "financial_report"]);
}
