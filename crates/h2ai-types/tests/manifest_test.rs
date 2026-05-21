use h2ai_types::manifest::{
    CalibrationAccepted, CotStyle, ExplorerSlotConfig, MergeRequest, MergeResolution, TaskAccepted,
    TaskManifest, TaskStatusResponse, TopologyRequest,
};

#[test]
fn explorer_slot_config_new_fields_default_empty() {
    let slot: ExplorerSlotConfig =
        serde_json::from_str(r#"{"cot_style": "step_by_step"}"#).unwrap();
    assert!(slot.focus_mandate.is_empty());
    assert!(slot.rejection_criteria.is_empty());
}

#[test]
fn explorer_slot_config_new_fields_round_trip() {
    let slot = ExplorerSlotConfig {
        role_frame: "You are a security engineer.".into(),
        cot_style: CotStyle::DevilsAdvocate,
        focus_mandate: "Responsible for CONSTRAINT-001 and CONSTRAINT-002.".into(),
        rejection_criteria: "Find: the most likely way an attacker exploits this.".into(),
        ..Default::default()
    };
    let json = serde_json::to_string(&slot).unwrap();
    let back: ExplorerSlotConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.role_frame, slot.role_frame);
    assert_eq!(back.focus_mandate, slot.focus_mandate);
    assert_eq!(back.rejection_criteria, slot.rejection_criteria);
}

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
            diversity_ids: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
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
            slot_configs: vec![
                ExplorerSlotConfig {
                    role_frame: "You are a systems architect.".into(),
                    cot_style: CotStyle::FirstPrinciples,
                    focus_mandate: String::new(),
                    rejection_criteria: "Irreversible technical debt.".into(),
                    ..Default::default()
                },
                ExplorerSlotConfig {
                    role_frame: "You are a security engineer.".into(),
                    cot_style: CotStyle::DevilsAdvocate,
                    focus_mandate: String::new(),
                    rejection_criteria: "Attacker exploit path.".into(),
                    ..Default::default()
                },
            ],
            diversity_ids: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let json = serde_json::to_string(&m).unwrap();
    let back: TaskManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.explorers.slot_configs.len(), 2);
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

// ── TopologyRequest::default ─────────────────────────────────────────────────

#[test]
fn topology_request_default_kind_is_auto() {
    let t = TopologyRequest::default();
    assert_eq!(t.kind, "auto");
    assert!(t.branching_factor.is_none());
}

// ── MergeResolution variants ──────────────────────────────────────────────────

#[test]
fn merge_resolution_synthesize_roundtrip() {
    let req = MergeRequest {
        resolution: MergeResolution::Synthesize,
        selected_proposals: vec![],
        synthesis_notes: Some("synthesized from A and B".into()),
        final_output: Some("combined output".into()),
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: MergeRequest = serde_json::from_str(&json).unwrap();
    assert!(matches!(back.resolution, MergeResolution::Synthesize));
}

#[test]
fn merge_resolution_reject_roundtrip() {
    let req = MergeRequest {
        resolution: MergeResolution::Reject,
        selected_proposals: vec![],
        synthesis_notes: None,
        final_output: None,
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: MergeRequest = serde_json::from_str(&json).unwrap();
    assert!(matches!(back.resolution, MergeResolution::Reject));
}

#[test]
fn merge_resolution_variants_are_distinct() {
    assert_ne!(MergeResolution::Select, MergeResolution::Synthesize);
    assert_ne!(MergeResolution::Select, MergeResolution::Reject);
    assert_ne!(MergeResolution::Synthesize, MergeResolution::Reject);
}

// ── TaskAccepted ──────────────────────────────────────────────────────────────

#[test]
#[allow(clippy::float_cmp)]
fn task_accepted_roundtrip() {
    let accepted = TaskAccepted {
        task_id: "task-abc-123".into(),
        status: "accepted".into(),
        events_url: "/tasks/task-abc-123/events".into(),
        topology_kind: "ensemble".into(),
        n_max: 5.0,
        interface_n_max: Some(3.0),
    };
    let json = serde_json::to_string(&accepted).unwrap();
    let back: TaskAccepted = serde_json::from_str(&json).unwrap();
    assert_eq!(back.task_id, "task-abc-123");
    assert_eq!(back.n_max, 5.0);
    assert_eq!(back.interface_n_max, Some(3.0));
}

#[test]
fn task_accepted_without_interface_n_max() {
    let accepted = TaskAccepted {
        task_id: "task-xyz".into(),
        status: "accepted".into(),
        events_url: "/tasks/task-xyz/events".into(),
        topology_kind: "auto".into(),
        n_max: 3.0,
        interface_n_max: None,
    };
    let json = serde_json::to_string(&accepted).unwrap();
    let back: TaskAccepted = serde_json::from_str(&json).unwrap();
    assert!(back.interface_n_max.is_none());
}

// ── CalibrationAccepted ───────────────────────────────────────────────────────

#[test]
fn calibration_accepted_roundtrip() {
    let cal = CalibrationAccepted {
        calibration_id: "cal-001".into(),
        status: "running".into(),
        events_url: "/calibrate/cal-001/events".into(),
        adapter_count: 3,
    };
    let json = serde_json::to_string(&cal).unwrap();
    let back: CalibrationAccepted = serde_json::from_str(&json).unwrap();
    assert_eq!(back.calibration_id, "cal-001");
    assert_eq!(back.adapter_count, 3);
}
