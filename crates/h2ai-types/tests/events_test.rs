use chrono::Utc;
use h2ai_types::config::{AdapterKind, AuditorConfig, ExplorerConfig, ParetoWeights, TopologyKind};
use h2ai_types::events::*;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::{
    CoherencyCoefficients, CoordinationThreshold, MergeStrategy, MultiplicationConditionFailure,
    RoleErrorCost, TauValue,
};

fn task_id() -> TaskId {
    TaskId::new()
}
fn explorer_id() -> ExplorerId {
    ExplorerId::new()
}
fn cloud_adapter() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://api.example.com".into(),
        api_key_env: "API_KEY".into(),
    }
}
fn calibration() -> CoherencyCoefficients {
    CoherencyCoefficients::new(0.12, 0.020, vec![0.6, 0.7, 0.65]).unwrap()
}

#[test]
fn calibration_completed_event_serde_round_trip() {
    let cc = calibration();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let e = CalibrationCompletedEvent {
        calibration_id: task_id(),
        coefficients: cc,
        coordination_threshold: theta,
        ensemble: None,
        eigen: None,
        timestamp: Utc::now(),
        pairwise_beta: None,
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: CalibrationCompletedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(e.coefficients.alpha, back.coefficients.alpha);
}

#[test]
fn task_bootstrapped_event_includes_j_eff() {
    let e = TaskBootstrappedEvent {
        task_id: task_id(),
        system_context: "You must follow ADR-004.".into(),
        pareto_weights: ParetoWeights::new(0.5, 0.3, 0.2).unwrap(),
        j_eff: 0.72,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: TaskBootstrappedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(e.system_context, back.system_context);
    assert_eq!(e.j_eff, back.j_eff);
}

#[test]
fn topology_provisioned_event_includes_physics_fields() {
    let cc = calibration();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let role_costs = vec![
        RoleErrorCost::new(0.3).unwrap(),
        RoleErrorCost::new(0.7).unwrap(),
    ];
    let merge_strategy = MergeStrategy::from_role_costs(&role_costs, 0.85, 0.95, 0);
    let e = TopologyProvisionedEvent {
        task_id: task_id(),
        topology_kind: TopologyKind::Ensemble,
        explorer_configs: vec![
            ExplorerConfig {
                explorer_id: explorer_id(),
                tau: TauValue::new(0.2).unwrap(),
                adapter: cloud_adapter(),
                role: None,
            },
            ExplorerConfig {
                explorer_id: explorer_id(),
                tau: TauValue::new(0.9).unwrap(),
                adapter: cloud_adapter(),
                role: None,
            },
        ],
        auditor_config: AuditorConfig {
            adapter: cloud_adapter(),
            ..Default::default()
        },
        n_max: 4.2,
        interface_n_max: None,
        beta_eff: 0.03,
        role_error_costs: role_costs,
        merge_strategy,
        coordination_threshold: theta,
        review_gates: vec![],
        retry_count: 0,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: TopologyProvisionedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(e.explorer_configs.len(), back.explorer_configs.len());
    assert_eq!(back.merge_strategy, MergeStrategy::ScoreOrdered);
    assert_eq!(e.retry_count, back.retry_count);
}

#[test]
fn branch_pruned_event_includes_error_cost() {
    let e = BranchPrunedEvent {
        task_id: task_id(),
        explorer_id: explorer_id(),
        reason: "Violates ADR-004: Stateless Auth requirement".into(),
        constraint_error_cost: RoleErrorCost::new(0.85).unwrap(),
        violated_constraints: vec![],
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: BranchPrunedEvent = serde_json::from_str(&json).unwrap();
    assert!(back.reason.contains("ADR-004"));
    assert_eq!(back.constraint_error_cost.value(), 0.85);
}

#[test]
fn multiplication_condition_failed_event_names_failing_condition() {
    let e = MultiplicationConditionFailedEvent {
        task_id: task_id(),
        failure: MultiplicationConditionFailure::InsufficientDecorrelation {
            actual: 0.94,
            threshold: 0.9,
        },
        retry_count: 0,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: MultiplicationConditionFailedEvent = serde_json::from_str(&json).unwrap();
    assert!(matches!(
        back.failure,
        MultiplicationConditionFailure::InsufficientDecorrelation { .. }
    ));
}

#[test]
fn semilattice_compiled_event_includes_merge_strategy() {
    let eid = explorer_id();
    let e = SemilatticeCompiledEvent {
        task_id: task_id(),
        valid_proposals: vec![eid.clone()],
        pruned_proposals: vec![(eid, "ADR-004 violation".into())],
        merge_strategy: MergeStrategy::ScoreOrdered,
        timestamp: Utc::now(),
        merge_elapsed_secs: None,
        n_input_proposals: 0,
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: SemilatticeCompiledEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(back.merge_strategy, MergeStrategy::ScoreOrdered);
}

#[test]
fn consensus_required_event_serde_round_trip() {
    let e = ConsensusRequiredEvent {
        task_id: task_id(),
        max_role_error_cost: RoleErrorCost::new(0.91).unwrap(),
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: ConsensusRequiredEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(back.max_role_error_cost.value(), 0.91);
}

#[test]
fn task_failed_event_may_include_multiplication_failure() {
    let e = TaskFailedEvent {
        task_id: task_id(),
        pruned_events: vec![],
        topologies_tried: vec![TopologyKind::Ensemble],
        tau_values_tried: vec![vec![0.2, 0.6, 0.9]],
        multiplication_condition_failure: Some(
            MultiplicationConditionFailure::InsufficientCompetence {
                actual: 0.42,
                required: 0.5,
            },
        ),
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: TaskFailedEvent = serde_json::from_str(&json).unwrap();
    assert!(back.multiplication_condition_failure.is_some());
}

#[test]
fn h2ai_event_enum_wraps_all_17_events() {
    let cc = calibration();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let role_costs = vec![RoleErrorCost::new(0.5).unwrap()];
    let merge = MergeStrategy::from_role_costs(&role_costs, 0.85, 0.95, 0);

    let events: Vec<H2AIEvent> = vec![
        H2AIEvent::CalibrationCompleted(CalibrationCompletedEvent {
            calibration_id: task_id(),
            coefficients: cc,
            coordination_threshold: theta.clone(),
            ensemble: None,
            eigen: None,
            timestamp: Utc::now(),
            pairwise_beta: None,
        }),
        H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
            task_id: task_id(),
            system_context: "ctx".into(),
            pareto_weights: ParetoWeights::new(0.5, 0.3, 0.2).unwrap(),
            j_eff: 0.65,
            timestamp: Utc::now(),
        }),
        H2AIEvent::TopologyProvisioned(TopologyProvisionedEvent {
            task_id: task_id(),
            topology_kind: TopologyKind::Ensemble,
            explorer_configs: vec![],
            auditor_config: AuditorConfig {
                adapter: cloud_adapter(),
                ..Default::default()
            },
            n_max: 3.0,
            interface_n_max: None,
            beta_eff: 0.05,
            role_error_costs: vec![RoleErrorCost::new(0.5).unwrap()],
            merge_strategy: merge.clone(),
            coordination_threshold: theta.clone(),
            review_gates: vec![],
            retry_count: 0,
            timestamp: Utc::now(),
        }),
        H2AIEvent::MultiplicationConditionFailed(MultiplicationConditionFailedEvent {
            task_id: task_id(),
            failure: MultiplicationConditionFailure::InsufficientCompetence {
                actual: 0.4,
                required: 0.5,
            },
            retry_count: 0,
            timestamp: Utc::now(),
        }),
        H2AIEvent::Proposal(ProposalEvent {
            task_id: task_id(),
            explorer_id: explorer_id(),
            tau: TauValue::new(0.5).unwrap(),
            raw_output: "out".into(),
            token_cost: 10,
            adapter_kind: cloud_adapter(),
            timestamp: Utc::now(),
        }),
        H2AIEvent::ProposalFailed(ProposalFailedEvent {
            task_id: task_id(),
            explorer_id: explorer_id(),
            reason: ProposalFailureReason::Timeout,
            timestamp: Utc::now(),
        }),
        H2AIEvent::GenerationPhaseCompleted(GenerationPhaseCompletedEvent {
            task_id: task_id(),
            total_explorers: 2,
            successful: 1,
            failed: 1,
            timestamp: Utc::now(),
        }),
        H2AIEvent::ReviewGateTriggered(ReviewGateTriggeredEvent {
            task_id: task_id(),
            gate_id: "g1".into(),
            blocked_explorer_id: explorer_id(),
            reviewer_explorer_id: explorer_id(),
            timestamp: Utc::now(),
        }),
        H2AIEvent::ReviewGateBlocked(ReviewGateBlockedEvent {
            task_id: task_id(),
            gate_id: "g1".into(),
            blocked_explorer_id: explorer_id(),
            reviewer_explorer_id: explorer_id(),
            rejection_reason: "ADR-007 violation".into(),
            timestamp: Utc::now(),
        }),
        H2AIEvent::Validation(ValidationEvent {
            task_id: task_id(),
            explorer_id: explorer_id(),
            timestamp: Utc::now(),
        }),
        H2AIEvent::BranchPruned(BranchPrunedEvent {
            task_id: task_id(),
            explorer_id: explorer_id(),
            reason: "ADR-004".into(),
            constraint_error_cost: RoleErrorCost::new(0.85).unwrap(),
            violated_constraints: vec![],
            timestamp: Utc::now(),
        }),
        H2AIEvent::ZeroSurvival(ZeroSurvivalEvent {
            task_id: task_id(),
            retry_count: 0,
            timestamp: Utc::now(),
        }),
        H2AIEvent::InterfaceSaturationWarning(InterfaceSaturationWarningEvent {
            task_id: task_id(),
            active_subtasks: 4,
            interface_n_max: 5.0,
            saturation_ratio: 0.8,
            timestamp: Utc::now(),
        }),
        H2AIEvent::ConsensusRequired(ConsensusRequiredEvent {
            task_id: task_id(),
            max_role_error_cost: RoleErrorCost::new(0.91).unwrap(),
            timestamp: Utc::now(),
        }),
        H2AIEvent::SemilatticeCompiled(SemilatticeCompiledEvent {
            task_id: task_id(),
            valid_proposals: vec![],
            pruned_proposals: vec![],
            merge_strategy: merge,
            timestamp: Utc::now(),
            merge_elapsed_secs: None,
            n_input_proposals: 0,
        }),
        H2AIEvent::MergeResolved(MergeResolvedEvent {
            task_id: task_id(),
            resolved_output: "final".into(),
            timestamp: Utc::now(),
        }),
        H2AIEvent::TaskFailed(TaskFailedEvent {
            task_id: task_id(),
            pruned_events: vec![],
            topologies_tried: vec![],
            tau_values_tried: vec![],
            multiplication_condition_failure: None,
            timestamp: Utc::now(),
        }),
    ];
    assert_eq!(events.len(), 17);
}

#[test]
fn review_gate_triggered_event_serde_round_trip() {
    let e = ReviewGateTriggeredEvent {
        task_id: task_id(),
        gate_id: "gate_eval_impl".into(),
        blocked_explorer_id: explorer_id(),
        reviewer_explorer_id: explorer_id(),
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: ReviewGateTriggeredEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(e.gate_id, back.gate_id);
}

#[test]
fn review_gate_blocked_event_serde_round_trip() {
    let e = ReviewGateBlockedEvent {
        task_id: task_id(),
        gate_id: "gate_eval_impl".into(),
        blocked_explorer_id: explorer_id(),
        reviewer_explorer_id: explorer_id(),
        rejection_reason: "violates ADR-007".into(),
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: ReviewGateBlockedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(e.rejection_reason, back.rejection_reason);
}

#[test]
fn interface_saturation_warning_event_serde_round_trip() {
    let e = InterfaceSaturationWarningEvent {
        task_id: task_id(),
        active_subtasks: 4,
        interface_n_max: 5.0,
        saturation_ratio: 0.8,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: InterfaceSaturationWarningEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(e.active_subtasks, back.active_subtasks);
    assert!((e.saturation_ratio - back.saturation_ratio).abs() < 1e-9);
}

#[test]
fn h2ai_event_serde_preserves_variant_tag() {
    let original = H2AIEvent::ZeroSurvival(ZeroSurvivalEvent {
        task_id: task_id(),
        retry_count: 2,
        timestamp: Utc::now(),
    });
    let json = serde_json::to_string(&original).unwrap();
    assert!(json.contains("\"event_type\":\"ZeroSurvival\""));
    let back: H2AIEvent = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, H2AIEvent::ZeroSurvival(_)));
}
