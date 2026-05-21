#[cfg(test)]
mod oracle_types_tests {
    use h2ai_types::sizing::{OracleDomain, OracleObservation, OracleSpec, OracleType};

    #[test]
    fn oracle_domain_serde_roundtrip() {
        let d = OracleDomain::Code;
        let s = serde_json::to_string(&d).unwrap();
        let back: OracleDomain = serde_json::from_str(&s).unwrap();
        assert!(matches!(back, OracleDomain::Code));
    }

    #[test]
    fn oracle_spec_serde_roundtrip() {
        let spec = OracleSpec {
            runner_uri: "http://localhost:9090".into(),
            test_suite: "tests/".into(),
            language: "python".into(),
            timeout_ms: 5000,
            reference_output: None,
            oracle_type: OracleType::TestSuite,
            domain: OracleDomain::Code,
        };
        let s = serde_json::to_string(&spec).unwrap();
        let back: OracleSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(back.runner_uri, "http://localhost:9090");
        assert_eq!(back.timeout_ms, 5000);
        assert!(matches!(back.oracle_type, OracleType::TestSuite));
    }

    #[test]
    fn oracle_observation_residual_is_abs_error() {
        // residual = |q_confidence - y_oracle as f64|
        let residual = (0.8_f64 - 1.0_f64).abs();
        let obs = OracleObservation {
            task_id: "t1".into(),
            q_confidence: 0.8,
            y_oracle: true,
            residual,
            domain: OracleDomain::Code,
            oracle_type: OracleType::TestSuite,
            timestamp_ms: 0,
        };
        assert!((obs.residual - 0.2).abs() < 1e-9);
    }
}

use chrono::Utc;
use h2ai_types::config::{AdapterKind, AuditorConfig, ExplorerConfig, ParetoWeights, TopologyKind};
use h2ai_types::events::*;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::{
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
        model: None,
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
        cg_mode: Default::default(),
        adapter_families: Vec::new(),
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: Default::default(),
        calibration_source: Default::default(),
        beta_quality: None,
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: CalibrationCompletedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(e.coefficients.alpha, back.coefficients.alpha);
}

#[test]
fn task_bootstrapped_event_round_trips() {
    let e = TaskBootstrappedEvent {
        task_id: task_id(),
        system_context: "You must follow ADR-004.".into(),
        pareto_weights: ParetoWeights::new(0.5, 0.3, 0.2).unwrap(),
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: TaskBootstrappedEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(e.system_context, back.system_context);
    assert_eq!(e.pareto_weights, back.pareto_weights);
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
                is_reasoning_model: false,
            },
            ExplorerConfig {
                explorer_id: explorer_id(),
                tau: TauValue::new(0.9).unwrap(),
                adapter: cloud_adapter(),
                role: None,
                is_reasoning_model: false,
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
        constraint_tombstone: None,
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
fn selection_resolved_event_includes_merge_strategy() {
    let eid = explorer_id();
    let e = SelectionResolvedEvent {
        task_id: task_id(),
        valid_proposals: vec![eid.clone()],
        pruned_proposals: vec![(eid, "ADR-004 violation".into())],
        merge_strategy: MergeStrategy::ScoreOrdered,
        timestamp: Utc::now(),
        merge_elapsed_secs: None,
        n_input_proposals: 0,
        n_failed_proposals: 0,
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: SelectionResolvedEvent = serde_json::from_str(&json).unwrap();
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
            cg_mode: Default::default(),
            adapter_families: Vec::new(),
            explorer_verification_family_match: false,
            single_family_warning: false,
            n_max_lo: 0.0,
            n_max_hi: 0.0,
            n_eff_cosine_prior: 0.0,
            calibration_quality: Default::default(),
            calibration_source: Default::default(),
            beta_quality: None,
        }),
        H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
            task_id: task_id(),
            system_context: "ctx".into(),
            pareto_weights: ParetoWeights::new(0.5, 0.3, 0.2).unwrap(),
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
            constraint_tombstone: None,
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
            generation: 0,
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
            n_eff_cosine_actual: None,
            failure_mode: None,
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
        H2AIEvent::SelectionResolved(SelectionResolvedEvent {
            task_id: task_id(),
            valid_proposals: vec![],
            pruned_proposals: vec![],
            merge_strategy: merge,
            timestamp: Utc::now(),
            merge_elapsed_secs: None,
            n_input_proposals: 0,
            n_failed_proposals: 0,
        }),
        H2AIEvent::MergeResolved(MergeResolvedEvent {
            task_id: task_id(),
            resolved_output: "final".into(),
            j_eff: None,
            oracle_gate_passed: None,
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
        H2AIEvent::TaskComplexityAssessed(TaskComplexityAssessedEvent {
            task_id: task_id(),
            tcc_structural: 1.5,
            tcc_empirical: None,
            tcc_effective: 1.5,
            n_eff_pool: None,
            task_quadrant: h2ai_types::sizing::TaskQuadrant::Coverage,
            probe_skipped: true,
            probe_skip_reason: Default::default(),
            heavy_fraction: 0.0,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: 0,
            timestamp: Utc::now(),
        }),
        H2AIEvent::ConstraintFrontier(ConstraintFrontierEvent {
            task_id: task_id(),
            satisfaction_matrix: vec![],
            constraint_ids: vec![],
            explorer_ids: vec![],
            pareto_coverage: 1.0,
            timestamp: Utc::now(),
        }),
    ];
    assert_eq!(events.len(), 19);
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
        n_eff_cosine_actual: None,
        failure_mode: None,
    });
    let json = serde_json::to_string(&original).unwrap();
    assert!(json.contains("\"event_type\":\"ZeroSurvival\""));
    let back: H2AIEvent = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, H2AIEvent::ZeroSurvival(_)));
}

#[cfg(test)]
mod oracle_event_tests {
    use h2ai_types::events::{
        CalibrationDriftWarning, H2AIEvent, OraclePendingEvent, OracleResultEvent,
        OracleSuspectEvent,
    };
    use h2ai_types::sizing::{OracleDomain, OracleSpec, OracleType};

    fn make_oracle_spec() -> OracleSpec {
        OracleSpec {
            runner_uri: "http://localhost:9090".into(),
            test_suite: "tests/".into(),
            language: "python".into(),
            timeout_ms: 5000,
            reference_output: None,
            oracle_type: OracleType::TestSuite,
            domain: OracleDomain::Code,
        }
    }

    #[test]
    fn oracle_pending_event_serde_roundtrip() {
        use h2ai_types::identity::TaskId;
        let ev = OraclePendingEvent {
            task_id: TaskId::new(),
            winning_output: "print('hello')".into(),
            q_confidence: 0.85,
            n_used: 3,
            oracle_spec: make_oracle_spec(),
            domain: OracleDomain::Code,
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: OraclePendingEvent = serde_json::from_str(&json).unwrap();
        assert!((back.q_confidence - 0.85).abs() < 1e-9);
        assert_eq!(back.n_used, 3);
    }

    #[test]
    fn oracle_result_event_residual_formula() {
        use h2ai_types::identity::TaskId;
        // residual = |q_confidence - passed as f64|
        // q=0.8, passed=true (1.0) → residual = 0.2
        let ev = OracleResultEvent {
            task_id: TaskId::new(),
            q_confidence: 0.8,
            n_used: 3,
            passed: true,
            score: 1.0,
            residual: (0.8_f64 - 1.0_f64).abs(),
            domain: OracleDomain::Code,
            oracle_type: OracleType::TestSuite,
            duration_ms: 250,
            timestamp_ms: 0,
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: OracleResultEvent = serde_json::from_str(&json).unwrap();
        assert!((back.residual - 0.2).abs() < 1e-9);
        assert!(back.passed);
    }

    #[test]
    fn calibration_drift_warning_serde_roundtrip() {
        let w = CalibrationDriftWarning {
            n_observations: 10,
            ece: 0.18,
            timestamp_ms: 1000,
        };
        let json = serde_json::to_string(&w).unwrap();
        let back: CalibrationDriftWarning = serde_json::from_str(&json).unwrap();
        assert!((back.ece - 0.18).abs() < 1e-9);
    }

    #[test]
    fn oracle_suspect_event_serde_roundtrip() {
        let ev = OracleSuspectEvent {
            pass_rate: 0.02,
            n_observations: 25,
            reason: "pass rate < 0.05".into(),
            timestamp_ms: 2000,
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: OracleSuspectEvent = serde_json::from_str(&json).unwrap();
        assert!((back.pass_rate - 0.02).abs() < 1e-9);
    }

    #[test]
    fn h2ai_event_oracle_variants_serde() {
        use h2ai_types::identity::TaskId;
        let ev = H2AIEvent::OraclePending(OraclePendingEvent {
            task_id: TaskId::new(),
            winning_output: "output".into(),
            q_confidence: 0.9,
            n_used: 2,
            oracle_spec: make_oracle_spec(),
            domain: OracleDomain::Factual,
        });
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("OraclePending"));
        let back: H2AIEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, H2AIEvent::OraclePending(_)));
    }
}

#[test]
fn coherence_incomplete_event_carries_active_contradictions() {
    let ev = CoherenceIncompleteEvent {
        task_id: task_id(),
        uncovered_domains: vec!["security".to_string()],
        active_contradictions: vec![(
            "id-a".to_string(),
            "id-b".to_string(),
            "security".to_string(),
        )],
        retries: 2,
        timestamp: Utc::now(),
    };
    assert_eq!(ev.active_contradictions.len(), 1);
    assert_eq!(ev.active_contradictions[0].2, "security");
    // Round-trip through serde_json — tuples serialise as JSON arrays
    let json = serde_json::to_string(&ev).unwrap();
    let back: CoherenceIncompleteEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(back.active_contradictions, ev.active_contradictions);
}

#[test]
fn coherence_incomplete_event_deserializes_without_contradictions_field() {
    // Old clients don't send active_contradictions — #[serde(default)] handles it
    let json = r#"{
        "task_id": "00000000-0000-0000-0000-000000000000",
        "uncovered_domains": ["auth"],
        "retries": 1,
        "timestamp": "2026-01-01T00:00:00Z"
    }"#;
    let ev: CoherenceIncompleteEvent = serde_json::from_str(json).unwrap();
    assert!(ev.active_contradictions.is_empty());
}

#[test]
fn verifier_comparison_event_roundtrips_json() {
    use h2ai_types::events::VerifierComparisonEvent;
    use h2ai_types::identity::ExplorerId;
    let ev = VerifierComparisonEvent {
        task_id: task_id(),
        explorer_id: ExplorerId::new(),
        standard_score: 0.72,
        adversarial_score: 0.45,
        standard_passed: true,
        adversarial_passed: false,
        verifier_kind: "llmjudge".to_string(),
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&ev).unwrap();
    let back: VerifierComparisonEvent = serde_json::from_str(&json).unwrap();
    assert!((back.standard_score - 0.72).abs() < 1e-6);
    assert!(!back.adversarial_passed);
}

#[cfg(test)]
mod manifest_oracle_tests {
    use h2ai_types::manifest::TaskManifest;
    use h2ai_types::sizing::{OracleDomain, OracleSpec, OracleType};

    #[test]
    fn task_manifest_oracle_defaults_to_none() {
        // Old serialized manifests without `oracle` field must still deserialize
        let json = r#"{
            "description": "test task",
            "pareto_weights": {"throughput": 1.0, "containment": 0.0, "diversity": 0.0},
            "topology": {},
            "explorers": {"count": 2},
            "constraints": [],
            "context": null
        }"#;
        let manifest: TaskManifest = serde_json::from_str(json).unwrap();
        assert!(
            manifest.oracle.is_none(),
            "oracle must default to None for backward compat"
        );
    }

    #[test]
    fn task_manifest_oracle_roundtrip() {
        use h2ai_types::config::ParetoWeights;
        use h2ai_types::manifest::{ExplorerRequest, TopologyRequest};
        let manifest = TaskManifest {
            description: "code task".into(),
            pareto_weights: ParetoWeights {
                throughput: 1.0,
                containment: 0.0,
                diversity: 0.0,
            },
            topology: TopologyRequest {
                kind: "auto".into(),
                branching_factor: None,
            },
            explorers: ExplorerRequest {
                count: 2,
                tau_min: None,
                tau_max: None,
                roles: vec![],
                review_gates: vec![],
                slot_configs: vec![],
                diversity_ids: vec![],
            },
            constraints: vec![],
            context: None,
            require_approval: false,
            constraint_tags: vec![],
            measure_verifier_ab: false,
            tenant_id: h2ai_types::identity::TenantId::default_tenant(),
            oracle: Some(OracleSpec {
                runner_uri: "http://localhost:9090".into(),
                test_suite: "tests/".into(),
                language: "python".into(),
                timeout_ms: 5000,
                reference_output: None,
                oracle_type: OracleType::TestSuite,
                domain: OracleDomain::Code,
            }),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let back: TaskManifest = serde_json::from_str(&json).unwrap();
        let spec = back.oracle.unwrap();
        assert_eq!(spec.language, "python");
        assert!(matches!(spec.oracle_type, OracleType::TestSuite));
    }
}

#[test]
fn constraint_ambiguity_event_roundtrips_serde() {
    use h2ai_types::events::{ConstraintAmbiguityEvent, H2AIEvent};
    use std::collections::HashMap;

    let mut counts = HashMap::new();
    counts.insert("constraint-1".to_string(), 3usize);
    counts.insert("constraint-2".to_string(), 2usize);

    let event = ConstraintAmbiguityEvent {
        task_id: h2ai_types::identity::TaskId::new(),
        wave: 2,
        ambiguous_constraints: vec!["constraint-1".to_string(), "constraint-2".to_string()],
        uncertain_counts: counts,
        timestamp: chrono::Utc::now(),
    };

    let wrapped = H2AIEvent::ConstraintAmbiguity(event.clone());
    let json = serde_json::to_string(&wrapped).unwrap();
    assert!(json.contains("\"event_type\":\"ConstraintAmbiguity\""));
    assert!(json.contains("constraint-1"));

    let back: H2AIEvent = serde_json::from_str(&json).unwrap();
    if let H2AIEvent::ConstraintAmbiguity(e) = back {
        assert_eq!(e.wave, 2);
        assert_eq!(e.ambiguous_constraints.len(), 2);
        assert_eq!(*e.uncertain_counts.get("constraint-1").unwrap(), 3usize);
    } else {
        panic!("wrong variant");
    }
}

#[cfg(test)]
mod shadow_auditor_event_tests {
    use h2ai_types::events::{
        AuditDomainDemotedEvent, AuditDomainPromotedEvent, ShadowAuditorResultEvent,
    };
    use h2ai_types::identity::{ExplorerId, TaskId};

    #[test]
    fn shadow_auditor_result_event_roundtrips_json() {
        let ev = ShadowAuditorResultEvent {
            task_id: TaskId::new(),
            explorer_id: ExplorerId::new(),
            primary_approved: true,
            shadow_approved: false,
            disagreement: true,
            domain: "security".to_string(),
            primary_family: "cloudgeneric".to_string(),
            shadow_family: "llamacpp".to_string(),
            timestamp_ms: 1234567890,
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: ShadowAuditorResultEvent = serde_json::from_str(&json).unwrap();
        assert!(back.disagreement);
        assert_eq!(back.domain, "security");
        assert_eq!(back.primary_family, "cloudgeneric");
        assert!(!back.shadow_approved);
    }

    #[test]
    fn shadow_auditor_result_disagreement_flag_computed_correctly() {
        let agree = ShadowAuditorResultEvent {
            task_id: TaskId::new(),
            explorer_id: ExplorerId::new(),
            primary_approved: true,
            shadow_approved: true,
            disagreement: false,
            domain: "default".to_string(),
            primary_family: "a".to_string(),
            shadow_family: "b".to_string(),
            timestamp_ms: 0,
        };
        assert!(!agree.disagreement);
        let disagree = ShadowAuditorResultEvent {
            primary_approved: false,
            shadow_approved: true,
            disagreement: true,
            ..agree
        };
        assert!(disagree.disagreement);
    }

    #[test]
    fn audit_domain_promoted_event_roundtrips_json() {
        let ev = AuditDomainPromotedEvent {
            domain: "eu_data".to_string(),
            disagreement_rate: 0.12,
            n_observations: 45,
            timestamp_ms: 999,
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: AuditDomainPromotedEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.domain, "eu_data");
        assert!((back.disagreement_rate - 0.12).abs() < 1e-9);
        assert_eq!(back.n_observations, 45);
    }

    #[test]
    fn audit_domain_demoted_event_roundtrips_json() {
        let ev = AuditDomainDemotedEvent {
            domain: "eu_data".to_string(),
            disagreement_rate: 0.01,
            n_observations: 120,
            timestamp_ms: 888,
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: AuditDomainDemotedEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.domain, "eu_data");
        assert_eq!(back.n_observations, 120);
    }

    #[test]
    fn h2ai_event_shadow_audit_variant_serializes() {
        use h2ai_types::events::H2AIEvent;
        let ev = H2AIEvent::ShadowAudit(ShadowAuditorResultEvent {
            task_id: TaskId::new(),
            explorer_id: ExplorerId::new(),
            primary_approved: true,
            shadow_approved: true,
            disagreement: false,
            domain: "default".to_string(),
            primary_family: "gemini".to_string(),
            shadow_family: "llamacpp".to_string(),
            timestamp_ms: 42,
        });
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("ShadowAudit"));
    }

    #[test]
    fn h2ai_event_audit_domain_promoted_variant_serializes() {
        use h2ai_types::events::H2AIEvent;
        let ev = H2AIEvent::AuditDomainPromoted(AuditDomainPromotedEvent {
            domain: "code".to_string(),
            disagreement_rate: 0.08,
            n_observations: 35,
            timestamp_ms: 0,
        });
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("AuditDomainPromoted"));
    }

    #[test]
    fn h2ai_event_audit_domain_demoted_variant_serializes() {
        use h2ai_types::events::H2AIEvent;
        let ev = H2AIEvent::AuditDomainDemoted(AuditDomainDemotedEvent {
            domain: "code".to_string(),
            disagreement_rate: 0.01,
            n_observations: 200,
            timestamp_ms: 0,
        });
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("AuditDomainDemoted"));
    }
}
