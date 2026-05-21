#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
use h2ai_autonomic::planner::{ProvisionInput, TopologyPlanner};
use h2ai_config::H2AIConfig;
use h2ai_types::config::{
    AdapterKind, AgentRole, AuditorConfig, ParetoWeights, ReviewGate, RoleSpec, TopologyKind,
};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{
    CoherencyCoefficients, EigenCalibration, MergeStrategy, TaskQuadrant, TauValue,
};

fn cc() -> CoherencyCoefficients {
    CoherencyCoefficients::new(0.1, 0.02, vec![0.8, 0.85, 0.9]).unwrap()
}

fn adapter() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://api.example.com".into(),
        api_key_env: "TEST_KEY".into(),
        model: None,
    }
}

fn auditor() -> AuditorConfig {
    AuditorConfig {
        adapter: adapter(),
        ..Default::default()
    }
}

fn two_roles() -> Vec<RoleSpec> {
    vec![
        RoleSpec {
            agent_id: "a".into(),
            role: AgentRole::Executor,
            tau: None,
            role_error_cost: None,
        },
        RoleSpec {
            agent_id: "b".into(),
            role: AgentRole::Evaluator,
            tau: None,
            role_error_cost: None,
        },
    ]
}

#[test]
fn planner_selects_team_swarm_when_weights_balanced() {
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    assert_eq!(event.topology_kind, TopologyKind::TeamSwarmHybrid);
}

#[test]
fn planner_selects_hierarchical_tree_when_containment_dominant() {
    let cc = cc();
    let weights = ParetoWeights::new(0.1, 0.8, 0.1).unwrap();
    let cfg = H2AIConfig::default();
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    assert!(matches!(
        event.topology_kind,
        TopologyKind::HierarchicalTree { .. }
    ));
}

#[test]
fn planner_selects_team_swarm_hybrid_when_review_gates_present() {
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let gates = vec![ReviewGate {
        reviewer: "b".into(),
        blocks: "a".into(),
    }];
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: gates,
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    assert_eq!(event.topology_kind, TopologyKind::TeamSwarmHybrid);
}

#[test]
fn planner_computes_positive_n_max_and_beta_eff() {
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    assert!(event.n_max > 0.0);
    assert!(event.beta_eff > 0.0);
}

#[test]
fn planner_creates_one_explorer_config_per_role_spec() {
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    assert_eq!(event.explorer_configs.len(), 2);
}

#[test]
fn planner_uses_role_default_tau_when_spec_has_none() {
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let specs = vec![RoleSpec {
        agent_id: "c".into(),
        role: AgentRole::Coordinator,
        tau: None,
        role_error_cost: None,
    }];
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &specs,
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    assert!((event.explorer_configs[0].tau.value() - 0.05).abs() < 1e-9);
}

#[test]
fn planner_uses_override_tau_when_spec_provides_one() {
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let specs = vec![RoleSpec {
        agent_id: "d".into(),
        role: AgentRole::Coordinator,
        tau: Some(TauValue::new(0.2).unwrap()),
        role_error_cost: None,
    }];
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &specs,
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    assert!((event.explorer_configs[0].tau.value() - 0.2).abs() < 1e-9);
}

#[test]
fn planner_selects_bft_consensus_when_evaluator_present() {
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let specs = vec![RoleSpec {
        agent_id: "e".into(),
        role: AgentRole::Evaluator,
        tau: None,
        role_error_cost: None,
    }];
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &specs,
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    assert_eq!(event.merge_strategy, MergeStrategy::ConsensusMedian);
}

#[test]
fn planner_eigen_caps_explorer_count_below_usl_ceiling() {
    // cc() gives n_max_usl ≈ 17 (high), eigen caps at n_pruned = 3.
    // role_specs has 6 roles so provisioning would produce 6 explorers without cap.
    // With eigen cap of 3, n_max is clamped to 3.0 — but explorer_configs reflects role_specs,
    // so we verify n_max in the event is ≤ 3.0 (eigen ceiling applied).
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let eigen = EigenCalibration {
        n_effective: 2.4,
        h_diversity: 0.7,
        eigenvalues: vec![3.0, 0.5, 0.3, 0.1, 0.1],
        n_pruned: 3,
    };
    let specs: Vec<RoleSpec> = (0..6)
        .map(|i| RoleSpec {
            agent_id: format!("exp_{i}"),
            role: AgentRole::Executor,
            tau: None,
            role_error_cost: None,
        })
        .collect();
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &specs,
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: Some(&eigen),
        task_quadrant: None,
    });
    // With eigen.n_pruned = 3 and n_max_usl ≈ 17, the ceiling applied is 3.0.
    assert!(
        event.n_max <= 3.0,
        "eigen ceiling must cap n_max at n_pruned=3, got n_max={}",
        event.n_max
    );
}

#[test]
fn planner_eigen_does_not_raise_below_usl_ceiling() {
    // When USL n_max is already smaller than eigen.n_pruned, eigen must not raise it.
    // Use high-contention CC: alpha=0.5, beta=0.1, cg=[0.0] → n_max_usl = 1.
    let cc = CoherencyCoefficients::new(0.49, 0.1, vec![0.0]).unwrap();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let eigen = EigenCalibration {
        n_effective: 4.0,
        h_diversity: 0.9,
        eigenvalues: vec![2.0, 1.0, 0.5, 0.3],
        n_pruned: 6,
    };
    let specs = vec![RoleSpec {
        agent_id: "a".into(),
        role: AgentRole::Executor,
        tau: None,
        role_error_cost: None,
    }];
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &specs,
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: Some(&eigen),
        task_quadrant: None,
    });
    let usl_n_max = cc.n_max();
    // eigen.n_pruned=6 > usl_n_max, so the ceiling must not raise n_max above USL.
    assert!(
        event.n_max <= usl_n_max,
        "eigen must not raise n_max above USL ceiling: usl={usl_n_max}, got={}",
        event.n_max
    );
}

#[test]
fn planner_precision_quadrant_caps_n_max_at_three() {
    // cc() gives n_max_usl >> 3. Precision must hard-cap at 3.0 regardless of eigen/USL.
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    // eigen allows up to 9 — Precision cap (3) must win.
    let eigen = EigenCalibration {
        n_effective: 7.0,
        h_diversity: 0.9,
        eigenvalues: vec![4.0, 2.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.02],
        n_pruned: 9,
    };
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: Some(&eigen),
        task_quadrant: Some(TaskQuadrant::Precision),
    });
    assert!(
        event.n_max <= 3.0,
        "Precision quadrant must cap n_max at 3.0, got {}",
        event.n_max
    );
    assert!(
        event.n_max >= 1.0,
        "Precision quadrant n_max must be at least 1.0, got {}",
        event.n_max
    );
}

#[test]
fn planner_complex_quadrant_bypasses_eigen_cap() {
    // cc() gives n_max_usl >> eigen.n_pruned. Complex must bypass eigen and use full USL n_max.
    // n_max_usl is computed context-aware when max_context_tokens is set in the config.
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let eigen = EigenCalibration {
        n_effective: 2.4,
        h_diversity: 0.7,
        eigenvalues: vec![3.0, 0.5, 0.3, 0.1, 0.1],
        n_pruned: 3,
    };
    // Mirror planner's n_max_usl calculation so the assertion stays correct
    // even when max_context_tokens changes in reference.toml.
    let usl_n_max = match cfg.max_context_tokens {
        Some(max_tokens) => cc.n_max_context_aware(
            cfg.explorer_max_tokens as f64,
            max_tokens as f64,
            cfg.context_pressure_gamma,
        ),
        None => cc.n_max(),
    };
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: Some(&eigen),
        task_quadrant: Some(TaskQuadrant::Complex),
    });
    // Complex bypasses eigen.n_pruned=3, so n_max must equal the uncapped USL value
    // (after any context-pressure reduction, but ignoring the eigen cap of 3).
    assert!(
        event.n_max > eigen.n_pruned as f64,
        "Complex quadrant must bypass eigen cap of {}: got {}",
        eigen.n_pruned,
        event.n_max
    );
    assert!(
        (event.n_max - usl_n_max).abs() < 0.5,
        "Complex quadrant n_max must equal USL value {usl_n_max}, got {}",
        event.n_max
    );
}

#[test]
fn planner_complex_quadrant_forces_ensemble_topology() {
    // Containment-heavy weights would normally select HierarchicalTree.
    // Complex quadrant must override to Ensemble regardless.
    let cc = cc();
    let weights = ParetoWeights::new(0.1, 0.8, 0.1).unwrap();
    let cfg = H2AIConfig::default();
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: Some(TaskQuadrant::Complex),
    });
    assert_eq!(
        event.topology_kind,
        TopologyKind::Ensemble,
        "Complex quadrant must force Ensemble topology, got {:?}",
        event.topology_kind
    );
}

#[test]
fn planner_complex_quadrant_force_topology_overrides_ensemble() {
    // When force_topology is set, it takes precedence even over Complex quadrant forcing.
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig::default();
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: Some(TopologyKind::TeamSwarmHybrid),
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: Some(TaskQuadrant::Complex),
    });
    assert_eq!(
        event.topology_kind,
        TopologyKind::TeamSwarmHybrid,
        "force_topology must override Complex quadrant Ensemble forcing"
    );
}

// ── select_topology edge cases (tested via provision) ────────────────────────

#[test]
fn select_topology_diversity_heavy_gives_team_swarm() {
    // weights(throughput=0.1, containment=0.1, diversity=0.8) → TeamSwarmHybrid
    let cc = cc();
    let weights = ParetoWeights::new(0.1, 0.1, 0.8).unwrap();
    let cfg = H2AIConfig::default();
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    assert!(
        matches!(event.topology_kind, TopologyKind::TeamSwarmHybrid),
        "diversity-heavy weights → TeamSwarmHybrid, got {:?}",
        event.topology_kind
    );
}

#[test]
fn planner_no_max_context_tokens_uses_n_max() {
    // Line 104: max_context_tokens = None → fallback to cc.n_max()
    let cc = cc();
    let weights = ParetoWeights::new(0.34, 0.33, 0.33).unwrap();
    let cfg = H2AIConfig {
        max_context_tokens: None,
        ..H2AIConfig::default()
    };
    let (event, _) = TopologyPlanner::provision(ProvisionInput {
        task_id: TaskId::new(),
        cc: &cc,
        pareto_weights: &weights,
        role_specs: &two_roles(),
        review_gates: vec![],
        auditor_config: auditor(),
        explorer_adapter: adapter(),
        force_topology: None,
        retry_count: 0,
        cfg: &cfg,
        eigen: None,
        task_quadrant: None,
    });
    let expected_n_max = cc.n_max();
    assert!(
        event.n_max <= expected_n_max + 0.5,
        "no context limit → n_max should use cc.n_max()={expected_n_max}, got {}",
        event.n_max
    );
}
