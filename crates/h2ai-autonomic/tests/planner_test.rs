use h2ai_autonomic::planner::{ProvisionInput, TopologyPlanner};
use h2ai_config::H2AIConfig;
use h2ai_types::config::{
    AdapterKind, AgentRole, AuditorConfig, ParetoWeights, ReviewGate, RoleSpec, TopologyKind,
};
use h2ai_types::identity::TaskId;
use h2ai_types::physics::{CoherencyCoefficients, EigenCalibration, MergeStrategy, TauValue};

fn cc() -> CoherencyCoefficients {
    CoherencyCoefficients::new(0.1, 0.02, vec![0.8, 0.85, 0.9]).unwrap()
}

fn adapter() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://api.example.com".into(),
        api_key_env: "TEST_KEY".into(),
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
    });
    let usl_n_max = cc.n_max();
    // eigen.n_pruned=6 > usl_n_max, so the ceiling must not raise n_max above USL.
    assert!(
        event.n_max <= usl_n_max,
        "eigen must not raise n_max above USL ceiling: usl={usl_n_max}, got={}",
        event.n_max
    );
}
