use h2ai_types::config::{
    AdapterKind, AgentRole, AuditorConfig, ExplorerConfig,
    ParetoWeights, ReviewGate, RoleSpec, TopologyKind,
};
use std::path::PathBuf;

#[test]
fn pareto_weights_valid_when_sum_is_one() {
    let w = ParetoWeights::new(0.4, 0.4, 0.2);
    assert!(w.is_ok());
}

#[test]
fn pareto_weights_invalid_when_sum_is_not_one() {
    let w = ParetoWeights::new(0.5, 0.5, 0.5);
    assert!(w.is_err());
}

#[test]
fn pareto_weights_invalid_when_any_weight_is_negative() {
    let w = ParetoWeights::new(-0.1, 0.6, 0.5);
    assert!(w.is_err());
}

#[test]
fn pareto_weights_serde_round_trip() {
    let w = ParetoWeights::new(0.5, 0.3, 0.2).unwrap();
    let json = serde_json::to_string(&w).unwrap();
    let back: ParetoWeights = serde_json::from_str(&json).unwrap();
    assert_eq!(w, back);
}

#[test]
fn topology_kind_ensemble_serde_round_trip() {
    let t = TopologyKind::Ensemble;
    let json = serde_json::to_string(&t).unwrap();
    let back: TopologyKind = serde_json::from_str(&json).unwrap();
    assert_eq!(t, back);
}

#[test]
fn topology_kind_hierarchical_tree_serde_round_trip() {
    let t = TopologyKind::HierarchicalTree { branching_factor: Some(3) };
    let json = serde_json::to_string(&t).unwrap();
    let back: TopologyKind = serde_json::from_str(&json).unwrap();
    assert_eq!(t, back);
}

#[test]
fn topology_kind_team_swarm_hybrid_serde_round_trip() {
    let t = TopologyKind::TeamSwarmHybrid;
    let json = serde_json::to_string(&t).unwrap();
    let back: TopologyKind = serde_json::from_str(&json).unwrap();
    assert_eq!(t, back);
}

#[test]
fn agent_role_default_tau_and_ci() {
    assert_eq!(AgentRole::Coordinator.default_tau(), 0.05);
    assert_eq!(AgentRole::Executor.default_tau(), 0.40);
    assert_eq!(AgentRole::Evaluator.default_tau(), 0.10);
    assert_eq!(AgentRole::Synthesizer.default_tau(), 0.80);
    assert_eq!(AgentRole::Evaluator.default_role_error_cost(), 0.9);
    let custom = AgentRole::Custom { name: "QA".into(), tau: 0.3, role_error_cost: 0.6 };
    assert_eq!(custom.default_tau(), 0.3);
    assert_eq!(custom.default_role_error_cost(), 0.6);
}

#[test]
fn agent_role_serde_round_trip() {
    let role = AgentRole::Custom { name: "QA".into(), tau: 0.3, role_error_cost: 0.6 };
    let json = serde_json::to_string(&role).unwrap();
    let back: AgentRole = serde_json::from_str(&json).unwrap();
    assert_eq!(role, back);
}

#[test]
fn role_spec_serde_round_trip() {
    let spec = RoleSpec {
        agent_id: "impl_1".into(),
        role: AgentRole::Executor,
        tau: None,
        role_error_cost: Some(0.7),
    };
    let json = serde_json::to_string(&spec).unwrap();
    let back: RoleSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(spec.agent_id, back.agent_id);
    assert_eq!(back.role, AgentRole::Executor);
}

#[test]
fn review_gate_serde_round_trip() {
    let gate = ReviewGate { reviewer: "eval".into(), blocks: "impl_1".into() };
    let json = serde_json::to_string(&gate).unwrap();
    let back: ReviewGate = serde_json::from_str(&json).unwrap();
    assert_eq!(gate.reviewer, back.reviewer);
    assert_eq!(gate.blocks, back.blocks);
}

#[test]
fn explorer_config_serde_round_trip() {
    use h2ai_types::identity::ExplorerId;
    let cfg = ExplorerConfig {
        explorer_id: ExplorerId::new(),
        tau: 0.7,
        adapter: AdapterKind::CloudGeneric {
            endpoint: "https://api.example.com".into(),
            api_key_env: "CLOUD_API_KEY".into(),
        },
        role: None,
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: ExplorerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.tau, back.tau);
    assert!(back.role.is_none());
}

#[test]
fn explorer_config_with_role_serde_round_trip() {
    use h2ai_types::identity::ExplorerId;
    let cfg = ExplorerConfig {
        explorer_id: ExplorerId::new(),
        tau: 0.1,
        adapter: AdapterKind::CloudGeneric {
            endpoint: "https://api.example.com".into(),
            api_key_env: "CLOUD_API_KEY".into(),
        },
        role: Some(AgentRole::Evaluator),
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: ExplorerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.role, Some(AgentRole::Evaluator));
}

#[test]
fn auditor_config_tau_is_always_zero() {
    let cfg = AuditorConfig {
        adapter: AdapterKind::CloudGeneric {
            endpoint: "https://api.example.com".into(),
            api_key_env: "CLOUD_API_KEY".into(),
        },
    };
    assert_eq!(cfg.tau(), 0.0);
}

#[test]
fn adapter_kind_local_llama_cpp_serde_round_trip() {
    let a = AdapterKind::LocalLlamaCpp {
        model_path: PathBuf::from("/models/llama-70b.gguf"),
        n_threads: 16,
    };
    let json = serde_json::to_string(&a).unwrap();
    let back: AdapterKind = serde_json::from_str(&json).unwrap();
    assert_eq!(a, back);
}
