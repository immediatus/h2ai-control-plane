use h2ai_types::config::{
    AdapterKind, AgentRole, AuditorConfig, ExplorerConfig, ParetoWeights, ReviewGate, RoleSpec,
    TopologyKind,
};
use h2ai_types::sizing::TauValue;
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
    let t = TopologyKind::HierarchicalTree {
        branching_factor: Some(3),
    };
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
#[allow(clippy::float_cmp)]
fn agent_role_default_tau_and_ci() {
    assert_eq!(AgentRole::Coordinator.default_tau(), 0.05);
    assert_eq!(AgentRole::Executor.default_tau(), 0.40);
    assert_eq!(AgentRole::Evaluator.default_tau(), 0.10);
    assert_eq!(AgentRole::Synthesizer.default_tau(), 0.80);
    assert_eq!(AgentRole::Evaluator.default_role_error_cost(), 0.9);
    let custom = AgentRole::Custom {
        name: "QA".into(),
        tau: TauValue::new(0.3).unwrap(),
        role_error_cost: 0.6,
    };
    assert_eq!(custom.default_tau(), 0.3);
    assert_eq!(custom.default_role_error_cost(), 0.6);
}

#[test]
fn agent_role_serde_round_trip() {
    let role = AgentRole::Custom {
        name: "QA".into(),
        tau: TauValue::new(0.3).unwrap(),
        role_error_cost: 0.6,
    };
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
    let gate = ReviewGate {
        reviewer: "eval".into(),
        blocks: "impl_1".into(),
    };
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
        tau: TauValue::new(0.7).unwrap(),
        adapter: AdapterKind::CloudGeneric {
            endpoint: "https://api.example.com".into(),
            api_key_env: "CLOUD_API_KEY".into(),
            model: None,
            provider: Default::default(),
        },
        role: None,
        is_reasoning_model: false,
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
        tau: TauValue::new(0.1).unwrap(),
        adapter: AdapterKind::CloudGeneric {
            endpoint: "https://api.example.com".into(),
            api_key_env: "CLOUD_API_KEY".into(),
            model: None,
            provider: Default::default(),
        },
        role: Some(AgentRole::Evaluator),
        is_reasoning_model: false,
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: ExplorerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.role, Some(AgentRole::Evaluator));
}

#[test]
fn auditor_config_has_default_tau() {
    let cfg = AuditorConfig {
        adapter: AdapterKind::CloudGeneric {
            endpoint: "https://api.example.com".into(),
            api_key_env: "CLOUD_API_KEY".into(),
            model: None,
            provider: Default::default(),
        },
        ..Default::default()
    };
    assert_eq!(cfg.tau, TauValue::new(0.1).unwrap());
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

#[test]
fn adapter_kind_openai_serde_round_trip() {
    let k = AdapterKind::OpenAI {
        api_key_env: "OPENAI_API_KEY".into(),
        model: "gpt-4o".into(),
    };
    let json = serde_json::to_string(&k).unwrap();
    let back: AdapterKind = serde_json::from_str(&json).unwrap();
    assert_eq!(k, back);
}

#[test]
fn adapter_kind_anthropic_serde_round_trip() {
    let k = AdapterKind::Anthropic {
        api_key_env: "ANTHROPIC_API_KEY".into(),
        model: "claude-3-5-sonnet-20241022".into(),
    };
    let json = serde_json::to_string(&k).unwrap();
    let back: AdapterKind = serde_json::from_str(&json).unwrap();
    assert_eq!(k, back);
}

#[test]
fn adapter_kind_ollama_serde_round_trip() {
    let k = AdapterKind::Ollama {
        endpoint: "http://localhost:11434".into(),
        model: "llama3.2".into(),
    };
    let json = serde_json::to_string(&k).unwrap();
    let back: AdapterKind = serde_json::from_str(&json).unwrap();
    assert_eq!(k, back);
}

// ── AgentRole::default_role_error_cost for all variants ──────────────────────

#[test]
fn agent_role_default_role_error_cost_all_variants() {
    assert!((AgentRole::Coordinator.default_role_error_cost() - 0.1).abs() < 1e-10);
    assert!((AgentRole::Executor.default_role_error_cost() - 0.5).abs() < 1e-10);
    assert!((AgentRole::Synthesizer.default_role_error_cost() - 0.1).abs() < 1e-10);
}

// ── AdapterFamily::Display ────────────────────────────────────────────────────

#[test]
fn adapter_family_display_all_variants() {
    use h2ai_types::config::AdapterFamily;
    assert_eq!(AdapterFamily::Anthropic.to_string(), "anthropic");
    assert_eq!(AdapterFamily::OpenAI.to_string(), "openai");
    assert_eq!(AdapterFamily::Local.to_string(), "local");
    assert_eq!(AdapterFamily::Cloud.to_string(), "cloud");
}

// ── TaoConfig::default ────────────────────────────────────────────────────────

#[test]
fn tao_config_default_values() {
    use h2ai_types::config::TaoConfig;
    let cfg = TaoConfig::default();
    assert_eq!(cfg.max_turns, 3);
    assert!(cfg.verify_pattern.is_none());
    assert!((cfg.repetition_threshold - 0.92).abs() < 1e-10);
    assert_eq!(cfg.per_turn_timeout_secs, 600);
    assert!(!cfg.observation_pass.is_empty());
    assert!(!cfg.observation_fail_pattern.is_empty());
    assert!(!cfg.observation_fail_schema.is_empty());
    assert!(!cfg.retry_instruction.is_empty());
}

// ── VerificationConfig::default ───────────────────────────────────────────────

#[test]
fn verification_config_default_values() {
    use h2ai_types::config::VerificationConfig;
    let cfg = VerificationConfig::default();
    assert!((cfg.threshold - 0.45).abs() < 1e-10);
    assert_eq!(cfg.evaluator_max_tokens, 32768);
    assert!(!cfg.record_adversarial_comparison);
    assert!(!cfg.rubric.is_empty());
    assert!(!cfg.evaluator_system_prompt.is_empty());
}

// ── ConfigError display messages ──────────────────────────────────────────────

#[test]
fn config_error_display_messages() {
    use h2ai_types::config::ConfigError;
    let msg = ConfigError::InvalidWeightSum(1.5).to_string();
    assert!(msg.contains("1.5"), "message must contain bad sum: {msg}");
    let neg = ConfigError::NegativeWeight.to_string();
    assert!(
        !neg.is_empty(),
        "NegativeWeight error message must not be empty: {neg}"
    );
}

// ── AdapterKind::A2a serde round-trip ─────────────────────────────────────────

#[test]
fn adapter_kind_a2a_serde_round_trip() {
    let k = AdapterKind::A2a {
        endpoint: "https://agent.example.com".into(),
        auth_scheme: "bearer".into(),
        auth_token_env: "A2A_TOKEN".into(),
        timeout_minutes: 5,
        poll_interval_ms: 500,
        max_poll_interval_ms: 5000,
        agent_card_cache_ttl_s: 300,
    };
    let json = serde_json::to_string(&k).unwrap();
    let back: AdapterKind = serde_json::from_str(&json).unwrap();
    assert_eq!(k, back);
}

// ── AdapterFamily::from_kind covers A2a → Cloud ───────────────────────────────

#[test]
fn adapter_family_from_kind_a2a_is_cloud() {
    use h2ai_types::config::AdapterFamily;
    let kind = AdapterKind::A2a {
        endpoint: "http://svc".into(),
        auth_scheme: "none".into(),
        auth_token_env: String::new(),
        timeout_minutes: 1,
        poll_interval_ms: 100,
        max_poll_interval_ms: 1000,
        agent_card_cache_ttl_s: 60,
    };
    assert_eq!(AdapterFamily::from_kind(&kind), AdapterFamily::Cloud);
}

// ── AdapterKind::model_lineage_key ────────────────────────────────────────────

fn cloud(endpoint: &str, model: &str) -> AdapterKind {
    use h2ai_types::config::CloudProvider;
    AdapterKind::CloudGeneric {
        endpoint: endpoint.to_string(),
        api_key_env: "KEY".to_string(),
        model: Some(model.to_string()),
        provider: CloudProvider::Generic,
    }
}

#[test]
fn same_cloud_model_same_key() {
    let a = cloud("https://api.openai.com", "gpt-4o");
    let b = cloud("https://api.openai.com", "gpt-4o");
    assert_eq!(a.model_lineage_key(), b.model_lineage_key());
}

#[test]
fn different_cloud_models_different_keys() {
    let a = cloud("https://api.openai.com", "gpt-4o");
    let b = cloud("https://api.openai.com", "gpt-4o-mini");
    assert_ne!(a.model_lineage_key(), b.model_lineage_key());
}

#[test]
fn different_endpoints_different_keys() {
    let a = cloud("https://api.openai.com", "gpt-4o");
    let b = cloud("https://api.anthropic.com", "gpt-4o");
    assert_ne!(a.model_lineage_key(), b.model_lineage_key());
}

#[test]
fn local_models_with_different_paths_different_keys() {
    let a = AdapterKind::LocalLlamaCpp {
        model_path: PathBuf::from("/models/llama-70b"),
        n_threads: 8,
    };
    let b = AdapterKind::LocalLlamaCpp {
        model_path: PathBuf::from("/models/qwen-72b"),
        n_threads: 8,
    };
    assert_ne!(a.model_lineage_key(), b.model_lineage_key());
}

#[test]
fn same_local_model_same_key() {
    let a = AdapterKind::LocalLlamaCpp {
        model_path: PathBuf::from("/models/llama-70b"),
        n_threads: 4,
    };
    let b = AdapterKind::LocalLlamaCpp {
        model_path: PathBuf::from("/models/llama-70b"),
        n_threads: 8,
    };
    // n_threads is NOT part of the lineage key — same model regardless of thread count
    assert_eq!(a.model_lineage_key(), b.model_lineage_key());
}

#[test]
fn monoculture_pool_has_one_distinct_lineage_key() {
    let adapters = vec![
        cloud("http://host.docker.internal:8080/v1", "local"),
        cloud("http://host.docker.internal:8080/v1", "local"),
        cloud("http://host.docker.internal:8080/v1", "local"),
    ];
    let keys: std::collections::HashSet<String> =
        adapters.iter().map(|a| a.model_lineage_key()).collect();
    assert_eq!(
        keys.len(),
        1,
        "monoculture pool must have exactly 1 lineage key"
    );
}

#[test]
fn diverse_pool_has_multiple_lineage_keys() {
    let adapters = vec![
        cloud("https://api.openai.com/v1", "gpt-4o"),
        cloud("https://api.anthropic.com/v1", "claude-sonnet-4-6"),
    ];
    let keys: std::collections::HashSet<String> =
        adapters.iter().map(|a| a.model_lineage_key()).collect();
    assert_eq!(
        keys.len(),
        2,
        "diverse pool must have 2 distinct lineage keys"
    );
}
