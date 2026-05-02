use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_constraints::loader::parse_constraint_doc;
use h2ai_orchestrator::engine::{EngineError, EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{
    AdapterKind, AgentRole, AuditorConfig, ParetoWeights, RoleSpec, TaoConfig, VerificationConfig,
};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use std::sync::Arc;

fn mock_adapter() -> MockAdapter {
    MockAdapter::new("stateless JWT authentication token refresh ADR-001".into())
}

fn mock_adapter2() -> MockAdapter {
    MockAdapter::new("session-less credential verification via RSA signing ADR-001".into())
}

fn verifier() -> MockAdapter {
    MockAdapter::new(r#"{"score": 0.9, "reason": "compliant"}"#.into())
}

async fn calibration() -> h2ai_types::events::CalibrationCompletedEvent {
    let adapter = mock_adapter();
    let cfg = H2AIConfig::default();
    CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Calibrate".into(), "Second task".into(), "Third".into()],
        adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        cfg: &cfg,
        embedding_model: None,
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn engine_runs_ensemble_to_semilattice() {
    let adapter = mock_adapter();
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let corpus = vec![parse_constraint_doc(
        "ADR-001",
        "## Constraints\nstateless auth\n",
    )];

    let manifest = TaskManifest {
        description: "Propose stateless auth with ADR-001 compliance".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 2,
            tau_min: Some(0.3),
            tau_max: Some(0.8),
            roles: vec![],
            review_gates: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    };

    let adapter2 = mock_adapter2();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![
            &adapter as &dyn h2ai_types::adapter::IComputeAdapter,
            &adapter2,
        ],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: &auditor as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine returned error: {:?}", result.err());

    let outcome = result.unwrap();
    let state = store.get(&outcome.task_id).unwrap();
    assert!(
        state.status == "merging" || state.status == "resolved",
        "unexpected status: {}",
        state.status
    );
    assert!(outcome.attribution.baseline_quality > 0.0);
    assert!(outcome.attribution.total_quality >= outcome.attribution.baseline_quality);
    assert!(outcome.attribution.total_quality <= 1.0);
}

#[tokio::test]
async fn engine_rejects_insufficient_context() {
    let adapter = mock_adapter();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();

    let manifest = TaskManifest {
        description: "do stuff".into(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
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
        },
        constraints: vec![],
        context: None,
    };

    // Corpus keywords that won't overlap with "do stuff" — forces J_eff below gate
    let corpus = vec![parse_constraint_doc(
        "ADR-001",
        "## Constraints\nmicroservice stateless distributed consensus byzantine\n",
    )];

    let scorer = verifier();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: &auditor as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("J_eff") || err_str.contains("context underflow"),
        "{err_str}"
    );
}

#[tokio::test]
async fn engine_structured_auditor_approved_passes_proposal() {
    // Auditor returns {"approved": true, "reason": "compliant"} → task resolves
    let explorer = mock_adapter();
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let corpus = vec![parse_constraint_doc(
        "ADR-001",
        "## Constraints\nstateless auth\n",
    )];
    let manifest = TaskManifest {
        description: "Propose stateless auth with ADR-001 compliance".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: Some(0.5),
            tau_max: Some(0.5),
            roles: vec![],
            review_gates: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    };
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn h2ai_types::adapter::IComputeAdapter],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: &auditor as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
    };
    let result = ExecutionEngine::run_offline(input).await;
    assert!(
        result.is_ok(),
        "approved auditor should resolve task: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn engine_structured_auditor_rejected_prunes_proposal() {
    // Auditor returns {"approved": false, ...} → ZeroSurvival → MaxRetriesExhausted
    let explorer = mock_adapter();
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": false, "reason": "violates ADR-42"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig {
        max_autonomic_retries: 0,
        ..H2AIConfig::default()
    };
    let corpus = vec![parse_constraint_doc(
        "ADR-001",
        "## Constraints\nstateless auth\n",
    )];
    let manifest = TaskManifest {
        description: "Propose stateless auth with ADR-001 compliance".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: Some(0.5),
            tau_max: Some(0.5),
            roles: vec![],
            review_gates: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    };
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn h2ai_types::adapter::IComputeAdapter],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: &auditor as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
    };
    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_err(), "rejected auditor should fail task");
    assert!(
        matches!(result.unwrap_err(), EngineError::MaxRetriesExhausted),
        "expected MaxRetriesExhausted"
    );
}

#[tokio::test]
async fn engine_structured_auditor_non_json_fails_safe() {
    // Auditor returns plain text → fail safe = reject → ZeroSurvival → MaxRetriesExhausted
    let explorer = mock_adapter();
    let scorer = verifier();
    let auditor = MockAdapter::new("I think this looks fine overall".into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig {
        max_autonomic_retries: 0,
        ..H2AIConfig::default()
    };
    let corpus = vec![parse_constraint_doc(
        "ADR-001",
        "## Constraints\nstateless auth\n",
    )];
    let manifest = TaskManifest {
        description: "Propose stateless auth with ADR-001 compliance".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: Some(0.5),
            tau_max: Some(0.5),
            roles: vec![],
            review_gates: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    };
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn h2ai_types::adapter::IComputeAdapter],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: &auditor as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
    };
    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_err(), "non-JSON auditor should fail safe");
    assert!(
        matches!(result.unwrap_err(), EngineError::MaxRetriesExhausted),
        "expected MaxRetriesExhausted"
    );
}

#[tokio::test]
async fn engine_output_contains_talagrand_diagnostic() {
    let adapter = mock_adapter();
    let adapter2 = mock_adapter2();
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let corpus = vec![parse_constraint_doc(
        "ADR-001",
        "## Constraints\nstateless auth\n",
    )];
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);

    let manifest = TaskManifest {
        description: "Propose stateless auth with ADR-001 compliance".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 2,
            tau_min: Some(0.5),
            tau_max: Some(0.8),
            roles: vec![],
            review_gates: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    };

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![
            &adapter as &dyn IComputeAdapter,
            &adapter2 as &dyn IComputeAdapter,
        ],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        output.talagrand.is_some(),
        "talagrand must be populated when verification events exist"
    );
}

#[tokio::test]
async fn engine_rejects_krum_when_quorum_not_satisfied() {
    // krum_fault_tolerance=1 requires n ≥ 5. Requesting only 3 explorers must fail.
    let adapter = mock_adapter();
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    // krum_fault_tolerance=1 with krum_threshold=0.5 (lower than default so it triggers)
    let cfg = H2AIConfig {
        krum_fault_tolerance: 1,
        krum_threshold: 0.5,
        ..H2AIConfig::default()
    };
    let corpus = vec![parse_constraint_doc(
        "ADR-001",
        "## Constraints\nstateless auth\n",
    )];
    let manifest = TaskManifest {
        description: "Propose stateless auth with ADR-001 compliance".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 3,
            tau_min: Some(0.5),
            tau_max: Some(0.5),
            roles: vec![
                RoleSpec {
                    agent_id: "exp1".into(),
                    role: AgentRole::Evaluator,
                    tau: None,
                    role_error_cost: Some(0.99),
                },
                RoleSpec {
                    agent_id: "exp2".into(),
                    role: AgentRole::Evaluator,
                    tau: None,
                    role_error_cost: Some(0.99),
                },
                RoleSpec {
                    agent_id: "exp3".into(),
                    role: AgentRole::Evaluator,
                    tau: None,
                    role_error_cost: Some(0.99),
                },
            ],
            review_gates: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    };
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![
            &adapter as &dyn h2ai_types::adapter::IComputeAdapter,
            &adapter,
            &adapter,
        ],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: &auditor as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), EngineError::InsufficientQuorum { .. }),
        "expected InsufficientQuorum when n=3 < 5 required for f=1"
    );
}

#[tokio::test]
async fn engine_output_contains_suggested_next_params() {
    let adapter = mock_adapter();
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let corpus = vec![parse_constraint_doc(
        "ADR-001",
        "## Constraints\nstateless auth\n",
    )];
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);

    let manifest = TaskManifest {
        description: "Propose stateless auth with ADR-001 compliance".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: Some(0.5),
            tau_max: Some(0.5),
            roles: vec![],
            review_gates: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    };

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&adapter as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        output.suggested_next_params.is_some(),
        "suggested_next_params must be populated on success"
    );
    let params = output.suggested_next_params.unwrap();
    assert!(
        params.n_agents >= 1,
        "suggested n_agents must be at least 1"
    );
    assert!(
        params.verify_threshold > 0.0,
        "verify_threshold must be positive"
    );
}
