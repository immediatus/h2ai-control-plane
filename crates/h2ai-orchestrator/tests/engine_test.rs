use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::{
    ConstraintDoc, ConstraintPredicate, ConstraintSeverity, VocabularyMode,
};

use h2ai_orchestrator::engine::{EngineError, EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_types::adapter::{
    AdapterError, AdapterRegistry, ComputeRequest, ComputeResponse, IComputeAdapter,
};
use h2ai_types::config::{
    AdapterKind, AgentRole, AuditorConfig, ParetoWeights, RoleSpec, TaoConfig, VerificationConfig,
};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use std::sync::Arc;

// An adapter that returns different outputs on successive calls (for synthesis tests)
#[derive(Debug)]
struct SequencedAdapter {
    responses: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    kind: AdapterKind,
}

impl SequencedAdapter {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: std::sync::Arc::new(std::sync::Mutex::new(responses)),
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://sequenced".into(),
                api_key_env: "NONE".into(),
            },
        }
    }
}

#[async_trait::async_trait]
impl IComputeAdapter for SequencedAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let mut responses = self.responses.lock().unwrap();
        let output = if responses.is_empty() {
            "fallback".to_string()
        } else {
            responses.remove(0)
        };
        Ok(ComputeResponse {
            output,
            token_cost: 100,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

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
        constraint_corpus: &[],
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
    let corpus = vec![ConstraintDoc::new_llm_judge(
        "ADR-001",
        "The solution must be stateless. No server-side sessions or shared mutable state permitted.",
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
            slot_configs: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
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
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
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
    assert!(outcome.attribution.q_confidence >= outcome.attribution.baseline_quality);
    assert!(outcome.attribution.q_confidence <= 1.0);
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
    let corpus = vec![ConstraintDoc::new_llm_judge(
        "ADR-001",
        "The solution must be stateless. No server-side sessions or shared mutable state permitted.",
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
            slot_configs: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
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
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
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
    let corpus = vec![ConstraintDoc::new_llm_judge(
        "ADR-001",
        "The solution must be stateless. No server-side sessions or shared mutable state permitted.",
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
            slot_configs: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
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
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
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
    let corpus = vec![ConstraintDoc::new_llm_judge(
        "ADR-001",
        "The solution must be stateless. No server-side sessions or shared mutable state permitted.",
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
            slot_configs: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
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
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
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
    let corpus = vec![ConstraintDoc::new_llm_judge(
        "ADR-001",
        "The solution must be stateless. No server-side sessions or shared mutable state permitted.",
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
            slot_configs: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
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
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
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
    let corpus = vec![ConstraintDoc::new_llm_judge(
        "ADR-001",
        "The solution must be stateless. No server-side sessions or shared mutable state permitted.",
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
            slot_configs: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
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
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
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
    let corpus = vec![ConstraintDoc::new_llm_judge(
        "ADR-001",
        "The solution must be stateless. No server-side sessions or shared mutable state permitted.",
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
            slot_configs: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
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
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
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

#[tokio::test]
async fn engine_synthesis_phase_bypasses_merge_and_returns_synthesis_text() {
    // Two explorer adapters with different outputs - both must contain "stateless" and "auth"
    // so they pass the VocabularyPresence constraint from the corpus
    let explorer1 =
        MockAdapter::new("stateless auth JWT implementation ADR-001 approach one".into());
    let explorer2 =
        MockAdapter::new("stateless auth RSA signing credential ADR-001 approach two".into());
    // Verifier returns compliant for all proposals (including synthesis re-verification)
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    // Enable synthesis with min_proposals=2 so 2 auditor-passed proposals trigger it
    let cfg = H2AIConfig {
        synthesis_enabled: true,
        synthesis_min_proposals: 2,
        ..H2AIConfig::default()
    };
    // ADR-001: LlmJudge (Light tier) — used for auditing/verification, not satisfaction matrix.
    //
    // Two Static-tier Soft constraints are added to create a genuine contradiction so coherence
    // stays open and synthesis is NOT bypassed by the is_closed() guard:
    //
    // SIG-BASE: VocabularyPresence AllOf ["stateless", "adr-001"]
    //   Both explorer outputs contain these terms → score 1.0 for both.
    //   Ensures both proposals PASS verification (combined soft aggregate ≥ 0.45).
    //
    // SIG-JWT: VocabularyPresence AllOf ["jwt"]
    //   explorer1 contains "jwt" → score 1.0  (satisfies constraint, ≥ 0.5 threshold)
    //   explorer2 does NOT contain "jwt" → score 0.0  (fails matrix threshold)
    //   This divergence creates a contradiction in the "token-format" domain.
    //
    // Combined aggregate for explorer2: (1.0*0.5 + 0.0*0.5) / 1.0 = 0.5 ≥ 0.45 → passes.
    let corpus = vec![
        ConstraintDoc::new_llm_judge(
            "ADR-001",
            "The solution must be stateless. No server-side sessions or shared mutable state permitted.",
        ),
        ConstraintDoc {
            id: "SIG-BASE".to_string(),
            source_file: "SIG-BASE.yaml".into(),
            description: "Solution must be stateless and reference ADR-001".into(),
            severity: ConstraintSeverity::Soft { weight: 0.5 },
            predicate: ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AllOf,
                terms: vec!["stateless".to_string(), "adr-001".to_string()],
            },
            remediation_hint: None,
            domains: vec!["architecture".to_string()],
            mandatory_for_tags: vec![],
            related_to: vec![],
        },
        ConstraintDoc {
            id: "SIG-JWT".to_string(),
            source_file: "SIG-JWT.yaml".into(),
            description: "JWT token format must be used".into(),
            severity: ConstraintSeverity::Soft { weight: 0.5 },
            predicate: ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AllOf,
                terms: vec!["jwt".to_string()],
            },
            remediation_hint: None,
            domains: vec!["token-format".to_string()],
            mandatory_for_tags: vec![],
            related_to: vec![],
        },
    ];
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
            slot_configs: vec![],
        },
        constraints: vec!["ADR-001".into(), "SIG-BASE".into(), "SIG-JWT".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
    };

    let valid_critique = r#"{"proposal_critiques":[{"proposal_id":"p1","strengths":["s1"],"weaknesses":[],"verdict":"partial"},{"proposal_id":"p2","strengths":["s2"],"weaknesses":[],"verdict":"strong"}],"contradictions":[],"synthesis_guidance":"Use p2."}"#;
    // Must contain "stateless" and "auth" to pass the VocabularyPresence constraint from the corpus
    let synthesis_text = "Synthesised stateless auth solution ADR-001 compliant unified output.";

    // SequencedAdapter: first call returns critique JSON, second call returns synthesis text
    let synth_adapter =
        SequencedAdapter::new(vec![valid_critique.to_string(), synthesis_text.to_string()]);

    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![
            &explorer1 as &dyn IComputeAdapter,
            &explorer2 as &dyn IComputeAdapter,
        ],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
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
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: Some(&synth_adapter as &dyn IComputeAdapter),
        bandit_state: None,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine returned error: {:?}", result.err());
    assert_eq!(
        result.unwrap().resolved_output,
        synthesis_text,
        "resolved_output must equal the synthesis text when synthesis succeeds"
    );
}

#[test]
fn constrained_exploration_tombstone_synthesis_unit() {
    use h2ai_autonomic::epistemic::synthesize_tombstone;
    use h2ai_types::events::ConstraintViolation;

    let violations = vec![ConstraintViolation {
        constraint_id: "ADR-001".into(),
        score: 0.0,
        severity_label: "Hard".into(),
        remediation_hint: None,
    }];
    let tombstone = synthesize_tombstone(&violations);
    assert!(
        tombstone.is_some(),
        "non-empty violations must produce tombstone"
    );
    let s = tombstone.unwrap();
    assert!(
        s.contains("ADR-001"),
        "tombstone must contain constraint ID"
    );
}

#[tokio::test]
async fn pool_diversity_guard_fires_when_n_eff_below_threshold() {
    let adapter = mock_adapter();
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let store = TaskStore::new();
    let cfg = H2AIConfig {
        diversity_threshold: 0.5,
        max_autonomic_retries: 0,
        ..H2AIConfig::default()
    };

    let mut cal = calibration().await;
    cal.n_eff_cosine_prior = 1.0; // collapsed pool: below 1 + 0.5 = 1.5 threshold

    let manifest = TaskManifest {
        description: "Test pool guard".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 2,
            tau_min: Some(0.3),
            tau_max: Some(0.7),
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
    };
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&adapter as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(
        result.is_err(),
        "collapsed pool must cause engine failure when retries=0"
    );
    assert!(
        matches!(result.unwrap_err(), EngineError::MaxRetriesExhausted),
        "expected MaxRetriesExhausted from pool diversity guard"
    );
}

#[test]
fn epistemic_yield_ratio_formula_correctness() {
    // N_requested=3, N_responded=2 (one timed out), n_eff_actual=1.5
    // Correct: 1.5/3=0.5, NOT 1.5/2=0.75
    let n_requested: usize = 3;
    let n_eff_actual: f64 = 1.5;
    let yield_ratio = n_eff_actual / n_requested as f64;
    assert!((yield_ratio - 0.5).abs() < 1e-9);
    assert!(yield_ratio < n_eff_actual / 2.0); // conservative vs N_responded
}

#[test]
fn epistemic_yield_event_yield_ratio_uses_n_requested() {
    use h2ai_types::events::EpistemicYieldEvent;
    use h2ai_types::identity::TaskId;

    let n_requested: usize = 3;
    let n_eff_cosine_actual: f64 = 1.5;
    let yield_ratio = n_eff_cosine_actual / n_requested as f64;

    let ev = EpistemicYieldEvent {
        task_id: TaskId::new(),
        n_eff_cosine_actual,
        n_eff_prior: 2.0,
        yield_ratio,
        adapters: vec!["a".into(), "b".into(), "c".into()],
    };

    assert!(
        (ev.yield_ratio - 0.5).abs() < 1e-9,
        "yield_ratio must be n_eff / N_requested=3, got {}",
        ev.yield_ratio
    );
}

// ── Verifier/Explorer Family Conflict Gate ──────────────────────────────────

/// A mock adapter that reports a specific AdapterKind (and thus family).
#[derive(Debug)]
struct FamilyAdapter {
    output: String,
    kind: AdapterKind,
}

impl FamilyAdapter {
    fn anthropic(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            kind: AdapterKind::Anthropic {
                api_key_env: "ANTHROPIC_KEY".into(),
                model: "claude-3-5-sonnet-20241022".into(),
            },
        }
    }
    fn openai(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            kind: AdapterKind::OpenAI {
                api_key_env: "OPENAI_KEY".into(),
                model: "gpt-4o".into(),
            },
        }
    }
}

#[async_trait::async_trait]
impl IComputeAdapter for FamilyAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Ok(ComputeResponse {
            output: self.output.clone(),
            token_cost: 10,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

/// When the verifier shares a family with the explorer pool and allow_single_family=false,
/// the engine must reject the task immediately with VerifierExplorerFamilyConflict.
/// No LLM tokens are burned — the gate fires before topology provisioning.
#[tokio::test]
async fn engine_rejects_verifier_explorer_family_conflict() {
    // Obtain a valid calibration via the harness, then set the conflict flag.
    let mut cal = calibration().await;
    cal.explorer_verification_family_match = true;

    let explorer = FamilyAdapter::anthropic("some proposal text");
    let verifier = FamilyAdapter::anthropic(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = FamilyAdapter::anthropic(r#"{"approved": true, "reason": "ok"}"#);

    let cfg = H2AIConfig {
        allow_single_family: false,
        ..Default::default()
    };

    let store = TaskStore::new();
    let registry =
        AdapterRegistry::new(Arc::new(FamilyAdapter::anthropic("")) as Arc<dyn IComputeAdapter>);

    let manifest = TaskManifest {
        description: "any task".into(),
        pareto_weights: h2ai_types::config::ParetoWeights::new(0.33, 0.34, 0.33).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 2,
            tau_min: Some(0.3),
            tau_max: Some(0.7),
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
    };

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn IComputeAdapter],
        verification_adapter: &verifier as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::Anthropic {
                api_key_env: "ANTHROPIC_KEY".into(),
                model: "claude-3-5-sonnet-20241022".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
    };

    let err = ExecutionEngine::run_offline(input).await.unwrap_err();
    match err {
        EngineError::MultiplicationConditionFailed(msg) => {
            assert!(
                msg.contains("VerifierExplorerFamilyConflict") || msg.contains("monoculture"),
                "error message should name the conflict, got: {msg}"
            );
        }
        other => panic!("expected MultiplicationConditionFailed, got: {other:?}"),
    }
}

/// When allow_single_family=true the conflict gate is bypassed and execution proceeds.
#[tokio::test]
async fn engine_bypasses_family_conflict_gate_when_allow_single_family() {
    let mut cal = calibration().await;
    cal.explorer_verification_family_match = true;

    // Explorer and verifier both Anthropic, but allow_single_family bypasses the gate.
    let explorer = FamilyAdapter::anthropic("stateless JWT authentication token refresh ADR-001");
    let verifier = FamilyAdapter::openai(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = FamilyAdapter::openai(r#"{"approved": true, "reason": "ok"}"#);

    let cfg = H2AIConfig {
        allow_single_family: true,
        ..Default::default()
    };

    let store = TaskStore::new();
    let registry =
        AdapterRegistry::new(Arc::new(FamilyAdapter::anthropic("")) as Arc<dyn IComputeAdapter>);

    let manifest = TaskManifest {
        description: "stateless JWT authentication token refresh ADR-001".into(),
        pareto_weights: h2ai_types::config::ParetoWeights::new(0.33, 0.34, 0.33).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 2,
            tau_min: Some(0.3),
            tau_max: Some(0.7),
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
    };

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![
            &explorer as &dyn IComputeAdapter,
            &explorer as &dyn IComputeAdapter,
        ],
        verification_adapter: &verifier as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::OpenAI {
                api_key_env: "OPENAI_KEY".into(),
                model: "gpt-4o".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
    };

    // Should not return VerifierExplorerFamilyConflict — may succeed or fail for other reasons.
    let result = ExecutionEngine::run_offline(input).await;
    if let Err(EngineError::MultiplicationConditionFailed(msg)) = &result {
        assert!(
            !msg.contains("monoculture"),
            "gate fired despite allow_single_family=true: {msg}"
        );
    }
}

/// Records system_context of every execute() call for later assertion.
#[derive(Debug, Clone)]
struct CapturingAdapter {
    response_sequence: Arc<std::sync::Mutex<Vec<String>>>,
    captured_contexts: Arc<std::sync::Mutex<Vec<String>>>,
    kind: AdapterKind,
}

impl CapturingAdapter {
    fn new(responses: Vec<String>) -> Self {
        Self {
            response_sequence: Arc::new(std::sync::Mutex::new(responses)),
            captured_contexts: Arc::new(std::sync::Mutex::new(Vec::new())),
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://capturing".into(),
                api_key_env: "NONE".into(),
            },
        }
    }

    fn captured_contexts(&self) -> Vec<String> {
        self.captured_contexts.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl IComputeAdapter for CapturingAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        self.captured_contexts
            .lock()
            .unwrap()
            .push(req.system_context.clone());
        let output = {
            let mut seq = self.response_sequence.lock().unwrap();
            if seq.is_empty() {
                "fallback proposal".to_string()
            } else {
                seq.remove(0)
            }
        };
        Ok(ComputeResponse {
            output,
            token_cost: 10,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

#[tokio::test]
async fn engine_hint_injected_into_explorer_on_retry() {
    // Constraint has a remediation_hint that should appear in the explorer's
    // system_context during the second generation iteration.
    let hint_text = "Use TTL-based caches with no node affinity required.";

    let corpus = vec![h2ai_constraints::types::ConstraintDoc {
        id: "C-HINT".into(),
        source_file: "C-HINT.yaml".into(),
        description: "Stateless caching rule".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.45 },
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "The proposal must use TTL-based caches with no node affinity.".into(),
        },
        remediation_hint: Some(hint_text.into()),
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    }];

    // Explorer: two proposals (one per iteration), content doesn't matter
    let explorer = CapturingAdapter::new(vec![
        "proposal iteration 0".into(),
        "proposal iteration 1".into(),
    ]);

    // Verifier: returns 0.0 on first call (iter 0 fails), 0.9 on second (iter 1 passes)
    let verifier = SequencedAdapter::new(vec![
        r#"{"score": 0.0, "reason": "missing TTL cache"}"#.into(),
        r#"{"score": 0.9, "reason": "compliant"}"#.into(),
    ]);

    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    // shadow_mode = true keeps Phase 1.5 purely observational so the Precision quadrant
    // does not force 2-3 explorers per iteration. With shadow_mode = false and a single
    // LlmJudge constraint the engine routes Precision and overrides count to clamp(2,3),
    // which consumes all verifier responses in one iteration and prevents the retry path.
    #[allow(clippy::field_reassign_with_default)]
    let cfg = {
        let mut c = H2AIConfig::default();
        c.max_autonomic_retries = 3;
        c.allow_single_family = true;
        c.task_complexity.shadow_mode = true;
        c
    };

    let manifest = TaskManifest {
        description: "Propose a stateless caching strategy".into(),
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
            slot_configs: vec![],
        },
        constraints: vec!["C-HINT".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
    };

    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("ignored".into())) as Arc<dyn IComputeAdapter>
    );
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn h2ai_types::adapter::IComputeAdapter],
        verification_adapter: &verifier as &dyn h2ai_types::adapter::IComputeAdapter,
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
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(
        result.is_ok(),
        "engine must resolve on second iteration; err: {:?}",
        result.err()
    );

    let contexts = explorer.captured_contexts();
    assert!(
        contexts.len() >= 2,
        "expected at least 2 explorer calls (one per iteration), got {}",
        contexts.len()
    );

    // Static remediation hint from the constraint doc is injected by the compiler
    // into every iteration's system context (not only on retry).
    let first_iter_ctx = &contexts[0];
    assert!(
        first_iter_ctx.contains(hint_text),
        "first iteration must contain static remediation hint from compiler;\n\
        hint: {hint_text}\n\
        context snippet: {}",
        &first_iter_ctx[..first_iter_ctx.len().min(500)]
    );

    // The SECOND iteration must additionally contain the dynamic CONSTRAINT FEEDBACK
    // block produced by RetryWithHints — this is the per-failure remediation injected
    // by the engine after the verifier rejects the first proposal.
    let second_iter_ctx = &contexts[1];
    assert!(
        second_iter_ctx.contains("CONSTRAINT FEEDBACK"),
        "second iteration explorer context must contain dynamic constraint feedback block;\n\
        context snippet: {}",
        &second_iter_ctx[..second_iter_ctx.len().min(500)]
    );
}
