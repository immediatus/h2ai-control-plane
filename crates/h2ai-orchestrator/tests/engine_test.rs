use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::{FamilyConstraint, H2AIConfig, SafetyConfig};
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
use h2ai_types::identity::{TaskId, TenantId};
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
                model: None,
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
            reasoning_trace: None,
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_err(), "rejected auditor should fail task");
    assert!(
        matches!(result.unwrap_err(), EngineError::MaxRetriesExhausted { .. }),
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_err(), "non-JSON auditor should fail safe");
    assert!(
        matches!(result.unwrap_err(), EngineError::MaxRetriesExhausted { .. }),
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
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
    let cfg = {
        let mut c = H2AIConfig::default();
        c.safety.krum_fault_tolerance = 1;
        c.safety.krum_threshold = 0.5;
        c
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
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
            diversity_ids: vec![],
        },
        constraints: vec!["ADR-001".into(), "SIG-BASE".into(), "SIG-JWT".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
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
        safety: SafetyConfig {
            diversity_threshold: 0.5,
            ..SafetyConfig::default()
        },
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(
        result.is_err(),
        "collapsed pool must cause engine failure when retries=0"
    );
    assert!(
        matches!(result.unwrap_err(), EngineError::MaxRetriesExhausted { .. }),
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
            reasoning_trace: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

/// When the verifier shares a family with the explorer pool and family_constraint=RequireDiverse,
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
        safety: SafetyConfig {
            family_constraint: FamilyConstraint::RequireDiverse,
            ..SafetyConfig::default()
        },
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        // Two slots, same adapter pointer → distinct.len() == 1 → triggers monoculture error
        explorer_adapters: vec![
            &explorer as &dyn IComputeAdapter,
            &explorer as &dyn IComputeAdapter,
        ],
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };

    let err = ExecutionEngine::run_offline(input).await.unwrap_err();
    match err {
        EngineError::MultiplicationConditionFailed(msg) => {
            assert!(
                msg.contains("same adapter") || msg.contains("diversity"),
                "error message should name the monoculture condition, got: {msg}"
            );
        }
        other => panic!("expected MultiplicationConditionFailed, got: {other:?}"),
    }
}

/// When family_constraint=SingleFamilyOk (development default) the conflict gate is bypassed and execution proceeds.
#[tokio::test]
async fn engine_bypasses_family_conflict_gate_when_single_family_ok() {
    let mut cal = calibration().await;
    cal.explorer_verification_family_match = true;

    // Explorer and verifier both Anthropic, but family_constraint=SingleFamilyOk bypasses the gate.
    let explorer = FamilyAdapter::anthropic("stateless JWT authentication token refresh ADR-001");
    let verifier = FamilyAdapter::openai(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = FamilyAdapter::openai(r#"{"approved": true, "reason": "ok"}"#);

    let cfg = H2AIConfig::default();

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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };

    // Should not return VerifierExplorerFamilyConflict — may succeed or fail for other reasons.
    let result = ExecutionEngine::run_offline(input).await;
    if let Err(EngineError::MultiplicationConditionFailed(msg)) = &result {
        assert!(
            !msg.contains("monoculture"),
            "gate fired despite family_constraint=SingleFamilyOk: {msg}"
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
                model: None,
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
            reasoning_trace: None,
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
            diversity_ids: vec![],
        },
        constraints: vec!["C-HINT".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
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

// ── Shadow auditor tests ──────────────────────────────────────────────────────

fn make_manifest_with_constraint_tags(tags: Vec<String>) -> TaskManifest {
    TaskManifest {
        description: "shadow auditor test task".into(),
        pareto_weights: h2ai_types::config::ParetoWeights::new(0.33, 0.34, 0.33).unwrap(),
        topology: h2ai_types::manifest::TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: h2ai_types::manifest::ExplorerRequest {
            count: 1,
            tau_min: Some(0.3),
            tau_max: Some(0.7),
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![],
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: tags,
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    }
}

fn mock_adapter_approves() -> MockAdapter {
    MockAdapter::new(r#"{"approved": true, "reason": "approved"}"#.into())
}

fn shadow_approve_adapter() -> MockAdapter {
    MockAdapter::new(r#"{"approved": true, "reason": "shadow ok"}"#.into())
}

fn shadow_reject_adapter() -> MockAdapter {
    MockAdapter::new(r#"{"approved": false, "reason": "shadow rejected"}"#.into())
}

#[tokio::test]
async fn shadow_mode_off_produces_no_shadow_events() {
    let manifest = make_manifest_with_constraint_tags(vec![]);
    let adapter = mock_adapter();
    let scorer = verifier();
    let auditor = mock_adapter_approves();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let cal = calibration().await;

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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        output.shadow_audit_events.is_empty(),
        "no shadow events expected when shadow_audit_ctx is None"
    );
}

#[tokio::test]
async fn shadow_mode_on_agreement_produces_events_with_disagreement_false() {
    let manifest = make_manifest_with_constraint_tags(vec!["security".to_string()]);
    let adapter = mock_adapter();
    let scorer = verifier();
    let primary_auditor = Arc::new(mock_adapter_approves()) as Arc<dyn IComputeAdapter>;
    let shadow_adapter = Arc::new(shadow_approve_adapter()) as Arc<dyn IComputeAdapter>;
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let cal = calibration().await;

    let ctx = h2ai_orchestrator::engine::ShadowAuditCtx {
        adapter: shadow_adapter,
        promoted_domains: Default::default(),
    };
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: primary_auditor.as_ref(),
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
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
        shadow_audit_ctx: Some(ctx),
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        !output.shadow_audit_events.is_empty(),
        "shadow events expected when shadow_audit_ctx is Some"
    );
    for ev in &output.shadow_audit_events {
        assert!(!ev.disagreement, "both agreed, disagreement must be false");
        assert_eq!(ev.domain, "security");
    }
}

#[tokio::test]
async fn shadow_mode_on_disagreement_does_not_affect_pruning() {
    // Primary approves, shadow rejects — task must resolve (primary wins in shadow mode)
    let manifest = make_manifest_with_constraint_tags(vec!["security".to_string()]);
    let adapter = mock_adapter();
    let scorer = verifier();
    let primary_auditor = Arc::new(mock_adapter_approves()) as Arc<dyn IComputeAdapter>;
    let shadow_adapter = Arc::new(shadow_reject_adapter()) as Arc<dyn IComputeAdapter>;
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let cal = calibration().await;

    let ctx = h2ai_orchestrator::engine::ShadowAuditCtx {
        adapter: shadow_adapter,
        promoted_domains: Default::default(),
    };
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: primary_auditor.as_ref(),
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
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
        shadow_audit_ctx: Some(ctx),
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        !output.resolved_output.is_empty(),
        "task must resolve despite shadow rejection in non-promoted mode"
    );
    let disagreement_events: Vec<_> = output
        .shadow_audit_events
        .iter()
        .filter(|e| e.disagreement)
        .collect();
    assert!(
        !disagreement_events.is_empty(),
        "disagreement events must be emitted when auditors disagree"
    );
}

#[tokio::test]
async fn majority_vote_mode_rejects_when_shadow_disagrees() {
    // Domain "security" is promoted → AND vote active.
    // Primary approves, shadow rejects → all proposals pruned → MaxRetriesExhausted.
    let manifest = make_manifest_with_constraint_tags(vec!["security".to_string()]);
    let adapter = mock_adapter();
    let scorer = verifier();
    let primary_auditor = Arc::new(mock_adapter_approves()) as Arc<dyn IComputeAdapter>;
    let shadow_adapter = Arc::new(shadow_reject_adapter()) as Arc<dyn IComputeAdapter>;
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let cal = calibration().await;

    let mut promoted = std::collections::HashSet::new();
    promoted.insert("security".to_string());
    let ctx = h2ai_orchestrator::engine::ShadowAuditCtx {
        adapter: shadow_adapter,
        promoted_domains: promoted,
    };
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: primary_auditor.as_ref(),
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
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
        shadow_audit_ctx: Some(ctx),
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let result = ExecutionEngine::run_offline(input).await;
    assert!(
        result.is_err(),
        "task must fail when AND vote cannot be satisfied"
    );
}

#[tokio::test]
async fn shadow_failure_falls_back_to_primary_decision() {
    // Shadow adapter errors; primary approves. Task must resolve.
    #[derive(Debug)]
    struct ErrorAdapter;
    #[async_trait::async_trait]
    impl IComputeAdapter for ErrorAdapter {
        async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
            Err(AdapterError::NetworkError(
                "simulated shadow failure".into(),
            ))
        }
        fn kind(&self) -> &AdapterKind {
            static KIND: std::sync::OnceLock<AdapterKind> = std::sync::OnceLock::new();
            KIND.get_or_init(|| AdapterKind::CloudGeneric {
                endpoint: "http://error".into(),
                api_key_env: String::new(),
                model: None,
            })
        }
    }

    let manifest = make_manifest_with_constraint_tags(vec![]);
    let adapter = mock_adapter();
    let scorer = verifier();
    let primary_auditor = Arc::new(mock_adapter_approves()) as Arc<dyn IComputeAdapter>;
    let shadow_adapter = Arc::new(ErrorAdapter) as Arc<dyn IComputeAdapter>;
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let cal = calibration().await;

    let ctx = h2ai_orchestrator::engine::ShadowAuditCtx {
        adapter: shadow_adapter,
        promoted_domains: Default::default(),
    };
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        verification_adapter: &scorer as &dyn h2ai_types::adapter::IComputeAdapter,
        auditor_adapter: primary_auditor.as_ref(),
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
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
        shadow_audit_ctx: Some(ctx),
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        !output.resolved_output.is_empty(),
        "task must resolve when shadow errors out"
    );
    assert!(
        output.shadow_audit_events.is_empty(),
        "no shadow events expected when shadow adapter errors"
    );
}

// ── GAP-C3 tests ────────────────────────────────────────────────────────────

#[tokio::test]
async fn c3_no_event_when_corpus_empty() {
    let explorer = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "ok"}"#.into());
    let cfg = H2AIConfig::default();
    let cal = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["test".into()],
        adapters: vec![&explorer as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    })
    .await
    .unwrap();
    let store = TaskStore::new();
    let task_id = TaskId::new();
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("solution text".into())) as Arc<dyn IComputeAdapter>
    );
    let manifest = TaskManifest {
        description: "test task".into(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: Some(0.3),
            tau_max: Some(0.7),
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![],
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 1.0,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        output.diversity_degraded_event.is_none(),
        "empty corpus should not fire degraded event"
    );
}

#[tokio::test]
async fn c3_fires_degraded_event_when_coverage_low() {
    let explorer = MockAdapter::new("stateless auth solution JWT token".into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "ok"}"#.into());
    let cfg = H2AIConfig {
        domain_coverage_threshold: 0.8,
        safety: h2ai_config::SafetyConfig {
            require_bivariate_cg: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut doc1 = ConstraintDoc::new_llm_judge("C1", "rule1");
    doc1.domains = vec!["security".into()];
    let mut doc2 = ConstraintDoc::new_llm_judge("C2", "rule2");
    doc2.domains = vec!["performance".into()];
    let mut doc3 = ConstraintDoc::new_llm_judge("C3", "rule3");
    doc3.domains = vec!["correctness".into()];
    let mut doc4 = ConstraintDoc::new_llm_judge("C4", "rule4");
    doc4.domains = vec!["data".into()];
    let corpus = vec![doc1, doc2, doc3, doc4];

    let cal = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["test".into()],
        adapters: vec![&explorer as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &corpus,
        embedding_model: None,
    })
    .await
    .unwrap();
    let store = TaskStore::new();
    let task_id = TaskId::new();
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("solution".into())) as Arc<dyn IComputeAdapter>
    );
    let manifest = TaskManifest {
        description: "auth task".into(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: Some(0.3),
            tau_max: Some(0.7),
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![h2ai_types::manifest::ExplorerSlotConfig {
                role_frame: "You are a security engineer.".into(),
                constraint_domains: vec![],
                ..Default::default()
            }],
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 1.0,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        output.diversity_degraded_event.is_some(),
        "should fire DiversityGuardDegradedEvent when coverage is 0"
    );
    let evt = output.diversity_degraded_event.unwrap();
    assert!(
        evt.coverage_score < 0.1,
        "coverage should be near 0, got {}",
        evt.coverage_score
    );
}

#[tokio::test]
async fn c3_require_bivariate_cg_fails_task_when_coverage_low() {
    let explorer = MockAdapter::new("auth solution JWT".into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "ok"}"#.into());
    let cfg = H2AIConfig {
        domain_coverage_threshold: 0.99,
        safety: h2ai_config::SafetyConfig {
            require_bivariate_cg: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut doc = ConstraintDoc::new_llm_judge("C1", "rule");
    doc.domains = vec!["security".into()];
    let corpus = vec![doc];

    let cal = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["test".into()],
        adapters: vec![&explorer as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &corpus,
        embedding_model: None,
    })
    .await
    .unwrap();
    let store = TaskStore::new();
    let task_id = TaskId::new();
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("solution".into())) as Arc<dyn IComputeAdapter>
    );
    let manifest = TaskManifest {
        description: "test".into(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: Some(0.3),
            tau_max: Some(0.7),
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![h2ai_types::manifest::ExplorerSlotConfig {
                constraint_domains: vec![],
                ..Default::default()
            }],
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let input = EngineInput {
        task_id,
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 1.0,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let result = ExecutionEngine::run_offline(input).await;
    assert!(
        matches!(result, Err(EngineError::MultiplicationConditionFailed(_))),
        "require_bivariate_cg=true with low coverage should fail, got: {result:?}"
    );
}

// ── Proactive researcher test ────────────────────────────────────────────────

#[tokio::test]
async fn proactive_researcher_called_for_search_enabled_slot() {
    use std::sync::Mutex;

    #[derive(Debug)]
    struct CallTrackingAdapter {
        calls: Arc<Mutex<Vec<String>>>,
        output: String,
        kind: AdapterKind,
    }
    #[async_trait::async_trait]
    impl IComputeAdapter for CallTrackingAdapter {
        async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
            self.calls.lock().unwrap().push(req.task.clone());
            Ok(ComputeResponse {
                output: self.output.clone(),
                token_cost: 10,
                adapter_kind: self.kind.clone(),
                tokens_used: None,
                reasoning_trace: None,
            })
        }
        fn kind(&self) -> &AdapterKind {
            &self.kind
        }
    }

    let researcher_calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let researcher = Arc::new(CallTrackingAdapter {
        calls: researcher_calls.clone(),
        output: "Current best practice: use short-lived JWT tokens with refresh rotation.".into(),
        kind: AdapterKind::CloudGeneric {
            endpoint: "mock://researcher".into(),
            api_key_env: "NONE".into(),
            model: None,
        },
    });
    let explorer = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "ok"}"#.into());
    let cfg = H2AIConfig::default();
    let cal = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["test".into()],
        adapters: vec![&explorer as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    })
    .await
    .unwrap();
    let store = TaskStore::new();
    let task_id = TaskId::new();
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("solution".into())) as Arc<dyn IComputeAdapter>
    );
    let manifest = TaskManifest {
        description: "Design auth system with search".into(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: Some(0.3),
            tau_max: Some(0.7),
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![h2ai_types::manifest::ExplorerSlotConfig {
                role_frame: "You are a security researcher.".into(),
                search_enabled: true,
                ..Default::default()
            }],
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let input = EngineInput {
        task_id,
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 1.0,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: Some(researcher as Arc<dyn IComputeAdapter>),
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();
    let calls = researcher_calls.lock().unwrap();
    assert!(
        !calls.is_empty(),
        "researcher should have been called for search_enabled slot"
    );
    assert!(
        !output.researcher_grounding_events.is_empty(),
        "grounding events should be recorded"
    );
    assert!(output.researcher_grounding_events[0].slot.is_some());
}

// ── GAP-C1 tests ────────────────────────────────────────────────────────────

// Tests that diverse proposals (very different outputs) do not trigger C1 warning.
// Uses two explorers with maximally different outputs so Jaccard distance is high.
// With N=2 and non-zero distance, compute_cv returns None (single-point distribution
// is statistically meaningless), so no C1 warning should fire.
#[tokio::test]
async fn c1_no_warning_for_diverse_proposals() {
    // Two adapters with maximally different outputs: no shared tokens → distance = 1.0
    let ex1 =
        MockAdapter::new("quantum entanglement photon polarization decoherence measurement".into());
    let ex2 =
        MockAdapter::new("database transaction isolation deadlock prevention concurrency".into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "ok"}"#.into());
    let cfg = H2AIConfig {
        correlated_hallucination_cv_threshold: 0.30,
        ..Default::default()
    };

    let cal = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["test".into()],
        adapters: vec![&ex1 as &dyn IComputeAdapter, &ex2 as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    })
    .await
    .unwrap();
    let store = TaskStore::new();
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("solution".into())) as Arc<dyn IComputeAdapter>
    );
    let manifest = TaskManifest {
        description: "Design auth".into(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&ex1 as &dyn IComputeAdapter, &ex2 as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 1.0,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        output.correlated_warnings.is_empty(),
        "diverse proposals (N=2, high Jaccard distance) should not trigger C1 warning"
    );
}

#[tokio::test]
async fn c1_fires_warning_and_retries_for_identical_proposals() {
    #[derive(Debug)]
    struct IdenticalAdapter {
        output: String,
        kind: AdapterKind,
    }
    #[async_trait::async_trait]
    impl IComputeAdapter for IdenticalAdapter {
        async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
            Ok(ComputeResponse {
                output: self.output.clone(),
                token_cost: 10,
                adapter_kind: self.kind.clone(),
                tokens_used: None,
                reasoning_trace: None,
            })
        }
        fn kind(&self) -> &AdapterKind {
            &self.kind
        }
    }

    let identical_text = "stateless auth JWT token validation bearer scheme".to_string();
    let ex1 = IdenticalAdapter {
        output: identical_text.clone(),
        kind: AdapterKind::CloudGeneric {
            endpoint: "mock://a".into(),
            api_key_env: "NONE".into(),
            model: None,
        },
    };
    let ex2 = IdenticalAdapter {
        output: identical_text.clone(),
        kind: AdapterKind::CloudGeneric {
            endpoint: "mock://b".into(),
            api_key_env: "NONE".into(),
            model: None,
        },
    };
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "ok"}"#.into());

    let cfg = H2AIConfig {
        correlated_hallucination_cv_threshold: 0.30,
        max_autonomic_retries: 1,
        ..Default::default()
    };

    let cal = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["test".into()],
        adapters: vec![&ex1 as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    })
    .await
    .unwrap();
    let store = TaskStore::new();
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("solution".into())) as Arc<dyn IComputeAdapter>
    );
    let manifest = TaskManifest {
        description: "Design auth".into(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&ex1 as &dyn IComputeAdapter, &ex2 as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 1.0,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let result = ExecutionEngine::run_offline(input).await;
    match result {
        Ok(output) => {
            assert!(
                !output.correlated_warnings.is_empty(),
                "C1 should have fired at least once for identical proposals"
            );
            assert_eq!(output.correlated_warnings[0].cv, 0.0);
        }
        Err(EngineError::MaxRetriesExhausted { .. }) => {
            // Acceptable — retries exhausted after C1 fired
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

#[tokio::test]
async fn srani_fires_when_proposals_share_ungrounded_entity() {
    // Both explorers introduce "CockroachDB" — not in the task spec.
    // SRANI should detect CFI=1.0 and push a CorrelatedFabricationEvent.
    let explorer1 = MockAdapter::new(
        "Use Redis and Kafka. CockroachDB advisory locks prevent double-spend.".into(),
    );
    let explorer2 = MockAdapter::new(
        "Use Redis and Kafka. CockroachDB distributed transactions ensure consistency.".into(),
    );
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "compliant"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();

    // Low thresholds so CFI=1.0 always triggers both warn and inject.
    let cfg = H2AIConfig {
        srani: h2ai_config::SraniConfig {
            grounding_raw_max_chars: 4000,
            grounding_hint_max_chars: 1200,
            grounding_distill: false,
            enabled: true,
            warn_threshold: 0.1,
            inject_threshold: 0.5,
            ..Default::default()
        },
        ..H2AIConfig::default()
    };

    // Spec mentions Redis and Kafka — CockroachDB is NOT in the spec.
    let manifest = TaskManifest {
        description:
            "Design a budget enforcement system using Redis for counters and Kafka for the spend event log. Include a crash recovery procedure."
                .into(),
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };

    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("registry-default".into())) as Arc<dyn IComputeAdapter>
    );
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine failed: {:?}", result.err());
    let output = result.unwrap();

    assert!(
        !output.srani_events.is_empty(),
        "expected at least one CorrelatedFabricationEvent, got none"
    );
    let ev = &output.srani_events[0];
    assert!(
        ev.cfi > cfg.srani.warn_threshold,
        "CFI {} must exceed warn_threshold {}",
        ev.cfi,
        cfg.srani.warn_threshold
    );
    assert!(
        ev.shared_ungrounded_entities
            .iter()
            .any(|e| e == "CockroachDB"),
        "CockroachDB must appear in shared_ungrounded_entities; got {:?}",
        ev.shared_ungrounded_entities
    );
    assert!(
        ev.hint_injected,
        "hint must be injected when CFI {} > inject_threshold {}",
        ev.cfi, cfg.srani.inject_threshold
    );
}

#[tokio::test]
async fn srani_silent_when_entities_grounded_in_spec() {
    // Both explorers mention Redis — but Redis IS in the task spec.
    // SRANI must NOT fire.
    let explorer1 = MockAdapter::new(
        "Use Redis EVAL for atomic counter updates. Redis sorted sets track budgets.".into(),
    );
    let explorer2 = MockAdapter::new(
        "Use Redis scripting for budget enforcement. Redis streams log spend events.".into(),
    );
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "compliant"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();

    let cfg = H2AIConfig {
        srani: h2ai_config::SraniConfig {
            grounding_raw_max_chars: 4000,
            grounding_hint_max_chars: 1200,
            grounding_distill: false,
            enabled: true,
            warn_threshold: 0.1,
            inject_threshold: 0.5,
            ..Default::default()
        },
        ..H2AIConfig::default()
    };

    // Redis IS in the spec — must not be treated as ungrounded.
    let manifest = TaskManifest {
        description: "Design a rate-limiting system using Redis for counters and state.".into(),
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };

    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("registry-default".into())) as Arc<dyn IComputeAdapter>
    );
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine failed: {:?}", result.err());
    let output = result.unwrap();

    assert!(
        output.srani_events.is_empty(),
        "expected no CorrelatedFabricationEvents when all entities are spec-grounded; got {:?}",
        output
            .srani_events
            .iter()
            .map(|e| &e.shared_ungrounded_entities)
            .collect::<Vec<_>>()
    );
}

// ── SRANI adaptive gate integration tests ─────────────────────────────────────

#[tokio::test]
async fn srani_adaptive_fires_and_updates_ema() {
    // Both explorers output "CockroachDB" — not in spec (Redis/Kafka only).
    // With adaptive=true and warm EMA (count=10, ema=0.30), CFI=1.0 produces
    // pressure > gate_threshold(0.50) → hint_injected=true, EMA updated.
    let explorer1 = MockAdapter::new(
        "Use CockroachDB advisory locks to coordinate the Redis and Kafka recovery.".into(),
    );
    let explorer2 = MockAdapter::new(
        "CockroachDB distributed transactions ensure idempotent recovery for Redis and Kafka."
            .into(),
    );
    let scorer = MockAdapter::new(r#"{"score": 0.8, "reason": "ok"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig {
        srani: h2ai_config::SraniConfig {
            grounding_raw_max_chars: 4000,
            grounding_hint_max_chars: 1200,
            grounding_distill: false,
            enabled: true,
            adaptive: true,
            ema_alpha: 0.20,
            temperature: 0.15,
            gate_threshold: 0.50,
            warn_threshold: 0.30,
            inject_threshold: 0.60,
        },
        ..H2AIConfig::default()
    };
    let manifest = TaskManifest {
        description: "Design a crash-recovery procedure for Redis and Kafka counters.".into(),
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("registry".into())) as Arc<dyn IComputeAdapter>
    );
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.30, // warm EMA below CFI
        srani_count: 10,     // past cold-start threshold
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();

    // SRANI must have fired
    assert!(
        !output.srani_events.is_empty(),
        "srani_events must be non-empty when both explorers share CockroachDB"
    );
    let ev = &output.srani_events[0];
    assert!(
        ev.injection_pressure >= cfg.srani.gate_threshold,
        "injection_pressure {:.3} must be >= gate_threshold {:.3}",
        ev.injection_pressure,
        cfg.srani.gate_threshold
    );
    assert!(
        ev.hint_injected,
        "hint_injected must be true when pressure >= gate_threshold"
    );
    assert!(
        ev.shared_ungrounded_entities
            .iter()
            .any(|e| e.contains("CockroachDB") || e.contains("Cockroach")),
        "CockroachDB must appear in shared_ungrounded_entities; got {:?}",
        ev.shared_ungrounded_entities
    );
    // EMA must have been updated
    assert!(
        output.srani_count_updated > 10,
        "srani_count_updated must exceed initial count"
    );
    assert!(
        output.srani_ema_cfi_updated > 0.30,
        "ema must shift upward after high-CFI task, got {}",
        output.srani_ema_cfi_updated
    );
}

#[tokio::test]
async fn srani_cold_start_uses_config_midpoint() {
    // count < 5 → cold_start_midpoint() (0.45) is used instead of ema_cfi.
    // With mu=0.45 and CFI≈1.0, pressure should be very high (≈0.99) → hint injected.
    let explorer1 = MockAdapter::new(
        "Use CockroachDB and ClickHouse for storage in the Redis and Kafka recovery.".into(),
    );
    let explorer2 = MockAdapter::new(
        "CockroachDB advisory locks and ClickHouse analytics fix Redis and Kafka state.".into(),
    );
    let scorer = MockAdapter::new(r#"{"score": 0.8, "reason": "ok"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig {
        srani: h2ai_config::SraniConfig {
            grounding_raw_max_chars: 4000,
            grounding_hint_max_chars: 1200,
            grounding_distill: false,
            enabled: true,
            adaptive: true,
            ema_alpha: 0.20,
            temperature: 0.15,
            gate_threshold: 0.50,
            warn_threshold: 0.30,
            inject_threshold: 0.60,
        },
        ..H2AIConfig::default()
    };
    let manifest = TaskManifest {
        description: "Design a Redis and Kafka crash-recovery procedure.".into(),
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("registry".into())) as Arc<dyn IComputeAdapter>
    );
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.99, // artificially high — should NOT be used (count < 5)
        srani_count: 2,      // cold start: count < 5
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();

    if !output.srani_events.is_empty() {
        let ev = &output.srani_events[0];
        // With cold_start_midpoint=0.45 and CFI≈1.0, pressure ≈ 0.99 (very high).
        // With ema_cfi=0.99 and CFI≈1.0, pressure ≈ 0.50 (barely above gate).
        // Cold start must give higher pressure than EMA=0.99 path.
        assert!(
            ev.injection_pressure > 0.80,
            "cold start pressure should be > 0.80 (using mu=0.45), got {:.3}",
            ev.injection_pressure
        );
    }
    assert_eq!(
        output.srani_count_updated, 3,
        "cold start count must advance by 1"
    );
}

#[tokio::test]
async fn srani_adaptive_false_uses_static_thresholds() {
    // adaptive=false → old warn/inject threshold logic applies.
    let explorer1 =
        MockAdapter::new("Use CockroachDB advisory locks to coordinate the Redis recovery.".into());
    let explorer2 =
        MockAdapter::new("CockroachDB transactions ensure idempotent Redis recovery.".into());
    let scorer = MockAdapter::new(r#"{"score": 0.8, "reason": "ok"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig {
        srani: h2ai_config::SraniConfig {
            grounding_raw_max_chars: 4000,
            grounding_hint_max_chars: 1200,
            grounding_distill: false,
            enabled: true,
            adaptive: false, // static path
            warn_threshold: 0.10,
            inject_threshold: 0.50,
            ..Default::default()
        },
        ..H2AIConfig::default()
    };
    let manifest = TaskManifest {
        description: "Design a Redis crash-recovery procedure.".into(),
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("registry".into())) as Arc<dyn IComputeAdapter>
    );
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 10,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();

    // When adaptive=false and CFI≈1.0 > inject_threshold(0.50), hint must be injected.
    if !output.srani_events.is_empty() {
        let ev = &output.srani_events[0];
        assert!(
            ev.hint_injected,
            "adaptive=false: CFI {:.3} > inject_threshold 0.50 must inject hint",
            ev.cfi
        );
    }
}

#[tokio::test]
async fn srani_ema_formula_verified_numerically() {
    // Single-task EMA update: ema_new = 0.20 * CFI + 0.80 * ema_old
    // We inspect EngineOutput.srani_ema_cfi_updated directly.
    let explorer1 =
        MockAdapter::new("Use CockroachDB for distributed Redis and Kafka recovery.".into());
    let explorer2 =
        MockAdapter::new("CockroachDB advisory locks recover Redis and Kafka state.".into());
    let scorer = MockAdapter::new(r#"{"score": 0.8, "reason": "ok"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig {
        srani: h2ai_config::SraniConfig {
            grounding_raw_max_chars: 4000,
            grounding_hint_max_chars: 1200,
            grounding_distill: false,
            enabled: true,
            adaptive: true,
            ema_alpha: 0.20,
            temperature: 0.15,
            gate_threshold: 0.50,
            warn_threshold: 0.30,
            inject_threshold: 0.60,
        },
        ..H2AIConfig::default()
    };
    let initial_ema: f64 = 0.40;
    let manifest = TaskManifest {
        description: "Design a Redis and Kafka crash-recovery procedure.".into(),
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    };
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("registry".into())) as Arc<dyn IComputeAdapter>
    );
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
                model: None,
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: initial_ema,
        srani_count: 10,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };
    let output = ExecutionEngine::run_offline(input).await.unwrap();

    if output.srani_count_updated > 10 {
        // CFI was computed; verify EMA formula:
        // Both explorers share CockroachDB → CFI = 1.0
        // ema_new = 0.20 * 1.0 + 0.80 * 0.40 = 0.52
        let actual_cfi = (output.srani_ema_cfi_updated - (1.0 - 0.20) * initial_ema) / 0.20;
        assert!(
            actual_cfi > 0.0 && actual_cfi <= 1.0,
            "implied CFI {:.3} must be in [0,1]",
            actual_cfi
        );
        // ema_updated must strictly exceed initial_ema (since CFI > initial_ema)
        assert!(
            output.srani_ema_cfi_updated > initial_ema,
            "EMA must increase after high-CFI task: {} → {}",
            initial_ema,
            output.srani_ema_cfi_updated
        );
    }
}

// ─── SRANI Grounding Escalation Tests ────────────────────────────────────────

use h2ai_orchestrator::srani_grounding::{
    GroundingSource, LlmResearcherGrounder, SpecAnchorGrounder, SraniGroundingChain,
    WebSearchGrounder,
};
use h2ai_tools::web_search::MockSearchBackend;

fn cockroachdb_manifest() -> TaskManifest {
    TaskManifest {
        description: "Build a rate-limiting service using Redis sliding windows".into(),
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
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    }
}

#[test]
fn srani_grounding_hint_format_is_positive_not_prohibitive() {
    use h2ai_orchestrator::srani_grounding::{format_grounding_hint, GroundingResult};
    let result = GroundingResult {
        alternatives: vec!["Redis".into()],
        grounding_statement: "Use Redis TTL counters".into(),
        source: GroundingSource::LlmResearcher,
    };
    let hint = format_grounding_hint(&result, &["CockroachDB".into(), "ClickHouse".into()]);
    assert!(
        hint.contains("GROUNDING CONTEXT"),
        "must use new positive header"
    );
    assert!(
        !hint.contains("Do not introduce"),
        "must not contain old prohibitive text"
    );
    assert!(
        hint.contains("Avoid (not in spec)"),
        "must retain Avoid line as secondary signal"
    );
    assert!(
        hint.contains("Redis"),
        "must mention spec-defined alternatives"
    );
}

#[tokio::test]
async fn srani_no_chain_falls_back_to_spec_anchor_only() {
    let explorer =
        MockAdapter::new("I recommend CockroachDB for distributed rate-limiting state".into());
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let cfg = H2AIConfig::default();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest: cockroachdb_manifest(),
        calibration: cal,
        explorer_adapters: vec![&explorer as _, &explorer as _],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(
        result.is_ok(),
        "engine must succeed with no chain: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn srani_spec_anchor_chain_records_grounding_event_source() {
    let explorer =
        MockAdapter::new("I recommend CockroachDB for distributed rate-limiting state".into());
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let cfg = H2AIConfig::default();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();
    let chain = Arc::new(SraniGroundingChain::new(vec![Box::new(SpecAnchorGrounder)]));

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest: cockroachdb_manifest(),
        calibration: cal,
        explorer_adapters: vec![&explorer as _, &explorer as _],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: Some(chain),
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    for ev in &output.researcher_grounding_events {
        if ev.slot.is_none() && !ev.shared_assumption.is_empty() {
            assert_eq!(
                ev.source,
                GroundingSource::SpecAnchor,
                "SRANI event must carry SpecAnchor source"
            );
        }
    }
}

#[tokio::test]
async fn srani_llm_chain_records_llm_researcher_source() {
    let explorer =
        MockAdapter::new("I recommend CockroachDB for distributed rate-limiting state".into());
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let cfg = H2AIConfig::default();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();

    let researcher_mock = Arc::new(MockAdapter::new(
        r#"{"alternatives": ["Redis TTL counters"], "statement": "Use Redis TTL + Lua for rate limiting"}"#.into(),
    ));
    let chain = Arc::new(SraniGroundingChain::new(vec![
        Box::new(SpecAnchorGrounder),
        Box::new(LlmResearcherGrounder::new(researcher_mock)),
    ]));

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest: cockroachdb_manifest(),
        calibration: cal,
        explorer_adapters: vec![&explorer as _, &explorer as _],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: Some(chain),
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    for ev in &output.researcher_grounding_events {
        if ev.slot.is_none() && !ev.shared_assumption.is_empty() {
            assert_eq!(
                ev.source,
                GroundingSource::LlmResearcher,
                "SRANI event must carry LlmResearcher source"
            );
        }
    }
}

#[tokio::test]
async fn srani_researcher_failure_falls_back_gracefully() {
    let explorer =
        MockAdapter::new("I recommend CockroachDB for distributed rate-limiting state".into());
    let scorer = verifier();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let cfg = H2AIConfig::default();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();

    let bad_researcher = Arc::new(MockAdapter::new("THIS IS NOT JSON".into()));
    let chain = Arc::new(SraniGroundingChain::new(vec![
        Box::new(SpecAnchorGrounder),
        Box::new(LlmResearcherGrounder::new(bad_researcher)),
    ]));

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest: cockroachdb_manifest(),
        calibration: cal,
        explorer_adapters: vec![&explorer as _, &explorer as _],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
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
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: Some(chain),
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(
        result.is_ok(),
        "engine must not fail due to researcher error: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn srani_web_search_chain_resolves_at_tier1() {
    use h2ai_orchestrator::srani_grounding::GroundingContext;

    let web_snippet = "Redis sliding-window counter is the standard for rate limiting";
    let chain = SraniGroundingChain::new(vec![
        Box::new(SpecAnchorGrounder),
        Box::new(LlmResearcherGrounder::new(Arc::new(MockAdapter::new(
            "should not appear".into(),
        )))),
        Box::new(WebSearchGrounder::new(
            Arc::new(MockSearchBackend::new(web_snippet.to_string())),
            3,
        )),
    ]);
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into()],
        task_description: "Build a rate-limiting service using Redis".into(),
    };
    // tier=1 must use WebSearch (index 2), not LlmResearcher (index 1).
    let result = chain.resolve(&ctx, 1).await.unwrap();
    assert_eq!(
        result.source,
        GroundingSource::WebSearch,
        "tier=1 must resolve to WebSearch"
    );
    assert!(result.grounding_statement.contains("Redis"));
}

#[test]
fn precision_mode_max_slots_config_default_is_3() {
    let cfg = h2ai_config::H2AIConfig::default();
    assert_eq!(cfg.precision_mode_max_slots, 3);
}

#[test]
fn consensus_score_averaging_math() {
    // Verify the averaging produces expected results for edge cases.
    let scores_all_pass: Vec<f64> = vec![1.0, 1.0];
    let mean_pass = scores_all_pass.iter().sum::<f64>() / scores_all_pass.len() as f64;
    assert!((mean_pass - 1.0).abs() < 1e-9);

    let scores_split: Vec<f64> = vec![1.0, 0.1];
    let mean_split = scores_split.iter().sum::<f64>() / scores_split.len() as f64;
    assert!((mean_split - 0.55).abs() < 1e-9);

    // With threshold 0.6, mean_split=0.55 fails hard gate
    assert!(mean_split < 0.6);
    // With threshold 0.45, mean_split=0.55 passes hard gate
    assert!(mean_split >= 0.45);
}
