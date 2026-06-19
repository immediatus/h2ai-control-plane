use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_test_utils::mock_adapter;

use h2ai_orchestrator::engine::{EngineError, EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{
    AdapterKind, AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig,
};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use std::sync::Arc;

async fn make_engine_input<'a>(
    explorer_adapters: Vec<&'a dyn IComputeAdapter>,
    verification_adapter: &'a dyn IComputeAdapter,
    auditor_adapter: &'a dyn IComputeAdapter,
    cfg: &'a H2AIConfig,
    store: TaskStore,
    registry: &'a AdapterRegistry,
) -> EngineInput<'a> {
    // Build calibration using the same pattern as engine_test.rs
    let cal_adapter = mock_adapter("The proposed solution uses stateless JWT auth.");
    let cal_cfg = H2AIConfig::default();
    let cal = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Calibrate".into(), "Second task".into(), "Third".into()],
        adapters: vec![&cal_adapter as &dyn IComputeAdapter],
        cfg: &cal_cfg,
        constraint_corpus: &[],
        embedding_model: None,
    })
    .await
    .unwrap();

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
            count: explorer_adapters.len(),
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

    EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters,
        verification_adapter,
        auditor_adapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
                model: None,
                provider: Default::default(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        embedding_model: None,
        cfg,
        store,
        nats_dispatch: None,
        registry,
        tao_multiplier: 0.6,
        tao_estimator: std::sync::Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    }
}

#[tokio::test]
async fn engine_deadline_exceeded_on_zero_second_budget() {
    // deadline_secs = 0 → deadline is Instant::now() at construction.
    // By the time retry_count=0 iteration starts, Instant::now() >= deadline → DeadlineExceeded.
    let explorer = mock_adapter("stateless JWT auth — ADR-001 compliant");
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "compliant"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let cfg = H2AIConfig {
        task_deadline_secs: Some(0),
        ..H2AIConfig::default()
    };
    let store = TaskStore::new();

    let reasoning: Arc<dyn IComputeAdapter> = Arc::new(mock_adapter("mock output"));
    let registry = AdapterRegistry::new(reasoning);
    let out = ExecutionEngine::run_offline(
        make_engine_input(
            vec![&explorer as &dyn IComputeAdapter],
            &scorer as &dyn IComputeAdapter,
            &auditor as &dyn IComputeAdapter,
            &cfg,
            store,
            &registry,
        )
        .await,
    )
    .await;

    assert!(out.is_err(), "expected DeadlineExceeded error");
    assert!(
        matches!(out.unwrap_err(), (EngineError::DeadlineExceeded { .. }, _)),
        "expected DeadlineExceeded variant"
    );
}

#[tokio::test]
async fn engine_no_deadline_runs_normally() {
    // task_deadline_secs = None (default) → no deadline → task completes
    let explorer = mock_adapter("stateless JWT auth — ADR-001 compliant");
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "compliant"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let cfg = H2AIConfig::default(); // task_deadline_secs defaults to None
    let store = TaskStore::new();

    let reasoning: Arc<dyn IComputeAdapter> = Arc::new(mock_adapter("mock output"));
    let registry = AdapterRegistry::new(reasoning);
    let out = ExecutionEngine::run_offline(
        make_engine_input(
            vec![&explorer as &dyn IComputeAdapter],
            &scorer as &dyn IComputeAdapter,
            &auditor as &dyn IComputeAdapter,
            &cfg,
            store,
            &registry,
        )
        .await,
    )
    .await;

    assert!(
        out.is_ok(),
        "no deadline should complete normally: {:?}",
        out.err()
    );
}
