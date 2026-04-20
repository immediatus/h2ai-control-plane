use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_context::adr::parse_adr;
use h2ai_orchestrator::engine::{EngineError, EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use std::sync::Arc;
use h2ai_types::config::{
    AdapterKind, AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig,
};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};

async fn make_engine_input<'a>(
    explorer_adapters: Vec<&'a dyn IComputeAdapter>,
    verification_adapter: &'a dyn IComputeAdapter,
    auditor_adapter: &'a dyn IComputeAdapter,
    cfg: &'a H2AIConfig,
    store: TaskStore,
    registry: &'a AdapterRegistry,
) -> EngineInput<'a> {
    // Build calibration using the same pattern as engine_test.rs
    let cal_adapter = MockAdapter::new("The proposed solution uses stateless JWT auth.".into());
    let cal_cfg = H2AIConfig::default();
    let cal = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Calibrate".into(), "Second task".into(), "Third".into()],
        adapters: vec![&cal_adapter as &dyn IComputeAdapter],
        cfg: &cal_cfg,
    })
    .await
    .unwrap();

    let corpus = vec![parse_adr(
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
            count: explorer_adapters.len(),
            tau_min: Some(0.5),
            tau_max: Some(0.5),
            roles: vec![],
            review_gates: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
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
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg,
        store,
        nats_dispatch: None,
        registry,
    }
}

#[tokio::test]
async fn engine_deadline_exceeded_on_zero_second_budget() {
    // deadline_secs = 0 → deadline is Instant::now() at construction.
    // By the time retry_count=0 iteration starts, Instant::now() >= deadline → DeadlineExceeded.
    let explorer = MockAdapter::new("stateless JWT auth — ADR-001 compliant".into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "compliant"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cfg = H2AIConfig {
        task_deadline_secs: Some(0),
        ..H2AIConfig::default()
    };
    let store = TaskStore::new();

    let reasoning: Arc<dyn IComputeAdapter> = Arc::new(MockAdapter::new("mock output".into()));
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
        matches!(out.unwrap_err(), EngineError::DeadlineExceeded { .. }),
        "expected DeadlineExceeded variant"
    );
}

#[tokio::test]
async fn engine_no_deadline_runs_normally() {
    // task_deadline_secs = None (default) → no deadline → task completes
    let explorer = MockAdapter::new("stateless JWT auth — ADR-001 compliant".into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "compliant"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cfg = H2AIConfig::default(); // task_deadline_secs defaults to None
    let store = TaskStore::new();

    let reasoning: Arc<dyn IComputeAdapter> = Arc::new(MockAdapter::new("mock output".into()));
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
