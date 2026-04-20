use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_context::adr::AdrConstraints;
use h2ai_orchestrator::engine::{EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_types::config::{AdapterKind, AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};

fn mock_adapter() -> MockAdapter {
    MockAdapter::new("The proposed solution uses stateless JWT auth.".into())
}

async fn calibration() -> h2ai_types::events::CalibrationCompletedEvent {
    let adapter = mock_adapter();
    let cfg = H2AIConfig::default();
    CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Calibrate".into(), "Second task".into(), "Third".into()],
        adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        cfg: &cfg,
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn engine_runs_ensemble_to_semilattice() {
    let adapter = mock_adapter();
    let auditor = mock_adapter();
    let cal = calibration().await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let corpus = vec![AdrConstraints {
        source: "ADR-001".into(),
        keywords: ["stateless".to_string(), "auth".to_string()]
            .into_iter()
            .collect(),
    }];

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

    let input = EngineInput {
        manifest,
        calibration: cal,
        explorer_adapters: vec![
            &adapter as &dyn h2ai_types::adapter::IComputeAdapter,
            &adapter,
        ],
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
        adr_corpus: corpus,
        cfg: &cfg,
        store: store.clone(),
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
    let auditor = mock_adapter();
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
    let corpus = vec![AdrConstraints {
        source: "ADR-001".into(),
        keywords: [
            "microservice",
            "stateless",
            "distributed",
            "consensus",
            "byzantine",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    }];

    let input = EngineInput {
        manifest,
        calibration: cal,
        explorer_adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
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
        adr_corpus: corpus,
        cfg: &cfg,
        store: store.clone(),
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_err());
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("J_eff") || err_str.contains("context underflow"),
        "{err_str}"
    );
}
