#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Unit tests for `h2ai_orchestrator::phases::oracle`.
//!
//! Only the two NATS-free branches are tested here:
//! - gate disabled (default config)   → `StepResult::Done(None)`
//! - gate enabled but nats_raw = None → `StepResult::Done(None)`
//!
//! The NATS-request branches are covered by integration tests that require a live server.

use std::sync::Arc;

use chrono::Utc;
use h2ai_config::{H2AIConfig, OracleGateConfig};
use h2ai_constraints::types::ConstraintDoc;
use h2ai_orchestrator::{
    engine::EngineInput,
    phases::{self, StepResult},
    tao_loop::TaoMultiplierEstimator,
    task_store::TaskStore,
};
use h2ai_test_utils::MockAdapter;
use h2ai_types::{
    adapter::{AdapterRegistry, IComputeAdapter},
    config::ParetoWeights,
    config::{AdapterKind, AuditorConfig, TaoConfig, VerificationConfig},
    events::{CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode},
    identity::{TaskId, TenantId},
    manifest::{ExplorerRequest, TaskManifest, TopologyRequest},
    sizing::{CoherencyCoefficients, CoordinationThreshold},
};

fn stub_calibration() -> CalibrationCompletedEvent {
    let coefficients = CoherencyCoefficients::new(0.10, 0.020, vec![0.60, 0.70, 0.80])
        .expect("valid coefficients");
    let coordination_threshold = CoordinationThreshold::from_calibration(&coefficients, 0.3);
    CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients,
        coordination_threshold,
        ensemble: None,
        eigen: None,
        timestamp: Utc::now(),
        pairwise_beta: None,
        cg_mode: CgMode::default(),
        adapter_families: vec!["Mock".into()],
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: CalibrationQuality::default(),
        calibration_source: CalibrationSource::Measured,
        beta_quality: None,
    }
}

fn stub_manifest() -> TaskManifest {
    TaskManifest {
        description: "Stub task for oracle phase tests".into(),
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
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: TenantId::default_tenant(),
    }
}

fn make_engine_input<'a>(
    explorer: &'a dyn IComputeAdapter,
    auditor: &'a dyn IComputeAdapter,
    cfg: &'a H2AIConfig,
    store: TaskStore,
    registry: &'a AdapterRegistry,
) -> EngineInput<'a> {
    EngineInput {
        task_id: TaskId::new(),
        manifest: stub_manifest(),
        calibration: stub_calibration(),
        explorer_adapters: vec![explorer],
        verification_adapter: auditor,
        auditor_adapter: auditor,
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
        constraint_corpus: vec![ConstraintDoc::new_llm_judge("STUB-1", "stub constraint")],
        embedding_model: None,
        cfg,
        store,
        nats_dispatch: None,
        registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            TaoMultiplierEstimator::new_with_alpha(0.1),
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
        conformal_margin: 0.0,
    }
}

// ── gate disabled (default config) ───────────────────────────────────────────

#[tokio::test]
async fn oracle_phase_gate_disabled_returns_done_none() {
    let adapter = MockAdapter::new("output".into());
    let cfg = H2AIConfig::default(); // oracle_gate.enabled = false
    let store = TaskStore::new();
    let registry =
        AdapterRegistry::new(Arc::new(MockAdapter::new("mock".into())) as Arc<dyn IComputeAdapter>);

    let engine_input = make_engine_input(&adapter, &adapter, &cfg, store, &registry);
    let result = phases::oracle::run(phases::oracle::Input {
        engine_input: &engine_input,
    })
    .await;

    assert!(
        matches!(result, StepResult::Done(None)),
        "disabled gate must return Done(None)"
    );
}

// ── gate enabled, no NATS client ─────────────────────────────────────────────

#[tokio::test]
async fn oracle_phase_gate_enabled_no_nats_returns_done_none() {
    let adapter = MockAdapter::new("output".into());
    let cfg = H2AIConfig {
        oracle_gate: OracleGateConfig {
            enabled: true,
            ..OracleGateConfig::default()
        },
        ..H2AIConfig::default()
    };
    let store = TaskStore::new();
    let registry =
        AdapterRegistry::new(Arc::new(MockAdapter::new("mock".into())) as Arc<dyn IComputeAdapter>);

    // nats_raw is None (set in make_engine_input) → gate skips to Done(None).
    let engine_input = make_engine_input(&adapter, &adapter, &cfg, store, &registry);
    let result = phases::oracle::run(phases::oracle::Input {
        engine_input: &engine_input,
    })
    .await;

    assert!(
        matches!(result, StepResult::Done(None)),
        "enabled gate with no NATS client must return Done(None)"
    );
}
