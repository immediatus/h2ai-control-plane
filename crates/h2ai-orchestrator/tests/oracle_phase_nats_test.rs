#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Integration tests for NATS-backed branches of `h2ai_orchestrator::phases::oracle`.
//!
//! Requires a running NATS server at 127.0.0.1:4222. Tests soft-skip when unavailable.
//!
//! Covered branches:
//! - gate passed (gate_passed = true)   → `StepResult::Done(Some(true))`
//! - gate failed (gate_passed = false)  → `StepResult::EarlyExit(OracleBlocked)`
//! - malformed response                 → on_timeout policy fallback
//! - timeout (no responder)             → on_timeout policy fallback

use futures::StreamExt as _;
use std::sync::Arc;

use chrono::Utc;
use h2ai_config::{H2AIConfig, OracleGateConfig};
use h2ai_constraints::types::ConstraintDoc;
use h2ai_orchestrator::{
    engine::EngineInput,
    phases::{self, ExitReason, StepResult},
    tao_loop::TaoMultiplierEstimator,
    task_store::TaskStore,
};
use h2ai_test_utils::mock_adapter;
use h2ai_types::{
    adapter::{AdapterRegistry, IComputeAdapter},
    config::ParetoWeights,
    config::{AdapterKind, AuditorConfig, TaoConfig, VerificationConfig},
    events::{CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode},
    identity::{TaskId, TenantId},
    manifest::{ExplorerRequest, TaskManifest, TopologyRequest},
    sizing::{CoherencyCoefficients, CoordinationThreshold},
};

async fn try_connect_nats() -> Option<Arc<async_nats::Client>> {
    let client = async_nats::connect("nats://127.0.0.1:4222").await.ok()?;
    Some(Arc::new(client))
}

fn oracle_cfg(subject: &str, on_timeout: &str, timeout_secs: u64) -> H2AIConfig {
    H2AIConfig {
        oracle_gate: OracleGateConfig {
            enabled: true,
            subject: subject.to_string(),
            timeout_secs,
            on_timeout: on_timeout.to_string(),
            ..OracleGateConfig::default()
        },
        ..H2AIConfig::default()
    }
}

fn gate_response_bytes(gate_passed: bool) -> bytes::Bytes {
    serde_json::to_vec(&serde_json::json!({
        "task_id": "test",
        "gate_passed": gate_passed,
        "confidence": 0.9,
        "summary": "test",
        "checked_proposals": 1,
        "passed_proposals": if gate_passed { 1 } else { 0 },
        "timestamp": "2024-01-01T00:00:00Z"
    }))
    .unwrap()
    .into()
}

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
        description: "Stub task".into(),
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
                provider: Default::default(),
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

#[tokio::test]
async fn oracle_phase_nats_gate_passed_returns_done_true() {
    let Some(nats) = try_connect_nats().await else {
        return;
    };

    let subject = "h2ai.oracle.gate.test.pass";
    let mut sub = nats.subscribe(subject).await.unwrap();

    let responder = nats.clone();
    tokio::spawn(async move {
        if let Some(msg) = sub.next().await {
            if let Some(reply) = msg.reply {
                let _ = responder.publish(reply, gate_response_bytes(true)).await;
            }
        }
    });

    let adapter = mock_adapter("output");
    let cfg = oracle_cfg(subject, "pass", 5);
    let store = TaskStore::new();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter("mock")) as Arc<dyn IComputeAdapter>);
    let mut engine_input = make_engine_input(&adapter, &adapter, &cfg, store, &registry);
    engine_input.nats_raw = Some(nats);

    let result = phases::oracle::run(phases::oracle::Input {
        engine_input: &engine_input,
    })
    .await;

    assert!(
        matches!(result, StepResult::Done(Some(true))),
        "gate passed must return Done(Some(true))"
    );
}

#[tokio::test]
async fn oracle_phase_nats_gate_failed_returns_oracle_blocked() {
    let Some(nats) = try_connect_nats().await else {
        return;
    };

    let subject = "h2ai.oracle.gate.test.fail";
    let mut sub = nats.subscribe(subject).await.unwrap();

    let responder = nats.clone();
    tokio::spawn(async move {
        if let Some(msg) = sub.next().await {
            if let Some(reply) = msg.reply {
                let _ = responder.publish(reply, gate_response_bytes(false)).await;
            }
        }
    });

    let adapter = mock_adapter("output");
    let cfg = oracle_cfg(subject, "pass", 5);
    let store = TaskStore::new();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter("mock")) as Arc<dyn IComputeAdapter>);
    let mut engine_input = make_engine_input(&adapter, &adapter, &cfg, store, &registry);
    engine_input.nats_raw = Some(nats);

    let result = phases::oracle::run(phases::oracle::Input {
        engine_input: &engine_input,
    })
    .await;

    assert!(
        matches!(result, StepResult::EarlyExit(ExitReason::OracleBlocked)),
        "gate failed must return EarlyExit(OracleBlocked)"
    );
}

#[tokio::test]
async fn oracle_phase_nats_malformed_response_falls_back_to_on_timeout_pass() {
    let Some(nats) = try_connect_nats().await else {
        return;
    };

    let subject = "h2ai.oracle.gate.test.malformed";
    let mut sub = nats.subscribe(subject).await.unwrap();

    let responder = nats.clone();
    tokio::spawn(async move {
        if let Some(msg) = sub.next().await {
            if let Some(reply) = msg.reply {
                let _ = responder
                    .publish(reply, bytes::Bytes::from("not-json"))
                    .await;
            }
        }
    });

    let adapter = mock_adapter("output");
    let cfg = oracle_cfg(subject, "pass", 5);
    let store = TaskStore::new();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter("mock")) as Arc<dyn IComputeAdapter>);
    let mut engine_input = make_engine_input(&adapter, &adapter, &cfg, store, &registry);
    engine_input.nats_raw = Some(nats);

    let result = phases::oracle::run(phases::oracle::Input {
        engine_input: &engine_input,
    })
    .await;

    assert!(
        matches!(result, StepResult::Done(Some(true))),
        "malformed response with on_timeout=pass must return Done(Some(true))"
    );
}

#[tokio::test]
async fn oracle_phase_nats_timeout_with_on_timeout_fail_returns_oracle_blocked() {
    let Some(nats) = try_connect_nats().await else {
        return;
    };

    let subject = "h2ai.oracle.gate.test.timeout_fail";
    // No subscriber — request times out after 1 second.
    let adapter = mock_adapter("output");
    let cfg = oracle_cfg(subject, "fail", 1);
    let store = TaskStore::new();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter("mock")) as Arc<dyn IComputeAdapter>);
    let mut engine_input = make_engine_input(&adapter, &adapter, &cfg, store, &registry);
    engine_input.nats_raw = Some(nats);

    let result = phases::oracle::run(phases::oracle::Input {
        engine_input: &engine_input,
    })
    .await;

    assert!(
        matches!(result, StepResult::EarlyExit(ExitReason::OracleBlocked)),
        "timeout with on_timeout=fail must return EarlyExit(OracleBlocked)"
    );
}
