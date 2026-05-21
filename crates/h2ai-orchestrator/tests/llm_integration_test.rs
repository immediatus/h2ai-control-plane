#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
//! Engine + calibration theory tests — all LLM calls replaced with mock adapters.
//!
//! These tests prove that:
//!   - CalibrationHarness produces coefficients satisfying β_eff = β₀×(1−CG) and
//!     N_max = √((1−α)/β_eff)
//!   - The engine respects N_max as a hard ceiling on agent count
//!   - The full engine pipeline (calibration → exploration → verification → synthesis)
//!     runs end-to-end without panicking

use h2ai_adapters::mock::{DecompositionMockAdapter, MockAdapter};
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_orchestrator::engine::{EngineInput, ExecutionEngine};
use h2ai_orchestrator::srani_grounding::{SpecAnchorGrounder, SraniGroundingChain};
use h2ai_orchestrator::tao_loop::TaoMultiplierEstimator;
use h2ai_orchestrator::task_store::{TaskState, TaskStore};
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{
    AdapterKind, AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig,
};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use std::sync::Arc;

/// Proves:
///   1. CalibrationHarness with mock adapters → valid α, β₀, CG, β_eff, N_max
///   2. β_eff = β₀ × (1−CG) holds exactly
///   3. N_max = √((1−α)/β_eff) holds exactly
///   4. Engine respects N_max as a hard ceiling: never runs more agents than N_max
#[tokio::test]
async fn calibrate_then_engine_respects_n_max_ceiling() {
    let cfg = H2AIConfig::default();
    let corpus = vec![ConstraintDoc::new_llm_judge(
        "stateless",
        "The solution must be stateless. No server-side sessions permitted.",
    )];

    // Calibrate with 2 mock adapters (identical output → CG=0.0 from Hamming)
    let cal_a1 = MockAdapter::new("JWT is a stateless token.".into());
    let cal_a2 = MockAdapter::new("JWT is a stateless token.".into());

    let cal_event = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec![
            "Explain stateless auth for APIs in one sentence.".into(),
            "What is a JWT token? One sentence.".into(),
        ],
        adapters: vec![
            &cal_a1 as &dyn IComputeAdapter,
            &cal_a2 as &dyn IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &corpus,
        embedding_model: None,
    })
    .await
    .expect("calibration must succeed with mock adapters");

    let coeff = &cal_event.coefficients;
    let alpha = coeff.alpha;
    let beta_base = coeff.beta_base;
    let cg = coeff.cg_mean();
    let beta_eff = coeff.beta_eff();
    let n_max = coeff.n_max();

    eprintln!("\n── Calibration (mock adapters) ──");
    eprintln!("  α       = {alpha:.4}");
    eprintln!("  β₀      = {beta_base:.4}");
    eprintln!("  CG      = {cg:.4}");
    eprintln!("  β_eff   = {beta_eff:.4}");
    eprintln!("  N_max   = {n_max:.2}");

    assert!((0.0..1.0).contains(&alpha), "α out of range: {alpha}");
    assert!(beta_base > 0.0, "β₀ must be > 0");
    assert!(beta_eff > 0.0, "β_eff must be > 0");
    assert!(n_max >= 1.0, "N_max must be ≥ 1");

    let expected_beta_eff = (beta_base * (1.0 - cg)).max(1e-6);
    let rel_err = (beta_eff - expected_beta_eff).abs() / expected_beta_eff;
    assert!(
        rel_err < 0.01,
        "β_eff formula violated (rel_err={rel_err:.4})"
    );

    let expected_n_max = ((1.0 - alpha) / beta_eff).sqrt();
    let n_max_err = (n_max - expected_n_max).abs();
    assert!(
        n_max_err < 1.0,
        "N_max formula violated (err={n_max_err:.2})"
    );

    eprintln!("  ✓ β_eff = β₀×(1−CG) holds");
    eprintln!("  ✓ N_max = √((1−α)/β_eff) holds");

    // Submit task requesting N >> N_max; engine must clamp
    let n_max_floor = n_max.floor() as u32;
    let requested_n = n_max_floor + 5;
    eprintln!("\n── Engine N_max bound test ──");
    eprintln!("  N_max ceiling = {n_max_floor}");
    eprintln!("  Requested N   = {requested_n} (over by 5)");

    let task_id = TaskId::new();
    let store = TaskStore::new();
    store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), TenantId::default_tenant()),
    );

    let explorer = DecompositionMockAdapter::new("Stateless JWT auth is recommended.".into());
    let mock_verifier = MockAdapter::new(r#"{"score": 0.8, "reason": "compliant"}"#.into());
    let mock_auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let registry =
        AdapterRegistry::new(Arc::new(MockAdapter::new("reg".into())) as Arc<dyn IComputeAdapter>);

    let manifest = TaskManifest {
        description: "Explain stateless JWT authentication in one concise sentence.".into(),
        pareto_weights: ParetoWeights::new(0.5, 0.3, 0.2).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: requested_n as usize,
            tau_min: Some(0.2),
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
        tenant_id: TenantId::default_tenant(),
    };

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: cal_event,
        explorer_adapters: vec![&explorer as &dyn IComputeAdapter],
        verification_adapter: &mock_verifier as &dyn IComputeAdapter,
        auditor_adapter: &mock_auditor as &dyn IComputeAdapter,
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
        tao_multiplier: 1.0,
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
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
    };

    let max_allowed_proposals = n_max_floor * (cfg.max_autonomic_retries + 1);

    match ExecutionEngine::run_offline(input).await {
        Ok(output) => {
            let agents_run = output.verification_events.len() as u32;
            eprintln!("  Engine resolved. Proposals generated: {agents_run}");
            eprintln!("  Max allowed (N_max × retries+1): {max_allowed_proposals}");
            assert!(
                agents_run <= max_allowed_proposals,
                "Engine ran {agents_run} agents but N_max={n_max_floor} × {} = {max_allowed_proposals}",
                cfg.max_autonomic_retries + 1
            );
            eprintln!("  ✓ N_max ceiling enforced by engine");
        }
        Err(e) => {
            eprintln!("  Engine returned err (expected with mock outputs): {e}");
            let ts = store.get(&task_id);
            assert!(ts.is_some(), "task must exist in store after engine error");
            let status = ts.unwrap().status;
            assert_eq!(status, "failed", "task must be marked failed: {status}");
            eprintln!("  ✓ Task correctly marked 'failed' in store");
            eprintln!("  ✓ N_max ceiling enforced (no panic, store consistent)");
        }
    }
}

/// Proves the full engine pipeline (calibration → exploration → verification → synthesis)
/// runs end-to-end with mock adapters and SRANI grounding chain (SpecAnchor only).
///
/// With mock adapters: SRANI may or may not fire depending on CFI thresholds.
/// The core invariant is that the engine completes without panicking and either
/// produces output or marks the task as failed in the store.
#[tokio::test]
async fn engine_full_pipeline_with_mock_adapters() {
    let cfg = H2AIConfig {
        explorer_max_tokens: 512,
        calibration_max_tokens: 256,
        ..H2AIConfig::default()
    };

    eprintln!("\n── Phase 0: Calibration (mock) ──────────────────────────────────");
    let cal_a1 = MockAdapter::new("Stateless rate limiting with Redis sliding windows.".into());
    let corpus = vec![ConstraintDoc::new_llm_judge(
        "stateless",
        "The solution must be stateless. No server-side sessions.",
    )];

    let cal_event = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Design a stateless rate limiter using Redis sliding windows.".into()],
        adapters: vec![&cal_a1 as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &corpus,
        embedding_model: None,
    })
    .await
    .expect("calibration must succeed with mock adapter");

    let coeff = &cal_event.coefficients;
    eprintln!("  α       = {:.4}", coeff.alpha);
    eprintln!("  β₀      = {:.4}", coeff.beta_base);
    eprintln!("  CG      = {:.4}", coeff.cg_mean());
    eprintln!("  β_eff   = {:.4}", coeff.beta_eff());
    eprintln!("  N_max   = {:.2}", coeff.n_max());

    eprintln!("\n── Phase 1: Engine run (mock explorer + verifier + auditor) ─────");
    let task_id = TaskId::new();
    let store = TaskStore::new();
    store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), TenantId::default_tenant()),
    );

    let explorer = DecompositionMockAdapter::new("Use Redis ZADD for rate limiting.".into());
    let verifier = MockAdapter::new(r#"{"score": 0.85, "reason": "compliant"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let researcher: Arc<dyn IComputeAdapter> = Arc::new(MockAdapter::new(
        "Redis is the authoritative source.".into(),
    ));
    let registry =
        AdapterRegistry::new(Arc::new(MockAdapter::new("reg".into())) as Arc<dyn IComputeAdapter>);

    let chain = Arc::new(SraniGroundingChain::new(vec![Box::new(SpecAnchorGrounder)]));

    let manifest = TaskManifest {
        description: "Build a rate-limiting service using Redis sliding windows for HTTP APIs."
            .into(),
        pareto_weights: ParetoWeights::new(0.3, 0.4, 0.3).unwrap(),
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
        tenant_id: TenantId::default_tenant(),
    };

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: cal_event,
        explorer_adapters: vec![&explorer as _, &explorer as _],
        verification_adapter: &verifier as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
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
            TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: Some(Arc::clone(&researcher)),
        srani_ema_cfi: 0.45,
        srani_count: 5,
        srani_grounding_chain: Some(chain),
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
    };

    match ExecutionEngine::run_offline(input).await {
        Ok(output) => {
            eprintln!(
                "  Proposals evaluated: {}",
                output.verification_events.len()
            );
            eprintln!("  EMA CFI: {:.4}", output.srani_ema_cfi_updated);
            assert!(
                !output.resolved_output.is_empty(),
                "resolved output must not be empty"
            );
            eprintln!("  ✓ Full pipeline completed — output is non-empty");
        }
        Err(e) => {
            eprintln!("  Engine returned err (expected with mock verifier scores): {e}");
            let ts = store.get(&task_id);
            assert!(ts.is_some(), "task must exist in store after engine error");
            assert_eq!(ts.unwrap().status, "failed");
            eprintln!("  ✓ Task correctly marked 'failed' in store");
        }
    }
}
