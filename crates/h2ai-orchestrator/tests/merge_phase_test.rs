#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::wildcard_imports
)]
//! Unit tests for `h2ai_orchestrator::phases::merge` and the Talagrand
//! diagnostic helpers it exercises.
//!
//! All tests are NATS-free. The `run()` function is exercised via the
//! ZeroSurvival path using an empty `ProposalSet` — no live adapters or
//! NATS connection required.

use std::sync::Arc;

use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_orchestrator::{
    coherence::CoherenceState,
    diagnostics::{CalibrationState, TalagrandDiagnostic},
    engine::EngineInput,
    phases::{self, ExitReason, StepResult},
    self_optimizer::OptimizerParams,
    tao_loop::TaoMultiplierEstimator,
    task_store::TaskStore,
};
use h2ai_state::semilattice::ProposalSet;
use h2ai_test_utils::mock_adapter;
use h2ai_types::{
    adapter::{AdapterRegistry, IComputeAdapter},
    config::{AdapterKind, AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig},
    events::{CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode},
    identity::{TaskId, TenantId},
    manifest::{ExplorerRequest, TaskManifest, TopologyRequest},
    sizing::{CoherencyCoefficients, CoordinationThreshold, MergeStrategy, PredictionBasis},
};

// ── Shared helpers ────────────────────────────────────────────────────────────

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
        description: "Stub task for merge phase tests".into(),
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
        induction_scheduler: None,
        conformal_margin: 0.0,
    }
}

fn default_optimizer_params() -> OptimizerParams {
    OptimizerParams {
        n_agents: 2,
        max_turns: 3,
        verify_threshold: 0.5,
    }
}

// ── Talagrand helpers (used inside merge.rs) ──────────────────────────────────

#[test]
fn talagrand_tau_kl_next_clamps_to_min_max() {
    // Build a minimal diagnostic with an artificial rank histogram.
    let diag = TalagrandDiagnostic {
        rank_histogram: vec![0, 5, 5], // 2 adapters, balanced
        chi_sq_from_uniform: 0.0,
        spread_error_ratio: 1.0,
        calibration_state: CalibrationState::Calibrated,
    };
    let result = diag.tau_kl_next(
        0.5,  // current_factor
        0.01, // eta
        0.1,  // tau_min
        2.0,  // tau_max
    );
    assert!(result >= 0.1, "should be ≥ tau_min, got {result}");
    assert!(result <= 2.0, "should be ≤ tau_max, got {result}");
}

#[test]
fn talagrand_tau_expansion_over_confident_expands_factor() {
    let diag = TalagrandDiagnostic {
        rank_histogram: vec![0, 25, 5], // U-shape: tail rank 1 over-represented
        chi_sq_from_uniform: 20.0,
        spread_error_ratio: 0.5,
        calibration_state: CalibrationState::OverConfident,
    };
    let expanded = diag.tau_expansion_factor(1.0, 3.0);
    assert!(expanded > 1.0, "OverConfident should expand factor");
    assert!(expanded <= 3.0, "should not exceed max_factor");
}

#[test]
fn talagrand_tau_expansion_calibrated_unchanged() {
    let diag = TalagrandDiagnostic {
        rank_histogram: vec![0, 10, 10],
        chi_sq_from_uniform: 0.5,
        spread_error_ratio: 1.0,
        calibration_state: CalibrationState::Calibrated,
    };
    let factor = diag.tau_expansion_factor(1.5, 3.0);
    assert!(
        (factor - 1.5).abs() < 1e-9,
        "Calibrated state should leave factor unchanged"
    );
}

#[test]
fn talagrand_tau_expansion_insufficient_unchanged() {
    let diag = TalagrandDiagnostic {
        rank_histogram: vec![0, 3, 2],
        chi_sq_from_uniform: 1.0,
        spread_error_ratio: 1.0,
        calibration_state: CalibrationState::Insufficient,
    };
    let factor = diag.tau_expansion_factor(1.2, 3.0);
    assert!(
        (factor - 1.2).abs() < 1e-9,
        "Insufficient state should leave factor unchanged"
    );
}

#[test]
fn talagrand_tau_expansion_under_dispersed_unchanged() {
    let diag = TalagrandDiagnostic {
        rank_histogram: vec![0, 2, 18], // Λ-shape: middle ranks over-represented
        chi_sq_from_uniform: 15.0,
        spread_error_ratio: 2.0,
        calibration_state: CalibrationState::UnderDispersed,
    };
    let factor = diag.tau_expansion_factor(1.0, 3.0);
    assert!(
        (factor - 1.0).abs() < 1e-9,
        "UnderDispersed should leave factor unchanged (returns current_factor)"
    );
}

// ── phases::merge::run() — ZeroSurvival path (empty ProposalSet) ──────────────

#[tokio::test]
async fn merge_phase_zero_survival_on_empty_proposal_set() {
    let adapter = mock_adapter("output");
    let cfg = H2AIConfig::default();
    let store = TaskStore::new();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter("mock")) as Arc<dyn IComputeAdapter>);

    let engine_input = make_engine_input(&adapter, &adapter, &cfg, store, &registry);

    let proposal_set = ProposalSet::new(); // empty → ZeroSurvival
    let params = default_optimizer_params();
    let coherence = CoherenceState::default();

    let (result, tau_expansion) = phases::merge::run(
        proposal_set,
        vec![], // no pruned branches
        0.0,    // synthesis_gain
        vec![], // synthesis_comparison_events
        phases::merge::Input {
            engine_input: &engine_input,
            task_id: &engine_input.task_id,
            retry_count: 0,
            explorer_count: 2,
            filter_ratio: 1.0,
            p_mean: 0.6,
            rho_mean: 0.1,
            tao_turns_mean: 1.0,
            attribution_basis: PredictionBasis::Heuristic,
            tau_values: vec![0.5, 0.7],
            all_raw_texts_this_wave: vec!["text1".into(), "text2".into()],
            surviving_texts: vec![],
            iteration_verification_events: &[],
            frontier_event: &None,
            adapter_correctness: vec![],
            oracle_gate_passed: None,
            wave_coherence: &coherence,
            quality_history: &[],
            n_max_ceiling: 4,
            cg_mean: 0.65,
            current_params: &params,
            verification_config: VerificationConfig::default(),
            assessed_quadrant: h2ai_types::sizing::TaskQuadrant::Precision,
            all_pruned: &[],
            synthesis_candidates_len: 0,
            provisioned_merge_strategy: MergeStrategy::ConsensusMedian,
            wave_violations: vec![],
            retry_accumulator: None,
            osp_config: None,
            current_tau_spread_factor: 1.0,
        },
    )
    .await;

    // Empty proposal set must produce ZeroSurvival early exit.
    assert!(
        matches!(
            result,
            StepResult::EarlyExit(ExitReason::ZeroSurvival { .. })
        ),
        "expected ZeroSurvival early exit for empty proposal set"
    );

    // tau_expansion is always computed regardless of outcome — even with no
    // verification events it returns None (insufficient data).
    assert!(
        tau_expansion.is_none(),
        "no verification events → tau_expansion should be None"
    );
}

#[tokio::test]
async fn merge_phase_zero_survival_carries_tau_values() {
    let adapter = mock_adapter("output");
    let cfg = H2AIConfig::default();
    let store = TaskStore::new();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter("mock")) as Arc<dyn IComputeAdapter>);

    let engine_input = make_engine_input(&adapter, &adapter, &cfg, store, &registry);
    let proposal_set = ProposalSet::new();
    let params = default_optimizer_params();
    let coherence = CoherenceState::default();
    let tau_values = vec![0.3, 0.6, 0.9];

    let (result, _tau_expansion) = phases::merge::run(
        proposal_set,
        vec![],
        0.0,
        vec![],
        phases::merge::Input {
            engine_input: &engine_input,
            task_id: &engine_input.task_id,
            retry_count: 1,
            explorer_count: 3,
            filter_ratio: 0.0,
            p_mean: 0.5,
            rho_mean: 0.2,
            tao_turns_mean: 1.5,
            attribution_basis: PredictionBasis::Heuristic,
            tau_values: tau_values.clone(),
            all_raw_texts_this_wave: vec![],
            surviving_texts: vec![],
            iteration_verification_events: &[],
            frontier_event: &None,
            adapter_correctness: vec![],
            oracle_gate_passed: None,
            wave_coherence: &coherence,
            quality_history: &[],
            n_max_ceiling: 6,
            cg_mean: 0.6,
            current_params: &params,
            verification_config: VerificationConfig::default(),
            assessed_quadrant: h2ai_types::sizing::TaskQuadrant::Precision,
            all_pruned: &[],
            synthesis_candidates_len: 0,
            provisioned_merge_strategy: MergeStrategy::ScoreOrdered,
            wave_violations: vec![],
            retry_accumulator: None,
            osp_config: None,
            current_tau_spread_factor: 1.0,
        },
    )
    .await;

    // The ZeroSurvival exit reason should carry back the tau_values.
    if let StepResult::EarlyExit(ExitReason::ZeroSurvival {
        tau_values: carried,
        ..
    }) = result
    {
        assert_eq!(
            carried, tau_values,
            "tau_values must be forwarded in ZeroSurvival exit reason"
        );
    } else {
        panic!("expected ZeroSurvival");
    }
}

#[tokio::test]
async fn merge_phase_zero_survival_filter_ratio_forwarded() {
    let adapter = mock_adapter("output");
    let cfg = H2AIConfig::default();
    let store = TaskStore::new();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter("mock")) as Arc<dyn IComputeAdapter>);

    let engine_input = make_engine_input(&adapter, &adapter, &cfg, store, &registry);
    let proposal_set = ProposalSet::new();
    let params = default_optimizer_params();
    let coherence = CoherenceState::default();

    let (result, _tau) = phases::merge::run(
        proposal_set,
        vec![],
        0.0,
        vec![],
        phases::merge::Input {
            engine_input: &engine_input,
            task_id: &engine_input.task_id,
            retry_count: 0,
            explorer_count: 1,
            filter_ratio: 0.33,
            p_mean: 0.5,
            rho_mean: 0.0,
            tao_turns_mean: 1.0,
            attribution_basis: PredictionBasis::Heuristic,
            tau_values: vec![],
            all_raw_texts_this_wave: vec![],
            surviving_texts: vec![],
            iteration_verification_events: &[],
            frontier_event: &None,
            adapter_correctness: vec![],
            oracle_gate_passed: None,
            wave_coherence: &coherence,
            quality_history: &[],
            n_max_ceiling: 2,
            cg_mean: 0.5,
            current_params: &params,
            verification_config: VerificationConfig::default(),
            assessed_quadrant: h2ai_types::sizing::TaskQuadrant::Precision,
            all_pruned: &[],
            synthesis_candidates_len: 0,
            provisioned_merge_strategy: MergeStrategy::ConsensusMedian,
            wave_violations: vec![],
            retry_accumulator: None,
            osp_config: None,
            current_tau_spread_factor: 1.0,
        },
    )
    .await;

    if let StepResult::EarlyExit(ExitReason::ZeroSurvival { filter_ratio, .. }) = result {
        assert!(
            (filter_ratio - 0.33).abs() < 1e-9,
            "filter_ratio must be forwarded in ZeroSurvival: got {filter_ratio}"
        );
    } else {
        panic!("expected ZeroSurvival exit");
    }
}
