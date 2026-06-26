//! Integration tests for `engine::run_epistemic_stage`.
//!
//! Each test constructs the minimal `EngineOutput` + `EngineInput` slice that
//! `run_epistemic_stage` actually reads, then asserts on the returned
//! `(rendered_text, Option<ProvenanceMap>)` pair.
//!
//! Test design decisions:
//! - `grounding.enabled = false`  → `HeuristicGroundingJudge` (no LLM call needed)
//! - `coherence_check_enabled = false` → `CoherenceChecker` skipped
//! - `recovery_enabled = false`   → `NullResolver`; `closed_ids` stays empty,
//!   so the post-loop grounding re-check and the re-verification step never fire
//! - Adapter in registry is a `mock_adapter` that panics on call — verifies no
//!   accidental LLM calls leak through from the disabled paths

use std::sync::Arc;

use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_orchestrator::attribution::HarnessAttribution;
use h2ai_orchestrator::engine::{EngineInput, EngineOutput};
use h2ai_orchestrator::provenance::{DocumentConfidence, ProvisionConfidence};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_test_utils::{failing_adapter, mock_adapter};
use h2ai_types::adapter::AdapterRegistry;
use h2ai_types::config::{AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig};
use h2ai_types::events::{
    CalibrationCompletedEvent, CheckVerdict, CheckVerdictKind, SelectionResolvedEvent,
    TaskComplexityAssessedEvent, VerificationScoredEvent,
};
use h2ai_types::identity::{ExplorerId, TaskId, TenantId};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use h2ai_types::sizing::{
    CoherencyCoefficients, CoordinationThreshold, MergeStrategy, PredictionBasis, ProbeSkipReason,
    TaskQuadrant,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn dummy_calibration() -> CalibrationCompletedEvent {
    let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.8, 0.85]).unwrap();
    let threshold = CoordinationThreshold::from_calibration(&cc, 0.3);
    CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients: cc,
        coordination_threshold: threshold,
        ensemble: None,
        eigen: None,
        timestamp: Utc::now(),
        pairwise_beta: None,
        cg_mode: Default::default(),
        adapter_families: vec![],
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: Default::default(),
        calibration_source: Default::default(),
        beta_quality: None,
    }
}

fn dummy_manifest(description: &str, context: Option<&str>) -> TaskManifest {
    TaskManifest {
        description: description.into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: None,
            tau_max: None,
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![],
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: context.map(str::to_string),
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: TenantId::default_tenant(),
    }
}

fn dummy_attribution() -> HarnessAttribution {
    HarnessAttribution {
        baseline_quality: 0.7,
        topology_gain: 0.0,
        verification_gain: 0.0,
        tao_gain: 0.0,
        q_confidence: 0.7,
        prediction_basis: PredictionBasis::Heuristic,
        q_measured: None,
        rho_adjusted: 0.0,
        case_b_flag: false,
        synthesis_gain: 0.0,
    }
}

fn dummy_complexity_event(task_id: &TaskId) -> TaskComplexityAssessedEvent {
    TaskComplexityAssessedEvent {
        task_id: task_id.clone(),
        tcc_structural: 0.0,
        tcc_empirical: None,
        tcc_effective: 0.0,
        n_eff_pool: None,
        task_quadrant: TaskQuadrant::Precision,
        probe_skipped: true,
        probe_skip_reason: ProbeSkipReason::None,
        heavy_fraction: 0.0,
        tcc_mismatch: false,
        probe_cost_tokens: 0,
        n_informative_static: 0,
        timestamp: Utc::now(),
    }
}

fn make_engine_output(
    task_id: &TaskId,
    resolved_output: &str,
    pruned_proposals: Vec<(ExplorerId, String)>,
    verification_events: Vec<VerificationScoredEvent>,
) -> EngineOutput {
    EngineOutput {
        task_id: task_id.clone(),
        resolved_output: resolved_output.into(),
        selection_resolved: SelectionResolvedEvent {
            task_id: task_id.clone(),
            valid_proposals: vec![],
            pruned_proposals,
            merge_strategy: MergeStrategy::ScoreOrdered,
            timestamp: Utc::now(),
            merge_elapsed_secs: None,
            n_input_proposals: 0,
            n_failed_proposals: 0,
            merge_selection_mode: None,
        },
        attribution: dummy_attribution(),
        attribution_interval: None,
        verification_events,
        failed_proposals: vec![],
        pruned_events: vec![],
        talagrand: None,
        suggested_next_params: None,
        waste_ratio: 0.0,
        applied_optimizations: vec![],
        topology_retry_events: vec![],
        mode_collapse_count: 0,
        epistemic_yield: None,
        provenance_map: None,
        task_quadrant: Some(TaskQuadrant::Precision),
        complexity_event: dummy_complexity_event(task_id),
        frontier_event: None,
        adapter_correctness: vec![],
        coherence_state: h2ai_orchestrator::coherence::CoherenceState::default(),
        comparison_events: vec![],
        shadow_audit_events: vec![],
        correlated_warnings: vec![],
        researcher_grounding_events: vec![],
        diversity_degraded_event: None,
        oracle_gate_passed: None,
        leader_elected_events: vec![],
        socratic_diagnosis_events: vec![],
        consensus_agreement_rate: None,
        tokens_used: 0,
    }
}

fn passing_verification_event(task_id: &TaskId, text: &str) -> VerificationScoredEvent {
    VerificationScoredEvent {
        task_id: task_id.clone(),
        explorer_id: ExplorerId::new(),
        score: 0.9,
        reason: "compliant".into(),
        passed: true,
        cache_hit: false,
        passed_checks: Some(1),
        total_checks: Some(1),
        score_lower: None,
        score_upper: None,
        per_check_verdicts: vec![CheckVerdict {
            index: 0,
            kind: CheckVerdictKind::Present,
            text: text.into(),
        }],
        timestamp: Utc::now(),
    }
}

fn failing_verification_event(task_id: &TaskId) -> VerificationScoredEvent {
    VerificationScoredEvent {
        task_id: task_id.clone(),
        explorer_id: ExplorerId::new(),
        score: 0.2,
        reason: "non-compliant".into(),
        passed: false,
        cache_hit: false,
        passed_checks: Some(0),
        total_checks: Some(1),
        score_lower: None,
        score_upper: None,
        per_check_verdicts: vec![],
        timestamp: Utc::now(),
    }
}

/// Base `H2AIConfig` for epistemic stage: heuristic grounding, no LLM calls needed.
fn base_cfg() -> H2AIConfig {
    H2AIConfig {
        grounding: h2ai_config::GroundingConfig {
            enabled: false,
            ..Default::default()
        },
        epistemic_quality: h2ai_config::EpistemicQualityConfig {
            enabled: true,
            coherence_check_enabled: false,
            recovery_enabled: false,
            output_mode: "passthrough".into(),
            zero_valid_proposals_policy: "deliver_unverified".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Passthrough mode: output text is returned byte-for-byte unchanged regardless
/// of what provisions or gaps exist.
#[tokio::test]
async fn passthrough_mode_returns_text_unchanged() {
    let cfg = base_cfg();
    let task_id = TaskId::new();
    let text = "stateless token authentication for the API";
    let manifest = dummy_manifest(text, None);
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let cal = dummy_calibration();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    let mut out = make_engine_output(&task_id, text, vec![], vec![]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![],
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
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;

    assert_eq!(
        rendered, text,
        "passthrough must not alter the output string"
    );
    assert!(
        pmap.is_some(),
        "ProvenanceMap must be returned in passthrough mode"
    );
}

/// `output_mode = "clean"` prepends a confidence header block quote before the text.
#[tokio::test]
async fn clean_mode_prepends_confidence_header() {
    let cfg = H2AIConfig {
        grounding: h2ai_config::GroundingConfig {
            enabled: false,
            ..Default::default()
        },
        epistemic_quality: h2ai_config::EpistemicQualityConfig {
            enabled: true,
            coherence_check_enabled: false,
            recovery_enabled: false,
            output_mode: "clean".into(),
            zero_valid_proposals_policy: "deliver_unverified".into(),
            ..Default::default()
        },
        ..Default::default()
    };
    let task_id = TaskId::new();
    let text = "stateless token authentication for the API";
    let manifest = dummy_manifest(text, None);
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    let ev = passing_verification_event(&task_id, "stateless auth");
    let mut out = make_engine_output(&task_id, text, vec![], vec![ev]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;

    assert!(
        rendered.starts_with("> **Epistemic Confidence:"),
        "clean mode must prepend a confidence header; got: {:?}",
        &rendered[..rendered.len().min(80)]
    );
    assert!(
        rendered.contains(text),
        "original text must appear after the header"
    );
    let pmap = pmap.expect("clean mode must return a ProvenanceMap");
    let provisions = pmap.provisions();
    assert_eq!(
        provisions.len(),
        1,
        "one passing verification event → one Verified provision"
    );
    assert_eq!(provisions[0].confidence, ProvisionConfidence::Verified);
}

/// `output_mode = "audit"` produces a header, per-provision annotations, and a footer.
#[tokio::test]
async fn audit_mode_includes_annotations_for_open_gaps() {
    let cfg = H2AIConfig {
        grounding: h2ai_config::GroundingConfig {
            enabled: false,
            ..Default::default()
        },
        epistemic_quality: h2ai_config::EpistemicQualityConfig {
            enabled: true,
            coherence_check_enabled: false,
            recovery_enabled: false,
            output_mode: "audit".into(),
            zero_valid_proposals_policy: "deliver_unverified".into(),
            ..Default::default()
        },
        ..Default::default()
    };
    let task_id = TaskId::new();
    let text = "stateless token authentication for the API";
    let manifest = dummy_manifest(text, None);
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    let pruned = vec![(ExplorerId::new(), "missing latency SLO".into())];
    let mut out = make_engine_output(&task_id, text, pruned, vec![]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;

    assert!(
        rendered.contains("⚠") || rendered.contains("Requires Review"),
        "audit mode must annotate open gaps; got: {:?}",
        &rendered[..rendered.len().min(200)]
    );
    let pmap = pmap.expect("audit mode must return a ProvenanceMap");
    assert_eq!(
        pmap.document_confidence(),
        DocumentConfidence::RequiresReview,
        "one open gap → RequiresReview document confidence"
    );
}

/// Two unique pruned-proposal reasons produce two `RequiresReview` provisions in the pmap.
#[tokio::test]
async fn pruned_proposals_become_requires_review_provisions() {
    let cfg = base_cfg();
    let task_id = TaskId::new();
    let text = "stateless token authentication";
    let manifest = dummy_manifest(text, None);
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    let pruned = vec![
        (ExplorerId::new(), "missing latency SLO".into()),
        (ExplorerId::new(), "missing error budget definition".into()),
    ];
    let mut out = make_engine_output(&task_id, text, pruned, vec![]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (_rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;
    let pmap = pmap.unwrap();

    let requires_review: Vec<_> = pmap
        .provisions()
        .iter()
        .filter(|p| p.confidence == ProvisionConfidence::RequiresReview)
        .collect();
    assert_eq!(
        requires_review.len(),
        2,
        "two distinct pruned reasons → two RequiresReview provisions; got: {:?}",
        pmap.provisions()
            .iter()
            .map(|p| &p.provision_label)
            .collect::<Vec<_>>()
    );
}

/// Duplicate pruned-proposal reasons (same text, different explorers) collapse to one gap.
#[tokio::test]
async fn duplicate_pruned_reasons_collapsed_to_single_gap() {
    let cfg = base_cfg();
    let task_id = TaskId::new();
    let text = "stateless token authentication";
    let manifest = dummy_manifest(text, None);
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    let pruned = vec![
        (ExplorerId::new(), "missing latency SLO".into()),
        (ExplorerId::new(), "missing latency SLO".into()),
    ];
    let mut out = make_engine_output(&task_id, text, pruned, vec![]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (_rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;
    let pmap = pmap.unwrap();

    let requires_review: Vec<_> = pmap
        .provisions()
        .iter()
        .filter(|p| p.confidence == ProvisionConfidence::RequiresReview)
        .collect();
    assert_eq!(
        requires_review.len(),
        1,
        "duplicate pruned reasons → single deduplicated gap"
    );
}

/// `zero_valid_proposals_policy = "fail"` with zero verified provisions
/// returns the original text unchanged and a `None` provenance map.
#[tokio::test]
async fn fail_policy_returns_none_pmap_when_no_verified_provisions() {
    let cfg = H2AIConfig {
        grounding: h2ai_config::GroundingConfig {
            enabled: false,
            ..Default::default()
        },
        epistemic_quality: h2ai_config::EpistemicQualityConfig {
            enabled: true,
            coherence_check_enabled: false,
            recovery_enabled: false,
            output_mode: "passthrough".into(),
            zero_valid_proposals_policy: "fail".into(),
            ..Default::default()
        },
        ..Default::default()
    };
    let task_id = TaskId::new();
    let text = "stateless token authentication";
    let manifest = dummy_manifest(text, None);
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    // No passing verification events → zero verified provisions
    let failing_ev = failing_verification_event(&task_id);
    let mut out = make_engine_output(&task_id, text, vec![], vec![failing_ev]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;

    assert_eq!(
        rendered, text,
        "fail policy: original text must be returned unmodified"
    );
    assert!(
        pmap.is_none(),
        "fail policy with zero verified provisions must return None ProvenanceMap"
    );
}

/// Passing `VerificationScoredEvent` entries create `Verified` provisions in the pmap.
/// With only Verified provisions, document confidence resolves to `High`.
#[tokio::test]
async fn passing_verification_events_produce_verified_provisions() {
    let cfg = base_cfg();
    let task_id = TaskId::new();
    let text = "stateless token authentication";
    let manifest = dummy_manifest(text, None);
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    let ev1 = passing_verification_event(&task_id, "stateless auth");
    let ev2 = passing_verification_event(&task_id, "token-based credentials");
    let mut out = make_engine_output(&task_id, text, vec![], vec![ev1, ev2]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (_rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;
    let pmap = pmap.expect("ProvenanceMap must be returned when verified provisions exist");

    let verified: Vec<_> = pmap
        .provisions()
        .iter()
        .filter(|p| p.confidence == ProvisionConfidence::Verified)
        .collect();
    assert_eq!(
        verified.len(),
        2,
        "two passing events → two Verified provisions"
    );
    assert_eq!(
        pmap.document_confidence(),
        DocumentConfidence::High,
        "all-Verified → High document confidence"
    );
}

/// Heuristic grounding judge (grounding.enabled=false) flags tech nouns in the output
/// that are absent from the effective spec as UngroundedContent gaps.
#[tokio::test]
async fn heuristic_grounding_detects_ungrounded_tech_noun() {
    let cfg = base_cfg();
    let task_id = TaskId::new();
    let spec = "stateless authentication with tokens";
    // "Kafka" appears in the LEXICON but is absent from the spec → should be flagged
    let output_with_ungrounded = "Kafka-backed stateless authentication with tokens";
    let manifest = dummy_manifest(spec, None);
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    let mut out = make_engine_output(&task_id, output_with_ungrounded, vec![], vec![]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (_rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;
    let pmap = pmap.unwrap();

    let grounding_gaps: Vec<_> = pmap
        .provisions()
        .iter()
        .filter(|p| p.confidence == ProvisionConfidence::RequiresReview)
        .collect();
    assert!(
        !grounding_gaps.is_empty(),
        "heuristic judge must produce at least one RequiresReview gap for 'Kafka'"
    );
    let has_kafka_gap = grounding_gaps
        .iter()
        .any(|p| p.provision_label.to_lowercase().contains("kafka"));
    assert!(
        has_kafka_gap,
        "one gap must reference 'Kafka' (the ungrounded noun); labels: {:?}",
        grounding_gaps
            .iter()
            .map(|p| &p.provision_label)
            .collect::<Vec<_>>()
    );
}

/// `seed_uncertainty_gaps` fires on keywords in the task context, injecting
/// `UncertainDomain` gaps that always remain as `RequiresReview` provisions.
#[tokio::test]
async fn uncertainty_context_keyword_produces_uncertain_domain_gap() {
    let cfg = base_cfg();
    let task_id = TaskId::new();
    let spec = "stateless authentication";
    // "unsettled" is a known UNCERTAINTY_KEYWORDS entry
    let manifest = dummy_manifest(
        spec,
        Some("The regulatory landscape is unsettled and evolving"),
    );
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    let mut out = make_engine_output(&task_id, spec, vec![], vec![]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (_rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;
    let pmap = pmap.unwrap();

    let uncertainty_gaps: Vec<_> = pmap
        .provisions()
        .iter()
        .filter(|p| {
            p.confidence == ProvisionConfidence::RequiresReview
                && p.provision_label.contains("uncertain")
        })
        .collect();
    assert!(
        !uncertainty_gaps.is_empty(),
        "task context with 'unsettled' must produce at least one UncertainDomain RequiresReview provision; provisions: {:?}",
        pmap.provisions().iter().map(|p| (&p.provision_label, &p.confidence)).collect::<Vec<_>>()
    );
}

/// Constraint corpus is forwarded to the effective spec; when the output references
/// a term that appears in the corpus but not the description, the heuristic judge
/// grounds it via the expanded spec (description + corpus) and does NOT flag it.
#[tokio::test]
async fn constraint_corpus_extends_effective_spec() {
    let cfg = base_cfg();
    let task_id = TaskId::new();
    let spec = "stateless authentication";
    // Redis appears in LEXICON; if present in constraint corpus it must be in effective_spec
    let manifest = dummy_manifest(spec, None);
    let corpus = vec![ConstraintDoc::new_with_description(
        "ARCH-001",
        "The solution must use Redis for session caching if caching is needed",
    )];
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    // Output mentions Redis — should be grounded via corpus, so no gap
    let output = "stateless authentication with Redis-backed session caching";
    let mut out = make_engine_output(&task_id, output, vec![], vec![]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
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
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (_rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;
    let pmap = pmap.unwrap();

    let redis_gap = pmap.provisions().iter().any(|p| {
        p.confidence == ProvisionConfidence::RequiresReview
            && p.provision_label.to_lowercase().contains("redis")
    });
    assert!(
        !redis_gap,
        "Redis appears in constraint corpus → effective spec → must NOT produce an ungrounded gap"
    );
}

/// Mixed provisions: one Verified + one RequiresReview → worst-wins → RequiresReview
/// document confidence. Verifies the worst-wins rule across provision types.
#[tokio::test]
async fn worst_wins_with_mixed_verified_and_requires_review() {
    let cfg = base_cfg();
    let task_id = TaskId::new();
    let spec = "stateless authentication";
    let manifest = dummy_manifest(spec, None);
    let registry = AdapterRegistry::new(Arc::new(failing_adapter()) as _);
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let store = TaskStore::new();
    let tao_estimator = Arc::new(tokio::sync::RwLock::new(
        h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
    ));
    // One passing verification event (Verified) + one pruned proposal (RequiresReview)
    let ev = passing_verification_event(&task_id, "stateless auth");
    let pruned = vec![(ExplorerId::new(), "missing latency SLO".into())];
    let mut out = make_engine_output(&task_id, spec, pruned, vec![ev]);

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: dummy_calibration(),
        explorer_adapters: vec![],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator,
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: vec![],
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        induction_scheduler: None,
        conformal_margin: 0.0,
    };

    let (_rendered, pmap) = h2ai_orchestrator::engine::run_epistemic_stage(&mut out, &input).await;
    let pmap = pmap.unwrap();

    let verified_count = pmap
        .provisions()
        .iter()
        .filter(|p| p.confidence == ProvisionConfidence::Verified)
        .count();
    assert_eq!(
        verified_count, 1,
        "one passing event → one Verified provision"
    );

    let req_review_count = pmap
        .provisions()
        .iter()
        .filter(|p| p.confidence == ProvisionConfidence::RequiresReview)
        .count();
    assert!(
        req_review_count >= 1,
        "one pruned proposal → at least one RequiresReview provision"
    );

    assert_eq!(
        pmap.document_confidence(),
        DocumentConfidence::RequiresReview,
        "worst-wins: RequiresReview dominates over Verified at document level"
    );
}
