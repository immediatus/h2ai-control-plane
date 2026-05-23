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
use async_trait::async_trait;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_test_utils::MockAdapter;

use h2ai_constraints::types::ConstraintDoc;
use h2ai_orchestrator::engine::{EngineError, EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::{TaskPhase, TaskStore};
use h2ai_types::adapter::{
    AdapterError, AdapterRegistry, ComputeRequest, ComputeResponse, IComputeAdapter,
};
use h2ai_types::config::{
    AdapterKind, AgentRole, AuditorConfig, ParetoWeights, RoleSpec, TaoConfig, VerificationConfig,
};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use h2ai_types::sizing::MergeStrategy;
use std::sync::Arc;

// ── Helper adapters ──────────────────────────────────────────────────────────

#[derive(Debug)]
struct FailingAdapter {
    kind: AdapterKind,
}

impl FailingAdapter {
    fn new() -> Self {
        Self {
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://failing".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for FailingAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Err(AdapterError::NetworkError("simulated agent loss".into()))
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

#[derive(Debug)]
struct TokenCostAdapter {
    output: String,
    cost: u64,
    kind: AdapterKind,
}

impl TokenCostAdapter {
    fn new(output: impl Into<String>, cost: u64) -> Self {
        Self {
            output: output.into(),
            cost,
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://token-cost".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for TokenCostAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Ok(ComputeResponse {
            output: self.output.clone(),
            token_cost: self.cost,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
            reasoning_trace: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

// ── Shared helpers ───────────────────────────────────────────────────────────

fn constraint_corpus() -> Vec<ConstraintDoc> {
    vec![ConstraintDoc::new_llm_judge(
        "ADR-001",
        "The solution must be stateless. No server-side sessions or shared mutable state permitted.",
    )]
}

async fn run_calibration(
    adapters: &[&dyn IComputeAdapter],
) -> h2ai_types::events::CalibrationCompletedEvent {
    let cfg = H2AIConfig::default();
    CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Calibrate".into(), "Second".into()],
        adapters: adapters.to_vec(),
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    })
    .await
    .unwrap()
}

fn default_manifest(count: usize) -> TaskManifest {
    TaskManifest {
        description: "Propose stateless auth solution with ADR-001 compliance".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count,
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
    }
}

fn auditor_cfg() -> AuditorConfig {
    AuditorConfig {
        adapter: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "NONE".into(),
            model: None,
        },
        ..Default::default()
    }
}

fn scoring_adapter() -> MockAdapter {
    MockAdapter::new(r#"{"score": 0.9, "reason": "compliant"}"#.into())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn system_solves_well_formed_problem() {
    let ex1 = MockAdapter::new("stateless JWT auth solution — ADR-001 compliant".into());
    let ex2 = MockAdapter::new("token-based stateless authentication RSA signing ADR-001".into());
    let ex3 = MockAdapter::new("stateless auth credential verification OAuth2 ADR-001".into());
    let scorer = scoring_adapter();
    let auditor =
        MockAdapter::new(r#"{"approved": true, "reason": "compliant with ADR-001"}"#.into());
    let cal = run_calibration(&[&ex1 as &dyn IComputeAdapter]).await;
    let store = TaskStore::new();
    // Disable C1 — this test predates correlated-hallucination detection and uses
    // similar auth proposals intentionally; dedicated c1_ tests cover that path.
    let cfg = H2AIConfig {
        correlated_hallucination_cv_threshold: 0.0,
        ..Default::default()
    };
    let task_id = TaskId::new();

    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new(
        "stateless JWT auth solution — ADR-001 compliant".into(),
    )) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: task_id.clone(),
        manifest: default_manifest(3),
        calibration: cal,
        explorer_adapters: vec![
            &ex1 as &dyn IComputeAdapter,
            &ex2 as &dyn IComputeAdapter,
            &ex3 as &dyn IComputeAdapter,
        ],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: constraint_corpus(),
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
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
        conformal_margin: 0.0,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine returned error: {:?}", result.err());

    let out = result.unwrap();
    store.mark_resolved(&out.task_id);
    let state = store.get(&out.task_id).unwrap();

    assert!(
        state.proposals_valid >= 6,
        "expected ≥6 validation events (3 explorers × 2 gates), got {}",
        state.proposals_valid
    );
    assert_eq!(state.proposals_pruned, 0);
    assert_eq!(state.explorers_completed, 3);
    assert!(out.attribution.baseline_quality > 0.0);
    assert!(out.attribution.q_confidence >= out.attribution.baseline_quality);
    assert!(out.attribution.q_confidence <= 1.0);
    assert_eq!(out.selection_resolved.valid_proposals.len(), 3);
    assert!(out.selection_resolved.pruned_proposals.is_empty());
    assert_eq!(
        state.status, "resolved",
        "task should reach resolved status after successful merge"
    );
}

#[tokio::test]
async fn system_detects_hallucinating_proposals_and_exhausts_retries() {
    // Two adapters with different enough text so the diversity gate does not fire,
    // allowing proposals to reach the auditor which rejects all of them.
    let ex1 = MockAdapter::new("stateless JWT auth implementation ADR-001".into());
    let ex2 = MockAdapter::new("invented credential protocol fabricated library ADR-001".into());
    let scorer = scoring_adapter();
    // Auditor rejects every proposal — simulates hallucination detected at semantic review (Phase 4).
    // Verification (Phase 3.5) passes since proposals are syntactically well-formed.
    let auditor = MockAdapter::new(
        r#"{"approved": false, "reason": "hallucination detected: output fabricated"}"#.into(),
    );
    let cal = run_calibration(&[&ex1 as &dyn IComputeAdapter]).await;
    let store = TaskStore::new();
    let cfg = H2AIConfig {
        max_autonomic_retries: 2,
        ..H2AIConfig::default()
    };
    let task_id = TaskId::new();

    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new(
        "stateless JWT auth implementation ADR-001".into(),
    )) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: task_id.clone(),
        manifest: default_manifest(2),
        calibration: cal,
        explorer_adapters: vec![&ex1 as &dyn IComputeAdapter, &ex2 as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: constraint_corpus(),
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
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
        conformal_margin: 0.0,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), EngineError::MaxRetriesExhausted { .. }),
        "expected MaxRetriesExhausted"
    );

    let state = store.get(&task_id).unwrap();
    assert_eq!(state.status, "failed");
    assert_eq!(TaskPhase::try_from(state.phase).unwrap(), TaskPhase::Failed);
    assert!(state.proposals_pruned >= 2);
    assert_eq!(
        state.autonomic_retries, 2,
        "full retry budget (max_autonomic_retries=2) should be exhausted"
    );
}

#[tokio::test]
async fn system_survives_agent_loss_and_resolves_with_survivors() {
    let failing = FailingAdapter::new();
    let survivor = MockAdapter::new("stateless JWT auth — ADR-001 compliant".into());
    let scorer = scoring_adapter();
    let auditor =
        MockAdapter::new(r#"{"approved": true, "reason": "compliant with ADR-001"}"#.into());
    let cal = run_calibration(&[&survivor as &dyn IComputeAdapter]).await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let task_id = TaskId::new();

    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new(
        "stateless JWT auth — ADR-001 compliant".into(),
    )) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: task_id.clone(),
        manifest: default_manifest(3),
        calibration: cal,
        // 2 adapters for 3 explorers → round-robin: exp0→failing, exp1→survivor, exp2→failing
        explorer_adapters: vec![
            &failing as &dyn IComputeAdapter,
            &survivor as &dyn IComputeAdapter,
        ],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: constraint_corpus(),
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
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
        conformal_margin: 0.0,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine returned error: {:?}", result.err());

    let out = result.unwrap();
    store.mark_resolved(&out.task_id);
    let state = store.get(&out.task_id).unwrap();

    assert_eq!(state.explorers_completed, 3);
    assert!(state.proposals_valid >= 1);
    assert!(!out.resolved_output.is_empty());
    assert!(!out.selection_resolved.valid_proposals.is_empty());
    assert_eq!(state.status, "resolved");
}

#[tokio::test]
async fn system_resolves_conflict_via_bft_consensus() {
    let cheap = TokenCostAdapter::new("low-cost auth solution — stateless ADR-001", 5);
    let pricey = TokenCostAdapter::new("high-cost auth solution — stateless ADR-001", 500);
    let scorer = scoring_adapter();
    let fair_evaluator =
        MockAdapter::new(r#"{"approved": true, "reason": "valid compliant proposal"}"#.into());
    let cal = run_calibration(&[
        &cheap as &dyn IComputeAdapter,
        &pricey as &dyn IComputeAdapter,
    ])
    .await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let task_id = TaskId::new();

    let manifest = TaskManifest {
        description: "Propose stateless auth solution with ADR-001 compliance".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 2,
            tau_min: None,
            tau_max: None,
            roles: vec![
                RoleSpec {
                    agent_id: "exp_cheap".into(),
                    role: AgentRole::Evaluator,
                    tau: None,
                    role_error_cost: None,
                },
                RoleSpec {
                    agent_id: "exp_pricey".into(),
                    role: AgentRole::Evaluator,
                    tau: None,
                    role_error_cost: None,
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

    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new(
        "low-cost auth solution — stateless ADR-001".into(),
    )) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![
            &cheap as &dyn IComputeAdapter,
            &pricey as &dyn IComputeAdapter,
        ],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &fair_evaluator as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: constraint_corpus(),
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
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
        conformal_margin: 0.0,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine returned error: {:?}", result.err());

    let out = result.unwrap();
    assert_eq!(
        out.selection_resolved.merge_strategy,
        MergeStrategy::ConsensusMedian
    );
    assert_eq!(out.selection_resolved.valid_proposals.len(), 2);
    // Condorcet picks the most consensus proposal; with 2 similar proposals either is valid
    assert!(
        out.resolved_output == "low-cost auth solution — stateless ADR-001"
            || out.resolved_output == "high-cost auth solution — stateless ADR-001",
        "expected one of the two valid proposals, got: {}",
        out.resolved_output
    );
}

// ── Shadow audit system integration tests ───────────────────────────────────

#[derive(Debug)]
struct ShadowAdapter {
    approved: bool,
    kind: AdapterKind,
}

impl ShadowAdapter {
    fn approves() -> Self {
        Self {
            approved: true,
            kind: AdapterKind::Anthropic {
                api_key_env: "NONE".into(),
                model: "claude-shadow".into(),
            },
        }
    }
    fn rejects() -> Self {
        Self {
            approved: false,
            kind: AdapterKind::Anthropic {
                api_key_env: "NONE".into(),
                model: "claude-shadow".into(),
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for ShadowAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let body = if self.approved {
            r#"{"approved": true, "reason": "shadow ok"}"#
        } else {
            r#"{"approved": false, "reason": "shadow rejected"}"#
        };
        Ok(ComputeResponse {
            output: body.into(),
            token_cost: 1,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
            reasoning_trace: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

/// Shadow mode on + both agree → task resolves, shadow_audit_events populated, disagreement=false.
#[tokio::test]
async fn system_shadow_mode_agreement_resolves_task() {
    let explorer = MockAdapter::new("stateless JWT auth — ADR-001 compliant".into());
    let scorer = scoring_adapter();
    let primary_auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let shadow = ShadowAdapter::approves();
    let cal = run_calibration(&[&explorer as &dyn IComputeAdapter]).await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new(
        "stateless JWT auth — ADR-001 compliant".into(),
    )) as Arc<dyn IComputeAdapter>);

    let mut manifest = default_manifest(1);
    manifest.constraint_tags = vec!["security".into()];

    let ctx = h2ai_orchestrator::engine::ShadowAuditCtx {
        adapter: Arc::new(shadow) as Arc<dyn IComputeAdapter>,
        promoted_domains: Default::default(),
        strict: false,
    };

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &primary_auditor as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: constraint_corpus(),
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: std::sync::Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: Some(ctx),
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
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(!output.resolved_output.is_empty(), "task must resolve");
    assert!(
        !output.shadow_audit_events.is_empty(),
        "shadow audit events must be populated"
    );
    assert!(
        output.shadow_audit_events.iter().all(|e| !e.disagreement),
        "all events must have disagreement=false when both auditors agree"
    );
}

/// Shadow mode on + shadow rejects (shadow mode, not AND-vote) → task still resolves,
/// disagreement=true in events, but primary's approve wins.
#[tokio::test]
async fn system_shadow_disagreement_does_not_affect_shadow_mode_result() {
    let explorer = MockAdapter::new("stateless JWT auth — ADR-001 compliant".into());
    let scorer = scoring_adapter();
    let primary_auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let shadow = ShadowAdapter::rejects();
    let cal = run_calibration(&[&explorer as &dyn IComputeAdapter]).await;
    let store = TaskStore::new();
    let cfg = H2AIConfig::default();
    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new(
        "stateless JWT auth — ADR-001 compliant".into(),
    )) as Arc<dyn IComputeAdapter>);

    // promoted_domains is empty → shadow observe mode, NOT AND-vote
    let ctx = h2ai_orchestrator::engine::ShadowAuditCtx {
        adapter: Arc::new(shadow) as Arc<dyn IComputeAdapter>,
        promoted_domains: Default::default(),
        strict: false,
    };

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest: default_manifest(1),
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &primary_auditor as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: constraint_corpus(),
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: std::sync::Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: Some(ctx),
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
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        !output.resolved_output.is_empty(),
        "task must resolve when not in AND-vote mode"
    );
    assert!(
        output.shadow_audit_events.iter().any(|e| e.disagreement),
        "at least one event must record disagreement=true"
    );
}

/// AND-vote mode active: primary approves, shadow rejects → task must fail.
#[tokio::test]
async fn system_and_vote_mode_rejects_when_shadow_disagrees() {
    let explorer = MockAdapter::new("stateless JWT auth — ADR-001 compliant".into());
    let scorer = scoring_adapter();
    let primary_auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());
    let shadow = ShadowAdapter::rejects();
    let cal = run_calibration(&[&explorer as &dyn IComputeAdapter]).await;
    let store = TaskStore::new();
    let cfg = H2AIConfig {
        max_autonomic_retries: 0,
        ..H2AIConfig::default()
    };
    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new(
        "stateless JWT auth — ADR-001 compliant".into(),
    )) as Arc<dyn IComputeAdapter>);

    let mut manifest = default_manifest(1);
    manifest.constraint_tags = vec!["security".into()];

    // "security" domain is in promoted_domains → AND-vote active
    let mut promoted = std::collections::HashSet::new();
    promoted.insert("security".to_string());
    let ctx = h2ai_orchestrator::engine::ShadowAuditCtx {
        adapter: Arc::new(shadow) as Arc<dyn IComputeAdapter>,
        promoted_domains: promoted,
        strict: false,
    };

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&explorer as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &primary_auditor as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: constraint_corpus(),
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: std::sync::Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: Some(ctx),
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
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(
        result.is_err(),
        "AND-vote must fail when shadow rejects and retries=0"
    );
}

// ── GAP-C1 system test ───────────────────────────────────────────────────────

#[derive(Debug)]
struct IdenticalOutputAdapter {
    output: String,
    kind: AdapterKind,
}

impl IdenticalOutputAdapter {
    fn new(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://identical".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for IdenticalOutputAdapter {
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

#[tokio::test]
async fn system_c1_fires_and_records_warning_for_identical_proposals() {
    let text = "stateless JWT auth token validation bearer scheme ADR-001 compliant".to_string();
    let ex1 = IdenticalOutputAdapter::new(text.clone());
    let ex2 = IdenticalOutputAdapter::new(text.clone());
    let scorer = scoring_adapter();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());

    let cfg = H2AIConfig {
        correlated_hallucination_cv_threshold: 0.30,
        max_autonomic_retries: 1,
        ..Default::default()
    };

    let cal = run_calibration(&[&ex1 as &dyn IComputeAdapter]).await;
    let store = h2ai_orchestrator::task_store::TaskStore::new();
    let task_id = TaskId::new();
    let registry =
        AdapterRegistry::new(Arc::new(MockAdapter::new(text.clone())) as Arc<dyn IComputeAdapter>);
    let mut manifest = default_manifest(2);
    manifest.explorers.count = 2;

    let input = EngineInput {
        task_id: task_id.clone(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&ex1 as &dyn IComputeAdapter, &ex2 as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: constraint_corpus(),
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 1.0,
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
        conformal_margin: 0.0,
    };

    let result = ExecutionEngine::run_offline(input).await;
    match result {
        Ok(output) => {
            assert!(
                !output.correlated_warnings.is_empty(),
                "C1 warning must fire for identical proposals"
            );
            let warn = &output.correlated_warnings[0];
            assert_eq!(warn.cv, 0.0, "cv must be 0 for identical proposals");
            assert_eq!(warn.mean_jaccard_distance, 0.0);
        }
        Err(EngineError::MaxRetriesExhausted { .. }) => {
            // Acceptable: retries exhausted after repeated C1 detection
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

// ── GAP-C3 system test ───────────────────────────────────────────────────────

#[tokio::test]
async fn system_c3_no_degraded_event_when_domains_covered() {
    let ex = MockAdapter::new("stateless JWT auth solution ADR-001 compliant security".into());
    let scorer = scoring_adapter();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());

    let cfg = H2AIConfig {
        domain_coverage_threshold: 0.5,
        ..Default::default()
    };

    let mut doc = ConstraintDoc::new_llm_judge("SEC-001", "The solution must use stateless auth.");
    doc.domains = vec!["security".into()];
    let corpus = vec![doc];

    let cal = run_calibration(&[&ex as &dyn IComputeAdapter]).await;
    let store = h2ai_orchestrator::task_store::TaskStore::new();
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("solution".into())) as Arc<dyn IComputeAdapter>
    );

    let mut manifest = default_manifest(1);
    manifest.explorers.slot_configs = vec![h2ai_types::manifest::ExplorerSlotConfig {
        role_frame: "You are a security engineer.".into(),
        constraint_domains: vec!["security".into()],
        ..Default::default()
    }];

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&ex as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 1.0,
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
        conformal_margin: 0.0,
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        output.diversity_degraded_event.is_none(),
        "domains fully covered → no degraded event expected"
    );
}

#[tokio::test]
async fn system_c3_degraded_event_when_domains_uncovered() {
    let ex =
        MockAdapter::new("stateless JWT auth solution security correctness performance".into());
    let scorer = scoring_adapter();
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "compliant"}"#.into());

    let cfg = H2AIConfig {
        domain_coverage_threshold: 0.8,
        safety: h2ai_config::SafetyConfig {
            require_bivariate_cg: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut d1 = ConstraintDoc::new_llm_judge("S1", "security rule");
    d1.domains = vec!["security".into()];
    let mut d2 = ConstraintDoc::new_llm_judge("P1", "perf rule");
    d2.domains = vec!["performance".into()];
    let mut d3 = ConstraintDoc::new_llm_judge("C1", "correctness rule");
    d3.domains = vec!["correctness".into()];
    let corpus = vec![d1, d2, d3];

    let cal = run_calibration(&[&ex as &dyn IComputeAdapter]).await;
    let store = h2ai_orchestrator::task_store::TaskStore::new();
    let registry = AdapterRegistry::new(
        Arc::new(MockAdapter::new("solution".into())) as Arc<dyn IComputeAdapter>
    );

    let mut manifest = default_manifest(1);
    manifest.explorers.slot_configs = vec![h2ai_types::manifest::ExplorerSlotConfig {
        role_frame: "You are a security engineer.".into(),
        constraint_domains: vec!["security".into()],
        ..Default::default()
    }];

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![&ex as &dyn IComputeAdapter],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: auditor_cfg(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store,
        nats_dispatch: None,
        registry: &registry,
        embedding_model: None,
        tao_multiplier: 1.0,
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
        conformal_margin: 0.0,
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(
        output.diversity_degraded_event.is_some(),
        "domains uncovered → DiversityGuardDegradedEvent expected"
    );
    let evt = output.diversity_degraded_event.unwrap();
    assert!(
        evt.coverage_score < 0.5,
        "coverage_score should be ~0.33, got {}",
        evt.coverage_score
    );
}
