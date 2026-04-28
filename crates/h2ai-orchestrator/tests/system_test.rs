use async_trait::async_trait;
use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_context::adr::parse_adr;
use h2ai_orchestrator::engine::{EngineError, EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::{TaskPhase, TaskStore};
use h2ai_types::adapter::{AdapterError, AdapterRegistry, ComputeRequest, ComputeResponse, IComputeAdapter};
use std::sync::Arc;
use h2ai_types::config::{
    AdapterKind, AgentRole, AuditorConfig, ParetoWeights, RoleSpec, TaoConfig, VerificationConfig,
};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use h2ai_types::physics::MergeStrategy;

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
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

// ── Shared helpers ───────────────────────────────────────────────────────────

fn constraint_corpus() -> Vec<ConstraintDoc> {
    vec![parse_adr(
        "ADR-001",
        "## Constraints\nstateless auth\n",
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
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    }
}

fn auditor_cfg() -> AuditorConfig {
    AuditorConfig {
        adapter: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "NONE".into(),
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
    let cfg = H2AIConfig::default();
    let task_id = TaskId::new();

    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new("stateless JWT auth solution — ADR-001 compliant".into())) as Arc<dyn IComputeAdapter>);
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
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine returned error: {:?}", result.err());

    let out = result.unwrap();
    let state = store.get(&out.task_id).unwrap();

    assert!(
        state.proposals_valid >= 6,
        "expected ≥6 validation events (3 explorers × 2 gates), got {}",
        state.proposals_valid
    );
    assert_eq!(state.proposals_pruned, 0);
    assert_eq!(state.explorers_completed, 3);
    assert!(out.attribution.baseline_quality > 0.0);
    assert!(out.attribution.total_quality >= out.attribution.baseline_quality);
    assert!(out.attribution.total_quality <= 1.0);
    assert_eq!(out.semilattice.valid_proposals.len(), 3);
    assert!(out.semilattice.pruned_proposals.is_empty());
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

    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new("stateless JWT auth implementation ADR-001".into())) as Arc<dyn IComputeAdapter>);
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
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), EngineError::MaxRetriesExhausted),
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

    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new("stateless JWT auth — ADR-001 compliant".into())) as Arc<dyn IComputeAdapter>);
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
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine returned error: {:?}", result.err());

    let out = result.unwrap();
    let state = store.get(&out.task_id).unwrap();

    assert_eq!(state.explorers_completed, 3);
    assert!(state.proposals_valid >= 1);
    assert!(!out.resolved_output.is_empty());
    assert!(out.semilattice.valid_proposals.len() >= 1);
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
        },
        constraints: vec!["ADR-001".into()],
        context: None,
    };

    let registry = AdapterRegistry::new(Arc::new(MockAdapter::new("low-cost auth solution — stateless ADR-001".into())) as Arc<dyn IComputeAdapter>);
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
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
    };

    let result = ExecutionEngine::run_offline(input).await;
    assert!(result.is_ok(), "engine returned error: {:?}", result.err());

    let out = result.unwrap();
    assert_eq!(
        out.semilattice.merge_strategy,
        MergeStrategy::ConsensusMedian
    );
    assert_eq!(out.semilattice.valid_proposals.len(), 2);
    // Condorcet picks the most consensus proposal; with 2 similar proposals either is valid
    assert!(
        out.resolved_output == "low-cost auth solution — stateless ADR-001"
            || out.resolved_output == "high-cost auth solution — stateless ADR-001",
        "expected one of the two valid proposals, got: {}",
        out.resolved_output
    );
}
