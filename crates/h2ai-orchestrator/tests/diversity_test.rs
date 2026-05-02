use chrono::Utc;
use h2ai_orchestrator::diversity::is_uniform;
use h2ai_types::config::AdapterKind;
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::TauValue;

fn proposal(text: &str) -> ProposalEvent {
    ProposalEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.5).unwrap(),
        generation: 0,
        raw_output: text.into(),
        token_cost: 1,
        adapter_kind: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "NONE".into(),
        },
        timestamp: Utc::now(),
    }
}

#[test]
fn empty_proposals_not_uniform() {
    assert!(!is_uniform(&[], 0.85), "zero proposals cannot be uniform");
}

#[test]
fn single_proposal_not_uniform() {
    let proposals = vec![proposal("some output")];
    assert!(
        !is_uniform(&proposals, 0.85),
        "single proposal cannot be uniform"
    );
}

#[test]
fn identical_proposals_are_uniform() {
    let text = "The answer is to use stateless JWT authentication with RS256.";
    let proposals = vec![proposal(text), proposal(text), proposal(text)];
    assert!(
        is_uniform(&proposals, 0.85),
        "identical proposals must be detected as uniform"
    );
}

#[test]
fn diverse_proposals_are_not_uniform() {
    let a = "Use JWT for authentication with short-lived tokens and refresh rotation.";
    let b = "Store sessions in Redis with a 30-minute sliding window expiry.";
    let c = "OAuth2 with PKCE flow; delegate identity to an external IdP.";
    let proposals = vec![proposal(a), proposal(b), proposal(c)];
    assert!(
        !is_uniform(&proposals, 0.85),
        "diverse proposals must not be flagged"
    );
}

#[test]
fn threshold_one_disables_gate_entirely() {
    // threshold >= 1.0 is the "disabled" sentinel — gate returns false even for
    // byte-identical inputs because no Jaccard value can exceed 1.0.
    let same = "The quick brown fox";
    let identical_pair = vec![proposal(same), proposal(same)];
    assert!(
        !is_uniform(&identical_pair, 1.0),
        "threshold 1.0 must return false even for identical proposals"
    );
    let a = "The quick brown fox";
    let b = "The quick brown fox jumps";
    let near_pair = vec![proposal(a), proposal(b)];
    assert!(
        !is_uniform(&near_pair, 1.0),
        "threshold 1.0 must return false for near-identical proposals too"
    );
}

#[test]
fn threshold_zero_always_uniform_for_two_plus_proposals() {
    let a = "completely different output here";
    let b = "nothing in common whatsoever";
    let proposals = vec![proposal(a), proposal(b)];
    // At threshold 0.0, any similarity >= 0.0, which is always true
    assert!(
        is_uniform(&proposals, 0.0),
        "threshold 0.0 must flag any pair as uniform"
    );
}

// ── Integration tests ────────────────────────────────────────────────────────

use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_constraints::loader::parse_constraint_doc;
use h2ai_orchestrator::engine::{EngineError, EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig};
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
    let cal_adapter = MockAdapter::new("The proposed solution uses stateless JWT auth.".into());
    let cal_cfg = H2AIConfig::default();
    let cal = CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Calibrate".into(), "Second task".into(), "Third".into()],
        adapters: vec![&cal_adapter as &dyn IComputeAdapter],
        cfg: &cal_cfg,
        embedding_model: None,
    })
    .await
    .unwrap();

    let corpus = vec![parse_constraint_doc(
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
        embedding_model: None,
    }
}

#[tokio::test]
async fn engine_diversity_gate_triggers_on_identical_outputs() {
    // All explorers return identical text. diversity_threshold=0.5 → is_uniform=true
    // → ZeroSurvivalEvent → RetryPolicy. With max_autonomic_retries=0 → MaxRetriesExhausted.
    let explorer =
        MockAdapter::new("identical output from every explorer stateless auth ADR-001".into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "compliant"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cfg = H2AIConfig {
        diversity_threshold: 0.5,
        max_autonomic_retries: 0,
        ..H2AIConfig::default()
    };
    let store = TaskStore::new();

    let reasoning: Arc<dyn IComputeAdapter> = Arc::new(MockAdapter::new("mock output".into()));
    let registry = AdapterRegistry::new(reasoning);
    let out = ExecutionEngine::run_offline(
        make_engine_input(
            vec![
                &explorer as &dyn IComputeAdapter,
                &explorer as &dyn IComputeAdapter,
                &explorer as &dyn IComputeAdapter,
            ],
            &scorer as &dyn IComputeAdapter,
            &auditor as &dyn IComputeAdapter,
            &cfg,
            store,
            &registry,
        )
        .await,
    )
    .await;

    assert!(out.is_err(), "uniform proposals must trigger failure");
    assert!(
        matches!(out.unwrap_err(), EngineError::MaxRetriesExhausted),
        "should fail with MaxRetriesExhausted after diversity gate"
    );
}

#[tokio::test]
async fn engine_diversity_gate_passes_on_diverse_outputs() {
    // With threshold=0.85 (gate active), two adapters with clearly different token sets
    // → pairwise Jaccard is well below 0.85 → gate does not fire → engine succeeds.
    let explorer_a =
        MockAdapter::new("stateless JWT authentication token ADR-001 compliant".into());
    let explorer_b = MockAdapter::new("Redis session store sliding window expiry approach".into());
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "compliant"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cfg = H2AIConfig {
        diversity_threshold: 0.85,
        ..H2AIConfig::default()
    };
    let store = TaskStore::new();

    let reasoning: Arc<dyn IComputeAdapter> = Arc::new(MockAdapter::new("mock output".into()));
    let registry = AdapterRegistry::new(reasoning);
    let out = ExecutionEngine::run_offline(
        make_engine_input(
            vec![
                &explorer_a as &dyn IComputeAdapter,
                &explorer_b as &dyn IComputeAdapter,
            ],
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
        "diverse explorers should succeed: {:?}",
        out.err()
    );
}
