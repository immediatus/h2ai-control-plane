use h2ai_autonomic::spec_repair::{RepairInput, RepairOutcome, SpecRepairAdvisor};
use h2ai_config::GapK1Config;
use h2ai_constraints::{
    nats_versioned::NatsVersionedSource, source::InMemorySource, spec::SemanticSpec,
    versioned::VersionedConstraintSource,
};
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use std::sync::Arc;

mockall::mock! {
    #[derive(Debug)]
    pub RepairAdapter {}
    #[async_trait::async_trait]
    impl IComputeAdapter for RepairAdapter {
        async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError>;
        fn kind(&self) -> &AdapterKind;
    }
}

fn make_spec(id: &str, check: &str) -> SemanticSpec {
    let mut s = SemanticSpec::default_for_test(id);
    s.rubric.checks = vec![check.to_owned()];
    s.rubric.pass = "good proposal text".into();
    s
}

fn mock_adapter_kind() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "mock".into(),
        api_key_env: "MOCK".into(),
        model: None,
        provider: Default::default(),
    }
}

fn make_pass_response() -> ComputeResponse {
    ComputeResponse {
        output: r#"{"verdict":"pass","score":1.0}"#.to_owned(),
        token_cost: 0,
        adapter_kind: mock_adapter_kind(),
        tokens_used: None,
        reasoning_trace: None,
    }
}

fn make_fail_response() -> ComputeResponse {
    ComputeResponse {
        output: r#"{"verdict":"fail","score":0.1}"#.to_owned(),
        token_cost: 0,
        adapter_kind: mock_adapter_kind(),
        tokens_used: None,
        reasoning_trace: None,
    }
}

fn make_candidates_response() -> ComputeResponse {
    ComputeResponse {
        output: "Use atomic CAS operation only.\nUse Redis EVAL script only.\nUse compare-and-swap only.".to_owned(),
        token_cost: 0,
        adapter_kind: mock_adapter_kind(),
        tokens_used: None,
        reasoning_trace: None,
    }
}

/// Mock: generates 3 rewrite lines and always passes coherence checks.
fn always_accept_adapter() -> MockRepairAdapter {
    let mut m = MockRepairAdapter::new();
    m.expect_execute().returning(|req| {
        // If it looks like a coherence probe (small max_tokens), return pass
        if req.max_tokens <= 64 {
            return Ok(make_pass_response());
        }
        // Otherwise it's the repair advisor — return 3 candidate lines
        Ok(make_candidates_response())
    });
    m.expect_kind().return_const(mock_adapter_kind()).times(0..);
    m
}

/// Mock: generates candidates but always fails coherence checks.
fn always_fail_adapter() -> MockRepairAdapter {
    let mut m = MockRepairAdapter::new();
    m.expect_execute().returning(|req| {
        if req.max_tokens <= 64 {
            return Ok(make_fail_response());
        }
        Ok(ComputeResponse {
            output: "candidate one.\ncandidate two.\ncandidate three.".to_owned(),
            token_cost: 0,
            adapter_kind: mock_adapter_kind(),
            tokens_used: None,
            reasoning_trace: None,
        })
    });
    m.expect_kind().return_const(mock_adapter_kind()).times(0..);
    m
}

#[tokio::test]
async fn repair_advisor_creates_new_version_when_candidate_accepted() {
    let inner = InMemorySource {
        specs: vec![make_spec("C-1", "ambiguous check")],
    };
    let source = Arc::new(NatsVersionedSource::new_in_memory(inner));
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(always_accept_adapter());
    let cfg = GapK1Config {
        repair_candidates: 3,
        probe_runs: 3,
        repair_acceptance_threshold: 0.80,
        ..Default::default()
    };

    let input = RepairInput {
        task_id: "t1".into(),
        constraint_id: "C-1".into(),
        check_index: 0,
        original_check_text: "ambiguous check".into(),
        divergent_reasons: vec!["reason A".into(), "reason B".into()],
        should_pass_example: "good proposal text".into(),
        should_prune_example: None,
        current_version: 1,
    };

    let advisor = SpecRepairAdvisor::new(cfg);
    let outcome = advisor.run(input, source.clone(), &*adapter).await;

    match outcome {
        RepairOutcome::Repaired {
            new_version,
            accepted_rewrite,
        } => {
            assert_eq!(new_version, 2);
            assert!(
                !accepted_rewrite.is_empty(),
                "Repaired outcome must carry the accepted rewrite text"
            );
        }
        RepairOutcome::Failed { best_score } => {
            panic!("expected Repaired, got Failed({best_score})")
        }
    }

    let vs = source.load_latest_versioned("C-1").await.unwrap();
    assert_eq!(vs.spec.version, 2);
}

#[tokio::test]
async fn repair_advisor_fails_when_below_threshold() {
    let inner = InMemorySource {
        specs: vec![make_spec("C-1", "bad check")],
    };
    let source = Arc::new(NatsVersionedSource::new_in_memory(inner));
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(always_fail_adapter());
    let cfg = GapK1Config {
        repair_candidates: 3,
        probe_runs: 3,
        repair_acceptance_threshold: 0.80,
        ..Default::default()
    };

    let input = RepairInput {
        task_id: "t1".into(),
        constraint_id: "C-1".into(),
        check_index: 0,
        original_check_text: "bad check".into(),
        divergent_reasons: vec!["r1".into()],
        should_pass_example: "example".into(),
        should_prune_example: None,
        current_version: 1,
    };

    let advisor = SpecRepairAdvisor::new(cfg);
    let outcome = advisor.run(input, source.clone(), &*adapter).await;

    assert!(matches!(outcome, RepairOutcome::Failed { .. }));
}
