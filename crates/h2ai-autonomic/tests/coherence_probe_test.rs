use h2ai_autonomic::coherence_probe::{CoherenceProbe, ProbeMode, ProbeResult};
use h2ai_config::GapK1Config;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

mockall::mock! {
    #[derive(Debug)]
    pub CoherenceAdapter {}
    #[async_trait::async_trait]
    impl IComputeAdapter for CoherenceAdapter {
        async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError>;
        fn kind(&self) -> &AdapterKind;
    }
}

fn mock_kind() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "mock".into(),
        api_key_env: "MOCK".into(),
        model: None,
        provider: Default::default(),
    }
}

fn pass_response() -> ComputeResponse {
    ComputeResponse {
        output: r#"{"verdict":"pass","score":0.9}"#.to_owned(),
        token_cost: 0,
        adapter_kind: mock_kind(),
        tokens_used: None,
        reasoning_trace: None,
    }
}

fn fail_response() -> ComputeResponse {
    ComputeResponse {
        output: r#"{"verdict":"fail","score":0.1}"#.to_owned(),
        token_cost: 0,
        adapter_kind: mock_kind(),
        tokens_used: None,
        reasoning_trace: None,
    }
}

/// Builds a mock adapter that alternates pass/fail responses.
fn alternating_adapter() -> MockCoherenceAdapter {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut m = MockCoherenceAdapter::new();
    m.expect_execute().returning(move |_| {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        if n.is_multiple_of(2) {
            Ok(pass_response())
        } else {
            Ok(fail_response())
        }
    });
    m.expect_kind().return_const(mock_kind()).times(0..);
    m
}

/// Builds a mock adapter that always returns pass.
fn always_pass_adapter() -> MockCoherenceAdapter {
    let mut m = MockCoherenceAdapter::new();
    m.expect_execute().returning(|_| Ok(pass_response()));
    m.expect_kind().return_const(mock_kind()).times(0..);
    m
}

#[tokio::test]
async fn alternating_adapter_produces_low_consistency() {
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(alternating_adapter());
    let cfg = GapK1Config {
        probe_runs: 6,
        ..Default::default()
    };
    let probe = CoherenceProbe::new(cfg);

    let result: ProbeResult = probe
        .run("check text", "the proposal text", &*adapter)
        .await;

    // 3 pass out of 6 → consistency = 0.5
    assert!((result.consistency - 0.5).abs() < 0.01);
    assert_eq!(result.mode, ProbeMode::ExampleBased);
}

fn make_text_response(text: &str) -> ComputeResponse {
    ComputeResponse {
        output: text.to_owned(),
        token_cost: 0,
        adapter_kind: mock_kind(),
        tokens_used: None,
        reasoning_trace: None,
    }
}

fn make_json_verdict_only_response(verdict: &str) -> ComputeResponse {
    // JSON with "verdict" but NO "score" key → exercises lines 97-100 in parse_verdict
    ComputeResponse {
        output: format!(r#"{{"verdict":"{verdict}"}}"#),
        token_cost: 0,
        adapter_kind: mock_kind(),
        tokens_used: None,
        reasoning_trace: None,
    }
}

#[tokio::test]
async fn verdict_only_json_without_score_key_is_parsed() {
    // Exercises lines 97-100: JSON has "verdict" but no "score" key
    let mut m = MockCoherenceAdapter::new();
    m.expect_execute()
        .returning(|_| Ok(make_json_verdict_only_response("pass")));
    m.expect_kind().return_const(mock_kind()).times(0..);
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(m);
    let cfg = GapK1Config {
        probe_runs: 2,
        ..Default::default()
    };
    let probe = CoherenceProbe::new(cfg);
    let result = probe.run("check", "example", &*adapter).await;
    assert!(
        (result.consistency - 1.0).abs() < 0.01,
        "verdict-only JSON must parse as pass"
    );
}

#[tokio::test]
async fn plain_text_pass_response_parsed() {
    // Exercises line 105: text fallback containing "pass"
    let mut m = MockCoherenceAdapter::new();
    m.expect_execute()
        .returning(|_| Ok(make_text_response("PASS")));
    m.expect_kind().return_const(mock_kind()).times(0..);
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(m);
    let cfg = GapK1Config {
        probe_runs: 2,
        ..Default::default()
    };
    let probe = CoherenceProbe::new(cfg);
    let result = probe.run("check", "example", &*adapter).await;
    assert!(
        (result.consistency - 1.0).abs() < 0.01,
        "plain 'PASS' text must parse as pass"
    );
}

#[tokio::test]
async fn plain_text_fail_response_parsed() {
    // Exercises line 107: text fallback containing "fail"
    let mut m = MockCoherenceAdapter::new();
    m.expect_execute()
        .returning(|_| Ok(make_text_response("FAIL")));
    m.expect_kind().return_const(mock_kind()).times(0..);
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(m);
    let cfg = GapK1Config {
        probe_runs: 2,
        ..Default::default()
    };
    let probe = CoherenceProbe::new(cfg);
    let result = probe.run("check", "example", &*adapter).await;
    assert!(
        (result.consistency - 0.0).abs() < 0.01,
        "plain 'FAIL' text must parse as fail"
    );
}

#[tokio::test]
async fn always_pass_adapter_produces_high_consistency() {
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(always_pass_adapter());
    let cfg = GapK1Config {
        probe_runs: 5,
        ..Default::default()
    };
    let probe = CoherenceProbe::new(cfg);

    let result: ProbeResult = probe
        .run("check text", "the proposal text", &*adapter)
        .await;

    assert!((result.consistency - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn below_threshold_sets_is_coherent_false() {
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(alternating_adapter());
    let cfg = GapK1Config {
        probe_runs: 4,
        coherence_threshold: 0.80,
        ..Default::default()
    };
    let probe = CoherenceProbe::new(cfg);
    let result: ProbeResult = probe.run("check", "example", &*adapter).await;
    assert!(!result.is_coherent);
}
