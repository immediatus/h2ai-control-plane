use h2ai_autonomic::complexity_probe::{ComplexityProbe, ComplexityProbeResult};
use h2ai_config::ComplexityRoutingConfig;
use h2ai_test_utils::{failing_adapter, mock_adapter};

fn default_cfg() -> ComplexityRoutingConfig {
    ComplexityRoutingConfig::default()
}

#[tokio::test]
async fn probe_parses_valid_json_response() {
    let adapter = mock_adapter(
        r#"{"complexity": 4, "rationale": "formal proof required", "decompose_recommended": true}"#,
    );
    let probe = ComplexityProbe::new(default_cfg());
    let result: ComplexityProbeResult = probe.run("Prove BFT safety", &adapter).await;
    assert_eq!(result.complexity, 4);
    assert!(result.decompose_recommended);
    assert_eq!(result.rationale, "formal proof required");
}

#[tokio::test]
async fn probe_defaults_to_2_on_invalid_json() {
    let adapter = mock_adapter("not json at all");
    let probe = ComplexityProbe::new(default_cfg());
    let result: ComplexityProbeResult = probe.run("some task", &adapter).await;
    assert_eq!(result.complexity, 2); // safe default
    assert!(!result.decompose_recommended);
}

#[tokio::test]
async fn probe_defaults_to_2_on_adapter_failure() {
    let adapter = failing_adapter();
    let probe = ComplexityProbe::new(default_cfg());
    let result: ComplexityProbeResult = probe.run("some task", &adapter).await;
    assert_eq!(result.complexity, 2);
}

#[tokio::test]
async fn probe_clamps_out_of_range_complexity_to_default() {
    let adapter = mock_adapter(
        r#"{"complexity": 9, "rationale": "too high", "decompose_recommended": false}"#,
    );
    let probe = ComplexityProbe::new(default_cfg());
    let result: ComplexityProbeResult = probe.run("some task", &adapter).await;
    assert_eq!(result.complexity, 2); // out of range → safe default
}

#[tokio::test]
async fn probe_parses_json_with_preamble_text() {
    let adapter = mock_adapter(
        r#"Sure, here is the rating: {"complexity": 3, "rationale": "constructive", "decompose_recommended": false}"#,
    );
    let probe = ComplexityProbe::new(default_cfg());
    let result: ComplexityProbeResult = probe.run("design an algorithm", &adapter).await;
    assert_eq!(result.complexity, 3);
}
