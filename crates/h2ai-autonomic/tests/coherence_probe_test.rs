use h2ai_autonomic::coherence_probe::{CoherenceProbe, ProbeMode, ProbeResult};
use h2ai_config::GapK1Config;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Mock adapter that alternates Pass/Fail responses.
#[derive(Debug)]
struct AlternatingAdapter {
    call_count: Arc<AtomicUsize>,
    kind: AdapterKind,
}

#[async_trait::async_trait]
impl IComputeAdapter for AlternatingAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        let output = if n.is_multiple_of(2) {
            r#"{"verdict":"pass","score":0.9}"#.to_owned()
        } else {
            r#"{"verdict":"fail","score":0.1}"#.to_owned()
        };
        Ok(ComputeResponse {
            output,
            token_cost: 0,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "MOCK".into(),
                model: None,
            },
            tokens_used: None,
            reasoning_trace: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

/// Mock adapter that always returns pass.
#[derive(Debug)]
struct AlwaysPassAdapter {
    kind: AdapterKind,
}

#[async_trait::async_trait]
impl IComputeAdapter for AlwaysPassAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Ok(ComputeResponse {
            output: r#"{"verdict":"pass","score":1.0}"#.to_owned(),
            token_cost: 0,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "MOCK".into(),
                model: None,
            },
            tokens_used: None,
            reasoning_trace: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

#[tokio::test]
async fn alternating_adapter_produces_low_consistency() {
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(AlternatingAdapter {
        call_count: Arc::new(AtomicUsize::new(0)),
        kind: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "MOCK".into(),
            model: None,
        },
    });
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

#[tokio::test]
async fn always_pass_adapter_produces_high_consistency() {
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(AlwaysPassAdapter {
        kind: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "MOCK".into(),
            model: None,
        },
    });
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
    let adapter: Arc<dyn IComputeAdapter> = Arc::new(AlternatingAdapter {
        call_count: Arc::new(AtomicUsize::new(0)),
        kind: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "MOCK".into(),
            model: None,
        },
    });
    let cfg = GapK1Config {
        probe_runs: 4,
        coherence_threshold: 0.80,
        ..Default::default()
    };
    let probe = CoherenceProbe::new(cfg);
    let result: ProbeResult = probe.run("check", "example", &*adapter).await;
    assert!(!result.is_coherent);
}
