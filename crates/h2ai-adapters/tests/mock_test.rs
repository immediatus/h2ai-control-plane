use h2ai_adapters::mock::MockAdapter;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::physics::TauValue;

fn request() -> ComputeRequest {
    ComputeRequest {
        system_context: "you are a test assistant".into(),
        task: "say hello".into(),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 100,
    }
}

#[tokio::test]
async fn mock_adapter_returns_configured_output() {
    let adapter = MockAdapter::new("hello world".into());
    let resp = adapter.execute(request()).await.unwrap();
    assert_eq!(resp.output, "hello world");
}

#[tokio::test]
async fn mock_adapter_returns_zero_token_cost() {
    let adapter = MockAdapter::new("ok".into());
    let resp = adapter.execute(request()).await.unwrap();
    assert_eq!(resp.token_cost, 0);
}

#[tokio::test]
async fn mock_adapter_kind_is_cloud_generic() {
    let adapter = MockAdapter::new("ok".into());
    assert!(matches!(adapter.kind(), AdapterKind::CloudGeneric { .. }));
}

#[tokio::test]
async fn mock_adapter_echoes_same_output_on_repeated_calls() {
    let adapter = MockAdapter::new("constant".into());
    let r1 = adapter.execute(request()).await.unwrap();
    let r2 = adapter.execute(request()).await.unwrap();
    assert_eq!(r1.output, r2.output);
}
