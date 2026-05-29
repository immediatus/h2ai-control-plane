use h2ai_test_utils::{decomposition_adapter, failing_adapter, mock_adapter, sequenced_adapter};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::sizing::TauValue;

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
    let adapter = mock_adapter("hello world");
    let resp = adapter.execute(request()).await.unwrap();
    assert_eq!(resp.output, "hello world");
}

#[tokio::test]
async fn mock_adapter_returns_zero_token_cost() {
    let adapter = mock_adapter("x");
    let resp = adapter.execute(request()).await.unwrap();
    assert_eq!(resp.token_cost, 0);
}

#[tokio::test]
async fn mock_adapter_kind_is_cloud_generic() {
    let adapter = mock_adapter("x");
    assert!(matches!(adapter.kind(), AdapterKind::CloudGeneric { .. }));
}

#[tokio::test]
async fn mock_adapter_echoes_same_output_on_repeated_calls() {
    let adapter = mock_adapter("y");
    let r1 = adapter.execute(request()).await.unwrap();
    let r2 = adapter.execute(request()).await.unwrap();
    assert_eq!(r1.output, r2.output);
}

#[tokio::test]
async fn decomposition_mock_adapter_constructs() {
    let adapter = decomposition_adapter("fb");
    assert!(matches!(adapter.kind(), AdapterKind::CloudGeneric { .. }));
}

#[tokio::test]
async fn decomposition_mock_adapter_returns_step3_json_for_json_formatter_context() {
    let adapter = decomposition_adapter("fb");
    let req = ComputeRequest {
        system_context: "You are a JSON formatter assistant".into(),
        task: "Decompose this task".into(),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 500,
    };
    let resp = adapter.execute(req).await.unwrap();
    assert!(
        resp.output.contains("role_frame"),
        "expected STEP3_MOCK_JSON, got: {}",
        resp.output
    );
    assert_eq!(resp.token_cost, 10);
}

#[tokio::test]
async fn decomposition_mock_adapter_returns_fallback_for_other_context() {
    let adapter = decomposition_adapter("fallback");
    let req = ComputeRequest {
        system_context: "You are a generic assistant".into(),
        task: "Do something".into(),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 100,
    };
    let resp = adapter.execute(req).await.unwrap();
    assert_eq!(resp.output, "fallback");
    assert_eq!(resp.token_cost, 10);
}

#[tokio::test]
async fn decomposition_mock_adapter_kind_is_cloud_generic() {
    let adapter = decomposition_adapter("x");
    assert!(matches!(adapter.kind(), AdapterKind::CloudGeneric { .. }));
}

#[tokio::test]
async fn sequenced_mock_adapter_constructs() {
    let adapter = sequenced_adapter(vec!["a".into()]);
    assert!(matches!(adapter.kind(), AdapterKind::CloudGeneric { .. }));
}

#[tokio::test]
async fn sequenced_mock_adapter_drains_responses_in_order() {
    let adapter = sequenced_adapter(vec!["first".into(), "second".into()]);
    let r1 = adapter.execute(request()).await.unwrap();
    let r2 = adapter.execute(request()).await.unwrap();
    assert_eq!(r1.output, "first");
    assert_eq!(r2.output, "second");
    assert_eq!(r1.token_cost, 10);
}

#[tokio::test]
async fn sequenced_mock_adapter_returns_fallback_when_exhausted() {
    let adapter = sequenced_adapter(vec!["only".into()]);
    let _r1 = adapter.execute(request()).await.unwrap();
    let r2 = adapter.execute(request()).await.unwrap();
    assert_eq!(r2.output, "fallback");
}

#[tokio::test]
async fn sequenced_mock_adapter_kind_is_cloud_generic() {
    let adapter = sequenced_adapter(vec![]);
    assert!(matches!(adapter.kind(), AdapterKind::CloudGeneric { .. }));
}

#[tokio::test]
async fn failing_adapter_returns_network_error() {
    let adapter = failing_adapter();
    let result = adapter.execute(request()).await;
    assert!(result.is_err(), "expected Err from failing_adapter");
}
