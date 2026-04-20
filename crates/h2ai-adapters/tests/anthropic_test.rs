use h2ai_adapters::anthropic::AnthropicAdapter;
use h2ai_types::adapter::{AdapterError, ComputeRequest, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::physics::TauValue;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request() -> ComputeRequest {
    ComputeRequest {
        system_context: "You are a helpful assistant.".into(),
        task: "Explain stateless auth in one sentence.".into(),
        tau: TauValue::new(0.3).unwrap(),
        max_tokens: 64,
    }
}

fn ok_body(text: &str) -> serde_json::Value {
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 20}
    })
}

#[tokio::test]
async fn anthropic_adapter_returns_text_and_token_cost() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(ok_body("JWT tokens are validated on every request.")),
        )
        .mount(&server)
        .await;

    unsafe { std::env::set_var("ANT_TEST_KEY_1", "sk-ant-test") };
    let adapter = AnthropicAdapter::new(
        server.uri(),
        "ANT_TEST_KEY_1".into(),
        "claude-3-5-sonnet-20241022".into(),
    );
    let resp = adapter.execute(request()).await.unwrap();

    assert_eq!(resp.output, "JWT tokens are validated on every request.");
    assert_eq!(resp.token_cost, 30); // 10 + 20
}

#[tokio::test]
async fn anthropic_adapter_network_error_on_http_500() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    unsafe { std::env::set_var("ANT_TEST_KEY_2", "any") };
    let adapter = AnthropicAdapter::new(
        server.uri(),
        "ANT_TEST_KEY_2".into(),
        "claude-3-5-haiku-20241022".into(),
    );
    let result = adapter.execute(request()).await;
    assert!(matches!(result, Err(AdapterError::NetworkError(_))));
}

#[tokio::test]
async fn anthropic_adapter_network_error_when_key_env_missing() {
    let adapter = AnthropicAdapter::new(
        "https://api.anthropic.com".into(),
        "ANT_KEY_CERTAINLY_NOT_SET_XYZ".into(),
        "claude-3-5-sonnet-20241022".into(),
    );
    let result = adapter.execute(request()).await;
    assert!(matches!(result, Err(AdapterError::NetworkError(_))));
}

#[tokio::test]
async fn anthropic_adapter_kind_reflects_constructor_args() {
    let adapter = AnthropicAdapter::new(
        "https://api.anthropic.com".into(),
        "MY_KEY".into(),
        "claude-3-opus-20240229".into(),
    );
    match adapter.kind() {
        AdapterKind::Anthropic { api_key_env, model } => {
            assert_eq!(api_key_env, "MY_KEY");
            assert_eq!(model, "claude-3-opus-20240229");
        }
        other => panic!("unexpected kind: {other:?}"),
    }
}
