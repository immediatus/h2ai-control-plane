use h2ai_adapters::openai::OpenAIAdapter;
use h2ai_types::adapter::{AdapterError, ComputeRequest, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::physics::TauValue;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request() -> ComputeRequest {
    ComputeRequest {
        system_context: "You are a helpful assistant.".into(),
        task: "What is JWT?".into(),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 128,
    }
}

fn ok_body(content: &str, total_tokens: u64) -> serde_json::Value {
    serde_json::json!({
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": total_tokens}
    })
}

#[tokio::test]
async fn openai_adapter_returns_content_and_token_cost() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer sk-oai-test"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(ok_body("JWT is a signed token.", 55)),
        )
        .mount(&server)
        .await;

    unsafe { std::env::set_var("OAI_TEST_KEY_1", "sk-oai-test") };
    let adapter = OpenAIAdapter::new(server.uri(), "OAI_TEST_KEY_1".into(), "gpt-4o".into());
    let resp = adapter.execute(request()).await.unwrap();

    assert_eq!(resp.output, "JWT is a signed token.");
    assert_eq!(resp.token_cost, 55);
}

#[tokio::test]
async fn openai_adapter_network_error_on_http_429() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;

    unsafe { std::env::set_var("OAI_TEST_KEY_2", "any") };
    let adapter = OpenAIAdapter::new(server.uri(), "OAI_TEST_KEY_2".into(), "gpt-4o".into());
    let result = adapter.execute(request()).await;
    assert!(matches!(result, Err(AdapterError::NetworkError(_))));
}

#[tokio::test]
async fn openai_adapter_network_error_when_key_env_missing() {
    let adapter = OpenAIAdapter::new(
        "https://api.openai.com/v1".into(),
        "OAI_KEY_CERTAINLY_NOT_SET_XYZ".into(),
        "gpt-4o-mini".into(),
    );
    let result = adapter.execute(request()).await;
    assert!(matches!(result, Err(AdapterError::NetworkError(_))));
}

#[tokio::test]
async fn openai_adapter_kind_reflects_constructor_args() {
    let adapter = OpenAIAdapter::new(
        "https://api.openai.com/v1".into(),
        "OPENAI_API_KEY".into(),
        "gpt-4o".into(),
    );
    match adapter.kind() {
        AdapterKind::OpenAI { api_key_env, model } => {
            assert_eq!(api_key_env, "OPENAI_API_KEY");
            assert_eq!(model, "gpt-4o");
        }
        other => panic!("unexpected kind: {other:?}"),
    }
}

#[tokio::test]
async fn openai_request_body_includes_model_field() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(wiremock::matchers::body_partial_json(
            serde_json::json!({"model": "gpt-4o-mini"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("ok", 10)))
        .mount(&server)
        .await;

    unsafe { std::env::set_var("OAI_TEST_KEY_3", "sk-x") };
    let adapter = OpenAIAdapter::new(server.uri(), "OAI_TEST_KEY_3".into(), "gpt-4o-mini".into());
    // If model field is missing from body, wiremock returns 404 (no mock matches)
    let result = adapter.execute(request()).await;
    assert!(
        result.is_ok(),
        "request body must include model field: {:?}",
        result.err()
    );
}

#[tokio::test]
async fn openai_adapter_network_error_when_response_has_no_choices() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [],
            "usage": {"total_tokens": 0}
        })))
        .mount(&server)
        .await;

    unsafe { std::env::set_var("OAI_TEST_KEY_4", "sk-x") };
    let adapter = OpenAIAdapter::new(server.uri(), "OAI_TEST_KEY_4".into(), "gpt-4o".into());
    let result = adapter.execute(request()).await;
    assert!(
        matches!(result, Err(AdapterError::NetworkError(_))),
        "empty choices must be an error"
    );
}
