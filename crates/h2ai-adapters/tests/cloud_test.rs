use h2ai_adapters::cloud::CloudGenericAdapter;
use h2ai_types::adapter::{AdapterError, ComputeRequest, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::sizing::TauValue;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request() -> ComputeRequest {
    ComputeRequest {
        system_context: "you are a helpful assistant".into(),
        task: "say hello".into(),
        tau: TauValue::new(0.3).unwrap(),
        max_tokens: 50,
    }
}

fn ok_body(content: &str, total_tokens: u64) -> serde_json::Value {
    serde_json::json!({
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": total_tokens}
    })
}

fn reasoning_body(reasoning_content: &str, total_tokens: u64) -> serde_json::Value {
    serde_json::json!({
        "choices": [{"message": {"content": "", "reasoning_content": reasoning_content}}],
        "usage": {"total_tokens": total_tokens}
    })
}

fn reasoning_body_no_content_key(reasoning_content: &str, total_tokens: u64) -> serde_json::Value {
    serde_json::json!({
        "choices": [{"message": {"reasoning_content": reasoning_content}}],
        "usage": {"total_tokens": total_tokens}
    })
}

#[tokio::test]
async fn cloud_adapter_returns_parsed_content_and_token_count() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", "Bearer sk-test-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("hello!", 42)))
        .mount(&server)
        .await;

    // SAFETY: test-only env mutation; tests run in separate processes
    unsafe { std::env::set_var("H2AI_TEST_KEY_1", "sk-test-1") };
    let adapter = CloudGenericAdapter::new(server.uri(), "H2AI_TEST_KEY_1".into());
    let resp = adapter.execute(request()).await.unwrap();

    assert_eq!(resp.output, "hello!");
    assert_eq!(resp.token_cost, 42);
}

#[tokio::test]
async fn cloud_adapter_returns_network_error_on_http_500() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    unsafe { std::env::set_var("H2AI_TEST_KEY_2", "any") };
    let adapter = CloudGenericAdapter::new(server.uri(), "H2AI_TEST_KEY_2".into());
    let result = adapter.execute(request()).await;
    assert!(matches!(result, Err(AdapterError::NetworkError(_))));
}

#[tokio::test]
async fn cloud_adapter_returns_network_error_when_env_var_missing() {
    let adapter = CloudGenericAdapter::new(
        "https://api.example.com".into(),
        "H2AI_TEST_KEY_CERTAINLY_NOT_SET_XYZ".into(),
    );
    let result = adapter.execute(request()).await;
    assert!(matches!(result, Err(AdapterError::NetworkError(_))));
}

/// Reasoning models (DeepSeek R1, Qwen3, etc.) leave `content` as an empty string and place
/// their output in `reasoning_content`. The adapter must fall back to `reasoning_content`
/// when `content` is absent or empty.
#[tokio::test]
async fn cloud_adapter_uses_reasoning_content_when_content_is_empty_string() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(reasoning_body("chain-of-thought answer", 77)),
        )
        .mount(&server)
        .await;

    unsafe { std::env::set_var("H2AI_TEST_KEY_REASONING_1", "sk-test-r1") };
    let adapter = CloudGenericAdapter::new(server.uri(), "H2AI_TEST_KEY_REASONING_1".into());
    let resp = adapter.execute(request()).await.unwrap();

    assert_eq!(resp.output, "chain-of-thought answer");
    assert_eq!(resp.token_cost, 77);
}

#[tokio::test]
async fn cloud_adapter_uses_reasoning_content_when_content_key_absent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(reasoning_body_no_content_key(
                "answer without content key",
                55,
            )),
        )
        .mount(&server)
        .await;

    unsafe { std::env::set_var("H2AI_TEST_KEY_REASONING_2", "sk-test-r2") };
    let adapter = CloudGenericAdapter::new(server.uri(), "H2AI_TEST_KEY_REASONING_2".into());
    let resp = adapter.execute(request()).await.unwrap();

    assert_eq!(resp.output, "answer without content key");
    assert_eq!(resp.token_cost, 55);
}

#[tokio::test]
async fn cloud_adapter_prefers_content_over_reasoning_content_when_both_present() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {
                "content": "final answer",
                "reasoning_content": "intermediate thinking"
            }}],
            "usage": {"total_tokens": 99}
        })))
        .mount(&server)
        .await;

    unsafe { std::env::set_var("H2AI_TEST_KEY_REASONING_3", "sk-test-r3") };
    let adapter = CloudGenericAdapter::new(server.uri(), "H2AI_TEST_KEY_REASONING_3".into());
    let resp = adapter.execute(request()).await.unwrap();

    assert_eq!(resp.output, "final answer");
}

#[tokio::test]
async fn cloud_adapter_kind_reflects_constructor_args() {
    let adapter = CloudGenericAdapter::new("https://api.example.com".into(), "MY_KEY".into());
    match adapter.kind() {
        AdapterKind::CloudGeneric {
            endpoint,
            api_key_env,
        } => {
            assert_eq!(endpoint, "https://api.example.com");
            assert_eq!(api_key_env, "MY_KEY");
        }
        other => panic!("unexpected kind: {other:?}"),
    }
}
