use h2ai_adapters::ollama::OllamaAdapter;
use h2ai_types::adapter::{AdapterError, ComputeRequest, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::physics::TauValue;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request() -> ComputeRequest {
    ComputeRequest {
        system_context: "You are a helpful assistant.".into(),
        task: "What is Docker?".into(),
        tau: TauValue::new(0.4).unwrap(),
        max_tokens: 200,
    }
}

fn ok_body(content: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "llama3.2",
        "message": {"role": "assistant", "content": content},
        "done": true,
        "prompt_eval_count": 12,
        "eval_count": 30
    })
}

#[tokio::test]
async fn ollama_adapter_returns_content_and_token_cost() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(ok_body("Docker is a container runtime.")),
        )
        .mount(&server)
        .await;

    let adapter = OllamaAdapter::new(server.uri(), "llama3.2".into());
    let resp = adapter.execute(request()).await.unwrap();

    assert_eq!(resp.output, "Docker is a container runtime.");
    assert_eq!(resp.token_cost, 42); // 12 + 30
}

#[tokio::test]
async fn ollama_adapter_handles_missing_eval_counts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Cached answer."},
            "done": true
        })))
        .mount(&server)
        .await;

    let adapter = OllamaAdapter::new(server.uri(), "llama3.2".into());
    let resp = adapter.execute(request()).await.unwrap();

    assert_eq!(resp.output, "Cached answer.");
    assert_eq!(resp.token_cost, 0); // graceful fallback
}

#[tokio::test]
async fn ollama_adapter_network_error_on_http_500() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let adapter = OllamaAdapter::new(server.uri(), "llama3.2".into());
    let result = adapter.execute(request()).await;
    assert!(matches!(result, Err(AdapterError::NetworkError(_))));
}

#[tokio::test]
async fn ollama_request_body_includes_model_and_stream_false() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(serde_json::json!({
            "model": "llama3.2",
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("ok")))
        .mount(&server)
        .await;

    let adapter = OllamaAdapter::new(server.uri(), "llama3.2".into());
    let resp = adapter.execute(request()).await.unwrap();
    assert_eq!(resp.output, "ok");
}

#[tokio::test]
async fn ollama_adapter_kind_reflects_constructor_args() {
    let adapter = OllamaAdapter::new("http://localhost:11434".into(), "mistral".into());
    match adapter.kind() {
        AdapterKind::Ollama { endpoint, model } => {
            assert_eq!(endpoint, "http://localhost:11434");
            assert_eq!(model, "mistral");
        }
        other => panic!("unexpected kind: {other:?}"),
    }
}
