use h2ai_adapters::a2a::A2aExplorerAdapter;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_adapter(base_url: &str) -> A2aExplorerAdapter {
    A2aExplorerAdapter::new(
        base_url.to_string(),
        "none".to_string(),
        "".to_string(),
        1,   // timeout_minutes
        50,  // poll_interval_ms — fast for tests
        500, // max_poll_interval_ms
        3600,
    )
    .expect("adapter construction should succeed with auth=none")
}

fn make_request() -> ComputeRequest {
    ComputeRequest {
        system_context: "You are a helpful assistant.".to_string(),
        task: "What is 2 + 2?".to_string(),
        tau: TauValue::new(0.7).unwrap(),
        max_tokens: 256,
    }
}

#[tokio::test]
async fn full_delegation_roundtrip_poll_to_completed() {
    let server = MockServer::start().await;

    // Agent Card
    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "test-agent",
            "skills": [{"outputModes": ["text"]}]
        })))
        .mount(&server)
        .await;

    // message/send returns task_id
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "id": "task-abc",
                "status": { "state": "submitted" }
            }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // tasks/get returns completed on first poll
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "id": "task-abc",
                "status": { "state": "completed" },
                "artifacts": [{
                    "parts": [{ "type": "text", "text": "The answer is 4." }]
                }]
            }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(resp.is_ok(), "expected Ok, got: {:?}", resp);
    assert!(
        resp.unwrap().output.contains("4"),
        "output should contain the answer"
    );
}

#[tokio::test]
async fn a2a_adapter_returns_error_on_rejection() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "test-agent",
            "skills": [{"outputModes": ["text"]}]
        })))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": { "id": "task-xyz", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": { "id": "task-xyz", "status": { "state": "rejected" } }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(h2ai_types::adapter::AdapterError::Unavailable)),
        "rejected task should return AdapterError::Unavailable, got: {:?}",
        resp
    );
}

#[tokio::test]
async fn agent_card_not_found_returns_unavailable() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let adapter = make_adapter(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(h2ai_types::adapter::AdapterError::Unavailable)),
        "404 agent card should return AdapterError::Unavailable, got: {:?}",
        resp
    );
}
