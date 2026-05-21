use h2ai_adapters::a2a::A2aExplorerAdapter;
use h2ai_types::adapter::{AdapterError, ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_adapter(base_url: &str) -> A2aExplorerAdapter {
    A2aExplorerAdapter::new(
        base_url.to_string(),
        "none".to_string(),
        String::new(),
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
    assert!(resp.is_ok(), "expected Ok, got: {resp:?}");
    assert!(
        resp.unwrap().output.contains('4'),
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
        "rejected task should return AdapterError::Unavailable, got: {resp:?}"
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
        "404 agent card should return AdapterError::Unavailable, got: {resp:?}"
    );
}

// ---------------------------------------------------------------------------
// Additional a2a coverage: poll states, auth, error paths
// ---------------------------------------------------------------------------

fn make_adapter_with_short_timeout(base_url: &str) -> A2aExplorerAdapter {
    A2aExplorerAdapter::new(
        base_url.to_string(),
        "none".to_string(),
        String::new(),
        1,   // timeout_minutes
        50,  // poll_interval_ms
        200, // max_poll_interval_ms
        3600,
    )
    .expect("adapter construction should succeed")
}

/// Poll state "failed" — should return `AdapterError::Remote`
#[tokio::test]
async fn a2a_adapter_returns_remote_error_on_failed_state() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-fail", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-fail",
                "status": { "state": "failed", "message": "internal error" }
            }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter_with_short_timeout(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::Remote(_))),
        "failed state should return Remote error, got: {resp:?}"
    );
}

/// Poll state "canceled" — should return `AdapterError::Cancelled`
#[tokio::test]
async fn a2a_adapter_returns_cancelled_on_canceled_state() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-cancel", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": { "id": "task-cancel", "status": { "state": "canceled" } }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter_with_short_timeout(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::Cancelled)),
        "canceled state should return Cancelled, got: {resp:?}"
    );
}

/// Poll state "`input_required`" — should return `AdapterError::Timeout`
#[tokio::test]
async fn a2a_adapter_returns_timeout_on_input_required_state() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-input", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": { "id": "task-input", "status": { "state": "input_required" } }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter_with_short_timeout(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::Timeout)),
        "input_required state should return Timeout, got: {resp:?}"
    );
}

/// Poll state "working" (Pending) — eventually completes after a pending cycle
#[tokio::test]
async fn a2a_adapter_retries_on_pending_working_state() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-working", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // First poll: working (pending)
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": { "id": "task-working", "status": { "state": "working" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Second poll: completed
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-working",
                "status": { "state": "completed" },
                "artifacts": [{ "parts": [{ "type": "text", "text": "done after working state" }] }]
            }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter_with_short_timeout(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        resp.is_ok(),
        "should succeed after working->completed, got: {resp:?}"
    );
    assert_eq!(resp.unwrap().output, "done after working state");
}

/// Poll state "`auth_required`" — should return `AdapterError::Unavailable` (same as rejected)
#[tokio::test]
async fn a2a_adapter_returns_unavailable_on_auth_required_state() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-auth", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": { "id": "task-auth", "status": { "state": "auth_required" } }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter_with_short_timeout(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::Unavailable)),
        "auth_required state should return Unavailable, got: {resp:?}"
    );
}

/// Unknown poll state — treated as Pending, so task should eventually time out
/// Use a very short timeout (0 minutes would be instant, but 1 min is minimum).
/// Instead: close the server after sending so we get a network error on poll.
#[tokio::test]
async fn a2a_adapter_handles_unknown_poll_state_as_pending() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    // send returns task id
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-unknown", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // First poll: unknown state → Pending (continues)
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": { "id": "task-unknown", "status": { "state": "some_new_state" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Second poll: completed, so we don't wait forever
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-unknown",
                "status": { "state": "completed" },
                "artifacts": [{ "parts": [{ "type": "text", "text": "recovered" }] }]
            }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter_with_short_timeout(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        resp.is_ok(),
        "unknown state → pending → completed should succeed, got: {resp:?}"
    );
}

/// `send_task` fails with missing task id in response → `NetworkError`
#[tokio::test]
async fn a2a_adapter_errors_when_send_task_response_missing_id() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    // message/send returns response without result.id
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "status": { "state": "submitted" } }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter_with_short_timeout(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::NetworkError(_))),
        "missing task id should return NetworkError, got: {resp:?}"
    );
}

/// Agent card fetch: connection refused → Unavailable (line 259-261)
#[tokio::test]
async fn a2a_adapter_unavailable_when_agent_card_connection_refused() {
    let adapter = A2aExplorerAdapter::new(
        "http://127.0.0.1:1".into(),
        "none".into(),
        String::new(),
        1,
        50,
        200,
        3600,
    )
    .unwrap();
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::Unavailable)),
        "connection refused on agent card should return Unavailable, got: {resp:?}"
    );
}

/// Agent card returns malformed JSON → Unavailable (line 269-271)
#[tokio::test]
async fn a2a_adapter_unavailable_when_agent_card_returns_invalid_json() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("not json", "application/json"))
        .mount(&server)
        .await;

    let adapter = make_adapter(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::Unavailable)),
        "malformed agent card JSON should return Unavailable, got: {resp:?}"
    );
}

/// Constructor fails when `auth_scheme` != "none" but env var is missing (line 169-171)
#[test]
fn a2a_adapter_construction_fails_when_auth_token_env_not_set() {
    let result = A2aExplorerAdapter::new(
        "http://example.com".into(),
        "bearer".into(),
        "A2A_TOKEN_CERTAINLY_NOT_SET_XYZ_12345".into(),
        1,
        50,
        500,
        3600,
    );
    assert!(
        result.is_err(),
        "should fail when env var for auth token is not set"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("A2A_TOKEN_CERTAINLY_NOT_SET_XYZ_12345"),
        "error should name the missing env var: {err}"
    );
}

/// Adapter with bearer auth — exercises `auth_header()` with Bearer scheme (line 255, 356, 385)
#[tokio::test]
async fn a2a_adapter_with_bearer_auth_sends_authorization_header() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-bearer", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-bearer",
                "status": { "state": "completed" },
                "artifacts": [{ "parts": [{ "type": "text", "text": "authenticated response" }] }]
            }
        })))
        .mount(&server)
        .await;

    unsafe { std::env::set_var("A2A_BEARER_TOKEN_TEST", "test-bearer-token") };
    let adapter = A2aExplorerAdapter::new(
        server.uri(),
        "bearer".into(),
        "A2A_BEARER_TOKEN_TEST".into(),
        1,
        50,
        200,
        3600,
    )
    .expect("should construct with bearer auth");

    let resp = adapter.execute(make_request()).await;
    assert!(
        resp.is_ok(),
        "bearer auth request should succeed, got: {resp:?}"
    );
    assert_eq!(resp.unwrap().output, "authenticated response");
}

/// Agent card cache TTL — second request uses cached card (lines 239-241, 247-249).
/// Set up all mocks upfront. The agent card mock is served only once (`up_to_n_times(1)`);
/// the agent card has a long TTL so the second `execute()` call hits the in-memory cache.
#[tokio::test]
async fn a2a_adapter_caches_agent_card() {
    let server = MockServer::start().await;

    // Agent card served at most once — second execute uses the in-memory cache
    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Mocks for first execute(): send → submit, poll → completed
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-c1", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-c1",
                "status": { "state": "completed" },
                "artifacts": [{ "parts": [{ "type": "text", "text": "first result" }] }]
            }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Mocks for second execute(): send → submit, poll → completed
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-c2", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-c2",
                "status": { "state": "completed" },
                "artifacts": [{ "parts": [{ "type": "text", "text": "second result" }] }]
            }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter(&server.uri());
    // First call fills the card cache
    let r1 = adapter.execute(make_request()).await;
    assert!(r1.is_ok(), "first call should succeed: {r1:?}");
    assert_eq!(r1.unwrap().output, "first result");

    // Second call should use the cached agent card (no GET to /.well-known/agent.json)
    let r2 = adapter.execute(make_request()).await;
    assert!(
        r2.is_ok(),
        "second call (cached card) should succeed: {r2:?}"
    );
    assert_eq!(r2.unwrap().output, "second result");
}

/// Poll error: `send_task` returns malformed JSON → `NetworkError`
#[tokio::test]
async fn a2a_adapter_network_error_on_malformed_send_response() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("not json", "application/json"))
        .mount(&server)
        .await;

    let adapter = make_adapter(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::NetworkError(_))),
        "malformed send response should return NetworkError, got: {resp:?}"
    );
}

/// Completed task with whitespace-only artifact inside fences → `extract_proposal` returns Ok("") →
/// `AdapterError::EmptyOutput` (line 332)
#[tokio::test]
async fn a2a_adapter_empty_output_when_artifact_is_whitespace_fenced() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-ws", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Artifact text is a code fence with only whitespace inside → extract_proposal returns Ok("")
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-ws",
                "status": { "state": "completed" },
                "artifacts": [{ "parts": [{ "type": "text", "text": "```\n   \n```" }] }]
            }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter_with_short_timeout(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::EmptyOutput)),
        "whitespace-fenced artifact should return EmptyOutput, got: {resp:?}"
    );
}

/// Completed task with empty artifact text → `AdapterError::EmptyOutput` (lines 327-329)
#[tokio::test]
async fn a2a_adapter_empty_output_when_artifact_text_is_empty() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-empty", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Completed with empty artifact text → extract_proposal("") returns Err
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-empty",
                "status": { "state": "completed" },
                "artifacts": [{ "parts": [{ "type": "text", "text": "" }] }]
            }
        })))
        .mount(&server)
        .await;

    let adapter = make_adapter_with_short_timeout(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::EmptyOutput)),
        "empty artifact text should return EmptyOutput, got: {resp:?}"
    );
}

/// Agent card cache with TTL=0 — second request sees expired cache and re-fetches
/// (exercises `CachedCard::is_expired()` true path, lines 140-142, 241)
#[tokio::test]
async fn a2a_adapter_refetches_agent_card_when_ttl_expired() {
    let server = MockServer::start().await;

    // Agent card served twice (TTL=0 means it expires immediately after the first fetch)
    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-ttl1", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-ttl1",
                "status": { "state": "completed" },
                "artifacts": [{ "parts": [{ "type": "text", "text": "first" }] }]
            }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-ttl2", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "id": "task-ttl2",
                "status": { "state": "completed" },
                "artifacts": [{ "parts": [{ "type": "text", "text": "second" }] }]
            }
        })))
        .mount(&server)
        .await;

    // TTL=0 means the card expires immediately — every call re-fetches
    let adapter = A2aExplorerAdapter::new(
        server.uri(),
        "none".into(),
        String::new(),
        1,
        50,
        200,
        0, // agent_card_cache_ttl_s = 0 → expires immediately
    )
    .unwrap();

    let r1 = adapter.execute(make_request()).await;
    assert!(r1.is_ok(), "first call should succeed: {r1:?}");

    let r2 = adapter.execute(make_request()).await;
    assert!(
        r2.is_ok(),
        "second call with expired cache should succeed: {r2:?}"
    );
}

/// `poll_task` returns malformed JSON → `NetworkError` (line 396-399)
#[tokio::test]
async fn a2a_adapter_network_error_on_malformed_poll_response() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/.well-known/agent.json"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"name": "agent"})),
        )
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "id": "task-poll-bad", "status": { "state": "submitted" } }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Poll returns malformed JSON
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("bad json", "application/json"))
        .mount(&server)
        .await;

    let adapter = make_adapter(&server.uri());
    let resp = adapter.execute(make_request()).await;
    assert!(
        matches!(resp, Err(AdapterError::NetworkError(_))),
        "malformed poll response should return NetworkError, got: {resp:?}"
    );
}
