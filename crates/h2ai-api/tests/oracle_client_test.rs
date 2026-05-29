#![allow(clippy::float_cmp)]

use axum::{routing::post, Json, Router};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{OracleDomain, OracleSpec};

fn spec(uri: &str) -> OracleSpec {
    OracleSpec {
        runner_uri: uri.to_string(),
        timeout_ms: 5000,
        domain: OracleDomain::Code,
    }
}

fn task_id() -> TaskId {
    TaskId::new()
}

#[tokio::test]
async fn oracle_client_pass() {
    let app = Router::new().route(
        "/evaluate",
        post(|| async {
            Json(serde_json::json!({"passed": true, "score": 0.95, "details": {"test": "ok"}}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = h2ai_api::oracle::client::OracleClient::new();
    let s = spec(&format!("http://{addr}/evaluate"));
    let resp = client.evaluate(&s, &task_id(), "output text").await;

    assert!(resp.passed);
    assert!((resp.score - 0.95).abs() < 1e-9);
    assert_eq!(resp.details["test"], "ok");
}

#[tokio::test]
async fn oracle_client_fail() {
    let app = Router::new().route(
        "/evaluate",
        post(|| async {
            Json(serde_json::json!({"passed": false, "score": 0.0, "details": {"error": "failed"}}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = h2ai_api::oracle::client::OracleClient::new();
    let s = spec(&format!("http://{addr}/evaluate"));
    let resp = client.evaluate(&s, &task_id(), "output text").await;

    assert!(!resp.passed);
    assert_eq!(resp.score, 0.0);
}

#[tokio::test]
async fn oracle_client_unreachable_returns_fail() {
    let client = h2ai_api::oracle::client::OracleClient::new();
    let s = spec("http://127.0.0.1:19999/evaluate");
    let resp = client.evaluate(&s, &task_id(), "output text").await;

    assert!(!resp.passed);
    assert_eq!(resp.score, 0.0);
    assert!(resp.details.get("error").is_some());
}

#[tokio::test]
async fn oracle_client_empty_runner_uri_returns_fail() {
    let client = h2ai_api::oracle::client::OracleClient::new();
    let s = OracleSpec {
        runner_uri: String::new(),
        timeout_ms: 1000,
        domain: OracleDomain::Code,
    };
    let resp = client.evaluate(&s, &task_id(), "output").await;

    assert!(!resp.passed);
    assert_eq!(resp.score, 0.0);
    assert_eq!(resp.details["error"], "runner_uri is empty");
}

#[tokio::test]
async fn oracle_client_non_2xx_returns_fail() {
    use axum::http::StatusCode;
    let app = Router::new().route(
        "/evaluate",
        post(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = h2ai_api::oracle::client::OracleClient::new();
    let s = spec(&format!("http://{addr}/evaluate"));
    let resp = client.evaluate(&s, &task_id(), "output").await;

    assert!(!resp.passed);
    assert_eq!(resp.score, 0.0);
    assert_eq!(resp.details["error"], "HTTP 500");
}

#[tokio::test]
async fn oracle_client_timeout_returns_fail() {
    let app = Router::new().route(
        "/evaluate",
        post(|| async {
            std::future::pending::<()>().await;
            Json(serde_json::json!({"passed": true, "score": 1.0, "details": {}}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = h2ai_api::oracle::client::OracleClient::new();
    let mut s = spec(&format!("http://{addr}/evaluate"));
    s.timeout_ms = 50;

    let resp = client.evaluate(&s, &task_id(), "output").await;

    assert!(!resp.passed);
    assert_eq!(resp.score, 0.0);
    assert_eq!(resp.details["error"], "timeout");
}

#[tokio::test]
async fn oracle_client_invalid_json_returns_fail() {
    let app = Router::new().route(
        "/evaluate",
        post(|| async {
            axum::response::Response::builder()
                .header("content-type", "application/json")
                .body(axum::body::Body::from("not-json"))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = h2ai_api::oracle::client::OracleClient::new();
    let s = spec(&format!("http://{addr}/evaluate"));
    let resp = client.evaluate(&s, &task_id(), "output").await;

    assert!(!resp.passed);
    assert_eq!(resp.score, 0.0);
    assert!(resp.details.get("error").is_some());
}

#[tokio::test]
async fn oracle_client_missing_fields_defaults() {
    let app = Router::new().route("/evaluate", post(|| async { Json(serde_json::json!({})) }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = h2ai_api::oracle::client::OracleClient::new();
    let s = spec(&format!("http://{addr}/evaluate"));
    let resp = client.evaluate(&s, &task_id(), "output").await;

    assert!(!resp.passed);
    assert_eq!(resp.score, 0.0);
    assert!(resp.details.is_null());
}

#[tokio::test]
async fn oracle_client_request_body_contains_task_id_and_domain() {
    use std::sync::{Arc, Mutex};
    let captured = Arc::new(Mutex::new(serde_json::Value::Null));
    let captured2 = captured.clone();

    let app = Router::new().route(
        "/evaluate",
        post(|Json(body): Json<serde_json::Value>| async move {
            *captured2.lock().unwrap() = body;
            Json(serde_json::json!({"passed": true, "score": 1.0, "details": {}}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let client = h2ai_api::oracle::client::OracleClient::new();
    let s = spec(&format!("http://{addr}/evaluate"));
    let tid = task_id();
    client.evaluate(&s, &tid, "my output").await;

    let body = captured.lock().unwrap().clone();
    assert_eq!(body["task_id"], tid.to_string());
    assert_eq!(body["output"], "my output");
    assert!(body.get("domain").is_some());
}

#[tokio::test]
async fn oracle_client_default_constructs_successfully() {
    use h2ai_api::oracle::client::OracleClient;
    let client = OracleClient::default();
    let s = OracleSpec {
        runner_uri: String::new(),
        timeout_ms: 1000,
        domain: h2ai_types::sizing::OracleDomain::Code,
    };
    let resp = client.evaluate(&s, &task_id(), "test").await;
    // empty runner_uri always returns passed=false
    assert!(!resp.passed);
}
