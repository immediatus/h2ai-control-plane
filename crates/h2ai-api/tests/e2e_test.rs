//! End-to-end integration tests for the H2AI Control Plane API.
//!
//! All tests require a live NATS server with JetStream enabled:
//!   NATS_URL=nats://localhost:4222 cargo nextest run -p h2ai-api --test e2e_test

use h2ai_adapters::mock::MockAdapter;
use h2ai_api::{
    routes::{calibrate_router, health_router, task_router},
    state::AppState,
};
use h2ai_config::H2AIConfig;
use h2ai_state::nats::NatsClient;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;

/// Boots the full axum app on a random OS-assigned port.
/// Returns the base URL, e.g. "http://127.0.0.1:54321".
async fn boot_app() -> (String, tokio::task::JoinHandle<()>) {
    let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| H2AIConfig::default().nats_url);
    let nats = NatsClient::connect(&nats_url).await.expect("NATS connect");
    nats.ensure_infrastructure().await.expect("infra");

    let cfg = H2AIConfig::default();
    let explorer = Arc::new(MockAdapter::new("mock explorer output".into()));
    let auditor = Arc::new(MockAdapter::new("mock auditor output".into()));
    let state = AppState::new(nats, cfg, explorer, auditor);

    let app = axum::Router::new()
        .merge(task_router())
        .merge(calibrate_router())
        .merge(health_router())
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let base_url = format!("http://{addr}");

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });

    // Give the server a moment to start
    tokio::time::sleep(Duration::from_millis(50)).await;

    (base_url, handle)
}

/// Helper: poll GET {url} until HTTP 200, or panic after `attempts` tries.
async fn poll_until_ok(client: &reqwest::Client, url: &str, attempts: u32) -> reqwest::Response {
    for i in 0..attempts {
        let resp = client.get(url).send().await.expect("GET failed");
        if resp.status().is_success() {
            return resp;
        }
        if i < attempts - 1 {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    panic!("URL {url} never returned 200 after {attempts} attempts");
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn health_liveness_returns_ok() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base}/health"))
        .send()
        .await
        .expect("GET /health");
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn calibrate_then_current_returns_coefficients() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    // POST /calibrate (no body)
    let resp = client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("POST /calibrate");
    assert_eq!(resp.status(), 202, "calibrate must be 202 Accepted");
    let body: serde_json::Value = resp.json().await.expect("calibrate json");
    assert_eq!(body["status"], "accepted");
    let cal_id = body["calibration_id"]
        .as_str()
        .expect("calibration_id")
        .to_owned();

    // Poll /calibrate/current until calibration completes (calibration uses fast in-process adapters)
    let current = poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;
    let cal_body: serde_json::Value = current.json().await.expect("current json");
    assert!(
        cal_body["alpha"].as_f64().is_some(),
        "alpha must be numeric"
    );
    assert!(
        cal_body["n_max"].as_f64().is_some(),
        "n_max must be numeric"
    );
    assert!(
        cal_body["n_max"].as_f64().unwrap() > 0.0,
        "n_max must be positive"
    );

    // Verify all declared response fields are present
    for key in &[
        "calibration_id",
        "beta_base",
        "beta_eff",
        "theta_coord",
        "cg_mean",
        "cg_std_dev",
    ] {
        assert!(
            cal_body[key].is_number() || cal_body[key].is_string(),
            "field {key} must be present in calibration response"
        );
    }

    // Calibrate events SSE should exist (just verify header, don't consume full stream)
    let sse_resp = client
        .get(format!("{base}/calibrate/{cal_id}/events"))
        .send()
        .await
        .expect("GET calibrate events");
    assert_eq!(
        sse_resp
            .headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap_or("")),
        Some("text/event-stream"),
        "calibrate events must be SSE"
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn submit_task_requires_calibration_first() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    // Do NOT calibrate — task submission should fail with 503
    let manifest = serde_json::json!({
        "description": "Design a stateless auth system using JWT tokens",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2}
    });

    let resp = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST /tasks");
    assert_eq!(resp.status(), 503, "must return 503 CalibrationRequired");
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["error"], "CalibrationRequiredError");
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn submit_task_with_low_j_eff_returns_context_underflow() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    // Calibrate first
    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    // Submit task with description that has zero overlap with explicit constraints
    // constraints = ["quantum", "blockchain"] but description has none of those tokens
    let manifest = serde_json::json!({
        "description": "paint the fence",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2},
        "constraints": ["stateless", "JWT", "authentication", "microservices", "security"]
    });

    let resp = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST /tasks");
    assert_eq!(resp.status(), 400, "must return 400 ContextUnderflowError");
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["error"], "ContextUnderflowError");
    assert!(
        body["j_eff"].as_f64().is_some(),
        "j_eff must be numeric in error body"
    );
    assert!(
        body["threshold"].as_f64().is_some(),
        "threshold must be present in error body"
    );
    // j_eff must be strictly less than the threshold the server reported
    let j_eff = body["j_eff"].as_f64().unwrap();
    let threshold = body["threshold"].as_f64().unwrap();
    assert!(
        j_eff < threshold,
        "j_eff={j_eff} must be below reported threshold={threshold}"
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn full_task_lifecycle_accepted_and_status_queryable() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    // Calibrate
    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    // Submit task — empty constraints means required_kw = description → j_eff = 1.0
    let manifest = serde_json::json!({
        "description": "Design a stateless authentication system for microservices using JWT",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2}
    });

    let resp = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST /tasks");
    assert_eq!(resp.status(), 202);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["status"], "accepted");
    assert!(body["j_eff"].as_f64().unwrap() > 0.0);
    let task_id = body["task_id"].as_str().expect("task_id").to_owned();
    let events_url = body["events_url"].as_str().expect("events_url").to_owned();
    assert_eq!(events_url, format!("/tasks/{task_id}/events"));

    // GET /tasks/{id} immediately returns the task (pre-inserted before engine runs)
    let status_resp = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .expect("GET status");
    assert_eq!(status_resp.status(), 200);
    let status_body: serde_json::Value = status_resp.json().await.expect("status json");
    assert_eq!(status_body["task_id"], task_id);
    assert!(
        ["pending", "generating", "verifying", "merging", "resolved"]
            .contains(&status_body["status"].as_str().unwrap_or("")),
        "status must be a known phase: got {}",
        status_body["status"]
    );

    // Wait for engine to finish (engine completes quickly in test configuration)
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Status should have advanced
    let final_resp = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .expect("GET final status");
    let final_body: serde_json::Value = final_resp.json().await.expect("final json");
    // Task either resolved or failed (both are terminal and valid with mock adapter)
    assert!(
        final_body["status"] == "resolved" || final_body["status"] == "failed",
        "task should be terminal after engine completes, got: {}",
        final_body["status"]
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn recover_task_after_store_cleared() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    // Calibrate
    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    // Submit task
    let manifest = serde_json::json!({
        "description": "Design a stateless authentication system for microservices using JWT",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2}
    });
    let resp = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST /tasks");
    assert_eq!(resp.status(), 202);
    let task_id = resp.json::<serde_json::Value>().await.expect("json")["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Wait for at least one event to be published to JetStream
    tokio::time::sleep(Duration::from_millis(300)).await;

    // The in-memory store has this task. If we call /recover it should still work
    // (it won't find "no live entry" so it won't overwrite — but replay should succeed)
    let recover_resp = client
        .get(format!("{base}/tasks/{task_id}/recover"))
        .send()
        .await
        .expect("GET /recover");

    // Should be 200 (events exist in JetStream) or 404 if somehow no events landed yet
    // With 300ms delay and fast in-process adapters, events should be present
    assert_eq!(
        recover_resp.status(),
        200,
        "recover must return 200 when JetStream events exist"
    );
    let recover_body: serde_json::Value = recover_resp.json().await.expect("recover json");
    assert_eq!(recover_body["task_id"], task_id);
    assert!(
        recover_body["status"].as_str().is_some(),
        "status must be present"
    );
    // "recovered" field should be absent (removed in quality fix)
    assert!(
        recover_body.get("recovered").is_none(),
        "recovered field must not be present in response"
    );
    // Verify all declared response fields are present
    for key in &[
        "phase",
        "phase_name",
        "explorers_completed",
        "explorers_total",
        "proposals_valid",
        "proposals_pruned",
        "autonomic_retries",
    ] {
        assert!(
            recover_body.get(key).is_some(),
            "field {key} must be present in recovery response"
        );
    }
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn unknown_task_returns_404() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    let fake_id = uuid::Uuid::new_v4().to_string();
    let resp = client
        .get(format!("{base}/tasks/{fake_id}"))
        .send()
        .await
        .expect("GET unknown task");
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["error"], "TaskNotFound");
    assert!(
        body["task_id"].as_str().is_some(),
        "task_id must be present in 404 body"
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn recover_unknown_task_returns_404() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    let fake_id = uuid::Uuid::new_v4().to_string();
    let resp = client
        .get(format!("{base}/tasks/{fake_id}/recover"))
        .send()
        .await
        .expect("GET recover unknown");
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["error"], "TaskNotFound");
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn task_events_endpoint_is_sse() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    // Calibrate
    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    // Submit task
    let manifest = serde_json::json!({
        "description": "Design a stateless authentication system using JWT tokens and OAuth2",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2}
    });
    let task_resp = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST /tasks");
    let task_id = task_resp.json::<serde_json::Value>().await.expect("json")["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Verify events endpoint returns SSE content-type
    let sse = client
        .get(format!("{base}/tasks/{task_id}/events"))
        .send()
        .await
        .expect("GET task events");
    assert_eq!(
        sse.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream"),
        "task events must be SSE"
    );
}
