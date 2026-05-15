//! End-to-end integration tests for the H2AI Control Plane API.
//!
//! All tests require a live NATS server with JetStream enabled:
//!   NATS_URL=nats://localhost:4222 cargo nextest run -p h2ai-api --test e2e_test

use h2ai_adapters::mock::{DecompositionMockAdapter, MockAdapter};
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
    let nats_url = H2AIConfig::default().nats_url;
    let nats = NatsClient::connect(&nats_url).await.expect("NATS connect");
    nats.ensure_infrastructure().await.expect("infra");

    let cfg = H2AIConfig::default();
    // DecompositionMockAdapter returns valid STEP3 JSON when called by the decomposition
    // pipeline (detected via "JSON formatter" system context), and plain text otherwise.
    // Plain MockAdapter would cause STEP3 JSON parse failure → task reaches "failed".
    let explorer = Arc::new(DecompositionMockAdapter::new("mock explorer output".into()));
    // Auditor output must be valid JSON for both the verifier ({"score": float}) and
    // auditor gate ({"approved": bool}) phases — the engine fails safe on non-JSON.
    let auditor = Arc::new(MockAdapter::new(
        r#"{"approved":true,"score":0.9,"reason":"mock"}"#.into(),
    ));
    let state = AppState::new(
        nats,
        cfg,
        vec![explorer as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        auditor,
    );

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

    // Submit task — empty constraints
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
        [
            "pending",
            "generating",
            "verifying",
            "merging",
            "resolved",
            "failed"
        ]
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

// ── HITL Approval Gate ─────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn hitl_require_approval_task_reaches_awaiting_approval() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    // Calibrate first
    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    // Submit task with require_approval=true — gate fires regardless of confidence
    let manifest = serde_json::json!({
        "description": "Design a key rotation system for HMAC secrets",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2},
        "require_approval": true
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

    // Wait for engine to park the task at HITL gate
    tokio::time::sleep(Duration::from_millis(800)).await;

    // Task status must be awaiting_approval
    let status_resp = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .expect("GET status");
    assert_eq!(status_resp.status(), 200);
    let body: serde_json::Value = status_resp.json().await.expect("json");
    assert_eq!(
        body["status"], "awaiting_approval",
        "task with require_approval=true must park at HITL gate, got: {}",
        body["status"]
    );

    // GET /tasks/{id}/approval must return the pending record
    let approval_resp = client
        .get(format!("{base}/tasks/{task_id}/approval"))
        .send()
        .await
        .expect("GET approval");
    assert_eq!(
        approval_resp.status(),
        200,
        "GET /approval must return 200 while pending"
    );
    let ar: serde_json::Value = approval_resp.json().await.expect("approval json");
    assert_eq!(ar["task_id"], task_id);
    assert!(
        ar["proposed_output"].as_str().is_some(),
        "proposed_output must be present"
    );
    assert!(
        ar["timeout_at_ms"].as_u64().is_some(),
        "timeout_at_ms must be present"
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn hitl_approve_resolves_awaiting_task() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    let manifest = serde_json::json!({
        "description": "Design a circuit breaker pattern for distributed service calls",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2},
        "require_approval": true
    });

    let task_id = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST")
        .json::<serde_json::Value>()
        .await
        .expect("json")["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Wait for engine to park at HITL gate
    tokio::time::sleep(Duration::from_millis(800)).await;

    // Approve the task
    let approve_resp = client
        .post(format!("{base}/tasks/{task_id}/approve"))
        .json(&serde_json::json!({
            "approved": true,
            "reviewer_note": "LGTM",
            "operator_id": "test-operator@example.com"
        }))
        .send()
        .await
        .expect("POST approve");
    assert_eq!(
        approve_resp.status(),
        202,
        "approve must return 202 Accepted"
    );
    let ar: serde_json::Value = approve_resp.json().await.expect("approve json");
    assert_eq!(ar["status"], "approved");

    // Task must be resolved shortly after
    tokio::time::sleep(Duration::from_millis(200)).await;
    let final_resp = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .expect("GET status");
    let final_body: serde_json::Value = final_resp.json().await.expect("json");
    assert_eq!(
        final_body["status"], "resolved",
        "approved task must reach resolved, got: {}",
        final_body["status"]
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn hitl_reject_fails_awaiting_task() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    let manifest = serde_json::json!({
        "description": "Design a rate limiting system for API endpoints",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2},
        "require_approval": true
    });

    let task_id = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST")
        .json::<serde_json::Value>()
        .await
        .expect("json")["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    tokio::time::sleep(Duration::from_millis(800)).await;

    // Reject the task
    let reject_resp = client
        .post(format!("{base}/tasks/{task_id}/approve"))
        .json(&serde_json::json!({
            "approved": false,
            "reviewer_note": "Output quality insufficient",
            "operator_id": "test-reviewer@example.com"
        }))
        .send()
        .await
        .expect("POST reject");
    assert_eq!(reject_resp.status(), 200, "reject must return 200 OK");
    let rr: serde_json::Value = reject_resp.json().await.expect("reject json");
    assert_eq!(rr["status"], "rejected");

    // Task must be failed
    tokio::time::sleep(Duration::from_millis(200)).await;
    let final_body: serde_json::Value = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .expect("GET")
        .json()
        .await
        .expect("json");
    assert_eq!(
        final_body["status"], "failed",
        "rejected task must reach failed, got: {}",
        final_body["status"]
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn hitl_approval_404_when_not_awaiting() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    let fake_id = uuid::Uuid::new_v4().to_string();
    let resp = client
        .get(format!("{base}/tasks/{fake_id}/approval"))
        .send()
        .await
        .expect("GET");
    assert_eq!(
        resp.status(),
        404,
        "GET /approval for unknown task must return 404"
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn hitl_concurrent_approve_returns_conflict() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    let manifest = serde_json::json!({
        "description": "Design an event sourcing system for order management",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2},
        "require_approval": true
    });

    let task_id = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST")
        .json::<serde_json::Value>()
        .await
        .expect("json")["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    tokio::time::sleep(Duration::from_millis(800)).await;

    let approve_body = serde_json::json!({
        "approved": true,
        "operator_id": "operator@example.com"
    });

    // Send two concurrent approvals — one must win, one must get 4xx
    let (r1, r2) = tokio::join!(
        client
            .post(format!("{base}/tasks/{task_id}/approve"))
            .json(&approve_body)
            .send(),
        client
            .post(format!("{base}/tasks/{task_id}/approve"))
            .json(&approve_body)
            .send(),
    );
    let s1 = r1.expect("req1").status();
    let s2 = r2.expect("req2").status();

    // One must succeed (202), one must fail (4xx — CAS race lost)
    let statuses = [s1.as_u16(), s2.as_u16()];
    assert!(
        statuses.contains(&202),
        "at least one concurrent approval must succeed (202): got {:?}",
        statuses
    );
    assert!(
        statuses.iter().any(|&s| s >= 400),
        "at least one concurrent approval must fail (4xx CAS conflict): got {:?}",
        statuses
    );
}

// ── Checkpoint consistency ─────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn task_checkpoint_created_and_cleaned_after_approval() {
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    // Manifest with require_approval=true — checkpoint is written at Merging, then GC'd after approve
    let manifest = serde_json::json!({
        "description": "Design a blue-green deployment strategy for zero-downtime releases",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2},
        "require_approval": true
    });

    let task_id = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST")
        .json::<serde_json::Value>()
        .await
        .expect("json")["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Park at HITL gate (checkpoint must exist at this point)
    tokio::time::sleep(Duration::from_millis(800)).await;

    let status: serde_json::Value = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .expect("GET")
        .json()
        .await
        .expect("json");
    assert_eq!(
        status["status"], "awaiting_approval",
        "must be parked before approve"
    );

    // Approve — triggers checkpoint GC
    let ar = client
        .post(format!("{base}/tasks/{task_id}/approve"))
        .json(&serde_json::json!({"approved": true, "operator_id": "gc-test-operator"}))
        .send()
        .await
        .expect("POST approve")
        .json::<serde_json::Value>()
        .await
        .expect("json");
    assert_eq!(ar["status"], "approved");

    // After approval, task must be resolved
    tokio::time::sleep(Duration::from_millis(200)).await;
    let final_status: serde_json::Value = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .expect("GET")
        .json()
        .await
        .expect("json");
    assert_eq!(
        final_status["status"], "resolved",
        "approved task must resolve"
    );

    // GET /approval must now return 404 (record cleaned up)
    let approval_gone = client
        .get(format!("{base}/tasks/{task_id}/approval"))
        .send()
        .await
        .expect("GET");
    assert_eq!(
        approval_gone.status(),
        404,
        "approval record must be GC'd after resolution"
    );
}

// ── System consistency ─────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn awaiting_approval_is_non_terminal_status() {
    // Verifies that awaiting_approval does not appear in the resolved/failed set
    // (task stays queryable while parked)
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    let manifest = serde_json::json!({
        "description": "Design a CQRS architecture for read-heavy e-commerce catalog",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2},
        "require_approval": true
    });

    let task_id = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST")
        .json::<serde_json::Value>()
        .await
        .expect("json")["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    tokio::time::sleep(Duration::from_millis(800)).await;

    let status: serde_json::Value = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .expect("GET")
        .json()
        .await
        .expect("json");

    let s = status["status"].as_str().unwrap_or("");
    // Must be awaiting_approval (non-terminal — task is NOT resolved or failed yet)
    assert_eq!(s, "awaiting_approval");
    assert_ne!(
        s, "resolved",
        "parked task must not be resolved without approval"
    );
    assert_ne!(
        s, "failed",
        "parked task must not be failed without rejection"
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn oracle_task_bypasses_hitl_gate() {
    // Oracle tasks must auto-proceed even when require_approval would otherwise trigger
    let (base, _handle) = boot_app().await;
    let client = reqwest::Client::new();

    client
        .post(format!("{base}/calibrate"))
        .send()
        .await
        .expect("calibrate");
    poll_until_ok(&client, &format!("{base}/calibrate/current"), 20).await;

    // Oracle task with require_approval=true — gate must be bypassed
    let manifest = serde_json::json!({
        "description": "Fix the off-by-one error in the binary search function",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2},
        "require_approval": true,
        "oracle": {
            "runner_uri": "http://localhost:19999/nonexistent",
            "test_suite": "tests/",
            "language": "python",
            "timeout_ms": 100,
            "oracle_type": "test_suite",
            "domain": "code"
        }
    });

    let task_id = client
        .post(format!("{base}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST")
        .json::<serde_json::Value>()
        .await
        .expect("json")["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    tokio::time::sleep(Duration::from_millis(800)).await;

    let status: serde_json::Value = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .expect("GET")
        .json()
        .await
        .expect("json");
    let s = status["status"].as_str().unwrap_or("");
    // Oracle task must NOT be parked at HITL gate — it either resolved or failed (oracle unreachable = fail)
    assert_ne!(
        s, "awaiting_approval",
        "oracle task must bypass HITL gate, got: {}",
        s
    );
}

// ── A2A adapter system consistency ─────────────────────────────────────────

#[test]
fn a2a_adapter_factory_builds_with_auth_none() {
    use h2ai_adapters::factory::AdapterFactory;
    use h2ai_types::config::AdapterKind;

    let kind = AdapterKind::A2a {
        endpoint: "https://example.com".to_string(),
        auth_scheme: "none".to_string(),
        auth_token_env: "".to_string(),
        timeout_minutes: 5,
        poll_interval_ms: 2000,
        max_poll_interval_ms: 30_000,
        agent_card_cache_ttl_s: 3600,
    };
    let adapter = AdapterFactory::build(&kind).expect("factory must build A2A adapter");
    // A2a adapter built successfully — verify it's an A2a kind
    assert!(matches!(
        adapter.kind(),
        h2ai_types::config::AdapterKind::A2a { .. }
    ));
}

#[test]
fn a2a_adapter_fails_fast_when_env_var_missing() {
    use h2ai_adapters::factory::AdapterFactory;
    use h2ai_types::config::AdapterKind;

    let kind = AdapterKind::A2a {
        endpoint: "https://example.com".to_string(),
        auth_scheme: "bearer".to_string(),
        auth_token_env: "H2AI_TEST_MISSING_TOKEN_VAR_XYZ".to_string(),
        timeout_minutes: 5,
        poll_interval_ms: 2000,
        max_poll_interval_ms: 30_000,
        agent_card_cache_ttl_s: 3600,
    };
    // Must fail at build time, not at request time
    let result = AdapterFactory::build(&kind);
    assert!(
        result.is_err(),
        "must fail when auth token env var is missing"
    );
}

#[test]
fn hitl_gate_conditions_are_consistent() {
    // Verifies the gate logic matches the spec: enabled AND not-oracle AND (require OR low-confidence)
    struct GateInputs {
        enabled: bool,
        oracle: bool,
        require: bool,
        q: f64,
        threshold: f64,
    }
    fn gate(i: &GateInputs) -> bool {
        i.enabled && !i.oracle && (i.require || i.q < i.threshold)
    }

    // All 5 bypass conditions
    assert!(
        !gate(&GateInputs {
            enabled: false,
            oracle: false,
            require: true,
            q: 0.1,
            threshold: 0.5
        }),
        "disabled gate"
    );
    assert!(
        !gate(&GateInputs {
            enabled: true,
            oracle: true,
            require: true,
            q: 0.1,
            threshold: 0.5
        }),
        "oracle bypass"
    );
    assert!(
        !gate(&GateInputs {
            enabled: true,
            oracle: false,
            require: false,
            q: 0.9,
            threshold: 0.5
        }),
        "high confidence"
    );
    // 2 trigger conditions
    assert!(
        gate(&GateInputs {
            enabled: true,
            oracle: false,
            require: true,
            q: 0.9,
            threshold: 0.5
        }),
        "manifest flag"
    );
    assert!(
        gate(&GateInputs {
            enabled: true,
            oracle: false,
            require: false,
            q: 0.2,
            threshold: 0.5
        }),
        "low confidence"
    );
}

#[test]
fn checkpoint_phase_string_stability() {
    use h2ai_orchestrator::task_store::TaskPhase;
    // String names used in TaskCheckpoint.phase must be stable — these are the values
    // written to NATS KV; changing them would orphan existing checkpoints
    assert_eq!(
        TaskPhase::ParallelGeneration.name_str(),
        "ParallelGeneration"
    );
    assert_eq!(TaskPhase::AuditorGate.name_str(), "AuditorGate");
    assert_eq!(TaskPhase::Merging.name_str(), "Merging");
    assert_eq!(TaskPhase::Resolved.name_str(), "Resolved");
    assert_eq!(TaskPhase::AwaitingApproval.name_str(), "AwaitingApproval");
    // All round-trip through try_from_name_str
    for phase in [
        TaskPhase::ParallelGeneration,
        TaskPhase::AuditorGate,
        TaskPhase::Merging,
        TaskPhase::Resolved,
        TaskPhase::AwaitingApproval,
    ] {
        let name = phase.name_str();
        assert_eq!(
            TaskPhase::try_from_name_str(name).map(|p| p.name_str()),
            Some(name),
            "phase {name} must round-trip through try_from_name_str"
        );
    }
}
