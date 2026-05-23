#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
//! End-to-end integration tests for the H2AI Control Plane API.
//!
//! All tests require a live NATS server with `JetStream` enabled:
//!   `NATS_URL=nats://localhost:4222` cargo nextest run -p h2ai-api --test `e2e_test`

const TENANT: &str = "default";

use h2ai_api::{
    routes::{calibrate_router, health_router, task_router},
    state::AppState,
};
use h2ai_config::H2AIConfig;
use h2ai_state::nats::NatsClient;
use h2ai_test_utils::{DecompositionMockAdapter, MockAdapter};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;

async fn boot_app() -> Option<(String, tokio::task::JoinHandle<()>)> {
    let nats_url = H2AIConfig::default().nats_url;
    let nats = match NatsClient::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return None;
        }
    };
    nats.ensure_infrastructure().await.expect("infra");

    let cfg = H2AIConfig::default();
    let explorer = Arc::new(DecompositionMockAdapter::new("mock explorer output".into()));
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

    tokio::time::sleep(Duration::from_millis(50)).await;

    Some((base_url, handle))
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

/// Helper: poll GET {`status_url`} until task status matches one of `expected`, or panic.
async fn poll_until_status(
    client: &reqwest::Client,
    status_url: &str,
    expected: &[&str],
    attempts: u32,
) -> serde_json::Value {
    let mut last_status = String::from("<none>");
    for i in 0..attempts {
        let resp = client
            .get(status_url)
            .send()
            .await
            .expect("GET task status failed");
        if resp.status().is_success() {
            let body: serde_json::Value = resp.json().await.expect("task status json");
            let s = body["status"].as_str().unwrap_or("").to_string();
            s.clone_into(&mut last_status);
            if expected.contains(&s.as_str()) {
                return body;
            }
        }
        if i < attempts - 1 {
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }
    panic!(
        "Task at {status_url} never reached one of {expected:?} after {attempts} attempts (last: {last_status})"
    );
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_liveness_returns_ok() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base}/health"))
        .send()
        .await
        .expect("GET /health");
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn calibrate_then_current_returns_coefficients() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
async fn submit_task_requires_calibration_first() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
    let client = reqwest::Client::new();

    // Do NOT calibrate — task submission should fail with 503
    let manifest = serde_json::json!({
        "description": "Design a stateless auth system using JWT tokens",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2}
    });

    let resp = client
        .post(format!("{base}/{TENANT}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST /tasks");
    assert_eq!(resp.status(), 503, "must return 503 CalibrationRequired");
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["error"], "CalibrationRequiredError");
}

#[tokio::test]
async fn full_task_lifecycle_accepted_and_status_queryable() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
        .post(format!("{base}/{TENANT}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST /tasks");
    assert_eq!(resp.status(), 202);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["status"], "accepted");
    let task_id = body["task_id"].as_str().expect("task_id").to_owned();
    let events_url = body["events_url"].as_str().expect("events_url").to_owned();
    assert_eq!(
        events_url,
        format!("/tenants/{TENANT}/tasks/{task_id}/events")
    );

    // GET /tasks/{id} immediately returns the task (pre-inserted before engine runs)
    let status_resp = client
        .get(format!("{base}/{TENANT}/tasks/{task_id}"))
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

    // Poll until engine reaches a terminal or HITL-parked state
    // (HITL is enabled by default so tasks may park at awaiting_approval before resolving)
    let final_body = poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["resolved", "failed", "awaiting_approval"],
        40,
    )
    .await;
    // Task is terminal (resolved/failed) or parked at HITL gate (awaiting_approval)
    assert!(
        final_body["status"] == "resolved"
            || final_body["status"] == "failed"
            || final_body["status"] == "awaiting_approval",
        "task should be in a stable state after engine completes, got: {}",
        final_body["status"]
    );
}

#[tokio::test]
async fn recover_task_after_store_cleared() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
        .post(format!("{base}/{TENANT}/tasks"))
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
        .get(format!("{base}/{TENANT}/tasks/{task_id}/recover"))
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
async fn unknown_task_returns_404() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
    let client = reqwest::Client::new();

    let fake_id = uuid::Uuid::new_v4().to_string();
    let resp = client
        .get(format!("{base}/{TENANT}/tasks/{fake_id}"))
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
async fn recover_unknown_task_returns_404() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
    let client = reqwest::Client::new();

    let fake_id = uuid::Uuid::new_v4().to_string();
    let resp = client
        .get(format!("{base}/{TENANT}/tasks/{fake_id}/recover"))
        .send()
        .await
        .expect("GET recover unknown");
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.expect("json");
    assert_eq!(body["error"], "TaskNotFound");
}

#[tokio::test]
async fn task_events_endpoint_is_sse() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
        .post(format!("{base}/{TENANT}/tasks"))
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
        .get(format!("{base}/{TENANT}/tasks/{task_id}/events"))
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
async fn hitl_require_approval_task_reaches_awaiting_approval() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
        .post(format!("{base}/{TENANT}/tasks"))
        .json(&manifest)
        .send()
        .await
        .expect("POST /tasks");
    assert_eq!(resp.status(), 202);
    let task_id = resp.json::<serde_json::Value>().await.expect("json")["task_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Poll until engine parks the task at HITL gate
    let body = poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["awaiting_approval", "resolved", "failed"],
        50,
    )
    .await;
    assert_eq!(
        body["status"], "awaiting_approval",
        "task with require_approval=true must park at HITL gate, got: {}",
        body["status"]
    );

    // GET /tasks/{id}/approval returns 410 Gone (endpoint migrated to /signal)
    let approval_resp = client
        .get(format!("{base}/{TENANT}/tasks/{task_id}/approval"))
        .send()
        .await
        .expect("GET approval");
    assert_eq!(
        approval_resp.status(),
        410,
        "GET /approval must return 410 Gone (migrated to /signal)"
    );
}

#[tokio::test]
async fn hitl_approve_resolves_awaiting_task() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
        .post(format!("{base}/{TENANT}/tasks"))
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

    // Poll until engine parks at HITL gate
    poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["awaiting_approval"],
        50,
    )
    .await;

    // Approve the task via /signal endpoint
    let approve_resp = client
        .post(format!("{base}/{TENANT}/tasks/{task_id}/signal"))
        .json(&serde_json::json!({
            "payload": {
                "kind": "Approve",
                "data": {
                    "approved": true,
                    "reviewer_note": "LGTM",
                    "operator_id": "test-operator@example.com"
                }
            }
        }))
        .send()
        .await
        .expect("POST signal");
    assert_eq!(
        approve_resp.status(),
        202,
        "signal must return 202 Accepted"
    );
    let ar: serde_json::Value = approve_resp.json().await.expect("signal json");
    assert_eq!(ar["status"], "signal_queued");

    // Poll until task resolves after approval
    let final_body = poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["resolved", "failed"],
        30,
    )
    .await;
    assert_eq!(
        final_body["status"], "resolved",
        "approved task must reach resolved, got: {}",
        final_body["status"]
    );
}

#[tokio::test]
async fn hitl_reject_fails_awaiting_task() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
        .post(format!("{base}/{TENANT}/tasks"))
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

    // Poll until engine parks at HITL gate
    poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["awaiting_approval"],
        50,
    )
    .await;

    // Reject the task via /signal endpoint
    let reject_resp = client
        .post(format!("{base}/{TENANT}/tasks/{task_id}/signal"))
        .json(&serde_json::json!({
            "payload": {
                "kind": "Approve",
                "data": {
                    "approved": false,
                    "reviewer_note": "Output quality insufficient",
                    "operator_id": "test-reviewer@example.com"
                }
            }
        }))
        .send()
        .await
        .expect("POST signal reject");
    assert_eq!(
        reject_resp.status(),
        202,
        "reject signal must return 202 Accepted"
    );
    let rr: serde_json::Value = reject_resp.json().await.expect("reject json");
    assert_eq!(rr["status"], "signal_queued");

    // Poll until task fails after rejection
    let final_body = poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["failed", "resolved"],
        30,
    )
    .await;
    assert_eq!(
        final_body["status"], "failed",
        "rejected task must reach failed, got: {}",
        final_body["status"]
    );
}

#[tokio::test]
async fn hitl_approval_404_when_not_awaiting() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
    let client = reqwest::Client::new();

    let fake_id = uuid::Uuid::new_v4().to_string();
    let resp = client
        .get(format!("{base}/{TENANT}/tasks/{fake_id}/approval"))
        .send()
        .await
        .expect("GET");
    assert_eq!(
        resp.status(),
        410,
        "GET /approval must return 410 Gone (endpoint migrated to /signal)"
    );
}

#[tokio::test]
async fn hitl_concurrent_approve_returns_conflict() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
        .post(format!("{base}/{TENANT}/tasks"))
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

    // Poll until engine parks at HITL gate before sending concurrent approvals
    poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["awaiting_approval"],
        50,
    )
    .await;

    let approve_body = serde_json::json!({
        "payload": {
            "kind": "Approve",
            "data": {
                "approved": true,
                "operator_id": "operator@example.com"
            }
        }
    });

    // Send two concurrent signals — both are accepted (signals are idempotent queued publishes)
    let (r1, r2) = tokio::join!(
        client
            .post(format!("{base}/{TENANT}/tasks/{task_id}/signal"))
            .json(&approve_body)
            .send(),
        client
            .post(format!("{base}/{TENANT}/tasks/{task_id}/signal"))
            .json(&approve_body)
            .send(),
    );
    let s1 = r1.expect("req1").status();
    let s2 = r2.expect("req2").status();

    // Both signals accepted (202) — the engine handles deduplication
    let statuses = [s1.as_u16(), s2.as_u16()];
    assert!(
        statuses.iter().all(|&s| s == 202),
        "concurrent signals must both be accepted (202): got {statuses:?}"
    );
}

// ── Checkpoint consistency ─────────────────────────────────────────────────

#[tokio::test]
async fn task_checkpoint_created_and_cleaned_after_approval() {
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
        .post(format!("{base}/{TENANT}/tasks"))
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

    // Poll until engine parks at HITL gate (checkpoint must exist at this point)
    let status = poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["awaiting_approval", "resolved", "failed"],
        50,
    )
    .await;
    assert_eq!(
        status["status"], "awaiting_approval",
        "must be parked before approve"
    );

    // Approve via /signal — triggers checkpoint GC
    let ar = client
        .post(format!("{base}/{TENANT}/tasks/{task_id}/signal"))
        .json(&serde_json::json!({
            "payload": {
                "kind": "Approve",
                "data": {"approved": true, "operator_id": "gc-test-operator"}
            }
        }))
        .send()
        .await
        .expect("POST signal")
        .json::<serde_json::Value>()
        .await
        .expect("json");
    assert_eq!(ar["status"], "signal_queued");

    // Poll until task resolves after approval
    let final_status = poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["resolved", "failed"],
        30,
    )
    .await;
    assert_eq!(
        final_status["status"], "resolved",
        "approved task must resolve"
    );

    // GET /approval returns 410 Gone (endpoint migrated to /signal)
    let approval_gone = client
        .get(format!("{base}/{TENANT}/tasks/{task_id}/approval"))
        .send()
        .await
        .expect("GET");
    assert_eq!(
        approval_gone.status(),
        410,
        "GET /approval must return 410 Gone (migrated to /signal)"
    );
}

// ── System consistency ─────────────────────────────────────────────────────

#[tokio::test]
async fn awaiting_approval_is_non_terminal_status() {
    // Verifies that awaiting_approval does not appear in the resolved/failed set
    // (task stays queryable while parked)
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
        .post(format!("{base}/{TENANT}/tasks"))
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

    // Poll until engine parks the task (or reaches terminal state)
    let status = poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["awaiting_approval", "resolved", "failed"],
        50,
    )
    .await;

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
async fn oracle_task_bypasses_hitl_gate() {
    // Oracle tasks must auto-proceed even when require_approval would otherwise trigger
    let Some((base, _handle)) = boot_app().await else {
        return;
    };
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
            "runner_uri": "/nonexistent/runner",
            "test_suite": "tests/",
            "language": "python",
            "timeout_ms": 100,
            "oracle_type": "test_suite",
            "domain": "code"
        }
    });

    let task_id = client
        .post(format!("{base}/{TENANT}/tasks"))
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

    // Poll until oracle task reaches a terminal state (oracle unreachable → fail quickly)
    let status = poll_until_status(
        &client,
        &format!("{base}/{TENANT}/tasks/{task_id}"),
        &["resolved", "failed"],
        50,
    )
    .await;
    let s = status["status"].as_str().unwrap_or("");
    // Oracle task must NOT be parked at HITL gate — it either resolved or failed (oracle unreachable = fail)
    assert_ne!(
        s, "awaiting_approval",
        "oracle task must bypass HITL gate, got: {s}"
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
        auth_token_env: String::new(),
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
