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
//! HTTP route handler tests for `crates/h2ai-api/src/routes/tasks.rs`.
//!
//! These tests exercise the HTTP surface of the task routes without a live NATS
//! server. `AppState::new_for_tests()` wires an `InMemoryStateBackend` so all
//! persistence and event-publishing traits are in-process.
//!
//! Handlers that require NATS at the async-task level (e.g. the engine spawned by
//! `submit_task`) are NOT followed into NATS-dependent code paths. The tests
//! verify only the synchronous HTTP response — 202, 404, 400, 503, etc.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use chrono::Utc;
use h2ai_api::{routes::task_router, state::AppState};
use h2ai_config::H2AIConfig;
use h2ai_orchestrator::task_store::TaskState;
use h2ai_test_utils::decomposition_adapter;
use h2ai_types::{
    events::{CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode},
    identity::{TaskId, TenantId},
    sizing::{CoherencyCoefficients, CoordinationThreshold},
};
use serde_json::{json, Value};
use tower::ServiceExt;

// ── helpers ───────────────────────────────────────────────────────────────────

fn synthetic_calibration() -> CalibrationCompletedEvent {
    let coefficients = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71])
        .expect("valid coefficients");
    let coordination_threshold = CoordinationThreshold::from_calibration(&coefficients, 0.3);
    CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients,
        coordination_threshold,
        ensemble: None,
        eigen: None,
        timestamp: Utc::now(),
        pairwise_beta: None,
        cg_mode: CgMode::default(),
        adapter_families: vec!["Mock".into()],
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: CalibrationQuality::default(),
        calibration_source: CalibrationSource::Measured,
        beta_quality: None,
    }
}

fn make_state() -> AppState {
    let adapter = Arc::new(decomposition_adapter("mock response"));
    AppState::new_for_tests(
        H2AIConfig::default(),
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    )
}

fn make_router(state: AppState) -> Router {
    task_router().with_state(state)
}

/// Decode the response body as a JSON `Value`. Panics on failure.
async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

/// Build a minimal valid `TaskManifest` JSON body.
fn valid_manifest_json() -> Value {
    json!({
        "description": "Propose a stateless authentication design",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2, "tau_min": 0.2, "tau_max": 0.9}
    })
}

// ── submit_task ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn submit_task_returns_202_with_task_id() {
    let state = make_state();

    // Inject calibration so the handler can proceed.
    let ts = state.tenant_state(&TenantId::default_tenant());
    *ts.calibration.write().await = Some(synthetic_calibration());

    let app = make_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/default/tasks")
                .header("Content-Type", "application/json")
                .body(Body::from(valid_manifest_json().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::ACCEPTED);

    let body = body_json(response).await;
    assert!(
        body.get("task_id").is_some(),
        "response body must contain task_id; got: {body}"
    );
    assert_eq!(body["status"], "accepted");
}

#[tokio::test]
async fn submit_task_returns_calibration_required_without_calibration() {
    // No calibration injected — handler must return an error (503 CalibrationRequired).
    let state = make_state();
    let app = make_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/default/tasks")
                .header("Content-Type", "application/json")
                .body(Body::from(valid_manifest_json().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(
        response.status().as_u16() >= 400,
        "expected non-2xx for missing calibration; got {}",
        response.status()
    );
    let body = body_json(response).await;
    assert_eq!(
        body["error"], "CalibrationRequiredError",
        "expected CalibrationRequiredError; got: {body}"
    );
}

#[tokio::test]
async fn submit_task_returns_400_on_invalid_pareto_weights() {
    // Weights sum to 1.5 — validation must reject before reaching calibration check.
    let state = make_state();
    let ts = state.tenant_state(&TenantId::default_tenant());
    *ts.calibration.write().await = Some(synthetic_calibration());

    let app = make_router(state);

    let bad_manifest = json!({
        "description": "test",
        "pareto_weights": {"diversity": 0.5, "containment": 0.5, "throughput": 0.5},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2, "tau_min": 0.2, "tau_max": 0.9}
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/default/tasks")
                .header("Content-Type", "application/json")
                .body(Body::from(bad_manifest.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "weights not summing to 1.0 must return 400"
    );
}

// ── task_status ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn task_status_returns_404_for_unknown_task() {
    let state = make_state();
    let app = make_router(state);

    // Use a valid UUID that is not in the store.
    let unknown_id = uuid::Uuid::new_v4();
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/default/tasks/{unknown_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn task_status_returns_200_for_known_task() {
    let state = make_state();

    // Pre-insert a task directly into the store.
    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();
    state.store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), tenant_id.clone()),
    );

    let app = make_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/default/tasks/{task_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    assert_eq!(
        body["task_id"],
        task_id.to_string(),
        "status body must echo task_id"
    );
    assert!(body.get("status").is_some(), "status field missing");
}

// ── merge_task ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn merge_task_returns_404_for_unknown_task() {
    let state = make_state();
    let app = make_router(state);

    let unknown_id = uuid::Uuid::new_v4();
    let body = json!({
        "resolution": "select",
        "selected_proposals": ["proposal A"],
        "final_output": "final answer"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/default/tasks/{unknown_id}/merge"))
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ── clarify_task ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn clarify_task_returns_not_found_for_unknown_task() {
    let state = make_state();
    let app = make_router(state);

    let unknown_id = uuid::Uuid::new_v4();
    let body = json!({"answer": "Yes, use OAuth2."});

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/default/tasks/{unknown_id}/clarify"))
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn clarify_task_returns_not_found_when_no_pending_waiter() {
    let state = make_state();

    // Insert the task so ownership check passes, but no waiter is registered.
    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();
    state.store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), tenant_id.clone()),
    );

    let app = make_router(state);

    let body = json!({"answer": "Use JWT."});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/default/tasks/{task_id}/clarify"))
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Task exists but no pending clarification waiter — should be 404.
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let resp_body = body_json(response).await;
    assert!(
        resp_body["error"]
            .as_str()
            .unwrap_or("")
            .contains("no pending clarification"),
        "unexpected error: {resp_body}"
    );
}

// ── task_status cross-tenant isolation ────────────────────────────────────────

#[tokio::test]
async fn task_status_returns_404_for_wrong_tenant() {
    let state = make_state();

    // Insert task under "default" tenant.
    let task_id = TaskId::new();
    state.store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), TenantId::default_tenant()),
    );

    let app = make_router(state);

    // Request the task under a different tenant — must return 404.
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/other_tenant/tasks/{task_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::NOT_FOUND,
        "cross-tenant access must be denied"
    );
}

// ── ExplorerBudgetExceeded ────────────────────────────────────────────────────

#[tokio::test]
async fn submit_task_returns_error_when_explorer_count_exceeds_n_max() {
    let state = make_state();
    let ts = state.tenant_state(&TenantId::default_tenant());
    *ts.calibration.write().await = Some(synthetic_calibration());
    let app = make_router(state);

    // n_max ≈ 12 for synthetic_calibration; request 20 explorers to exceed it.
    let manifest = serde_json::json!({
        "description": "High explorer count task",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 20, "tau_min": 0.2, "tau_max": 0.9}
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/default/tasks")
                .header("Content-Type", "application/json")
                .body(Body::from(manifest.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "explorer count exceeding n_max must return 400"
    );
}

// ── ServiceUnavailable (semaphore exhausted) ──────────────────────────────────

#[tokio::test]
async fn submit_task_returns_503_when_at_capacity() {
    let cfg = H2AIConfig {
        max_concurrent_tasks: 0, // no permits → immediately at capacity
        ..H2AIConfig::default()
    };
    let adapter = Arc::new(decomposition_adapter("mock response"));
    let state = AppState::new_for_tests(
        cfg,
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    );
    let ts = state.tenant_state(&TenantId::default_tenant());
    *ts.calibration.write().await = Some(synthetic_calibration());
    let app = make_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/default/tasks")
                .header("Content-Type", "application/json")
                .body(Body::from(valid_manifest_json().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "server at capacity must return 503"
    );
}

// ── task_status invalid UUID ──────────────────────────────────────────────────

#[tokio::test]
async fn task_status_returns_400_for_invalid_uuid() {
    let app = make_router(make_state());
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/default/tasks/not-a-valid-uuid")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "malformed UUID must return 400"
    );
}

// ── merge_task: already resolved ──────────────────────────────────────────────

#[tokio::test]
async fn merge_task_returns_409_for_already_resolved_task() {
    let state = make_state();

    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();
    state.store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), tenant_id.clone()),
    );
    state.store.mark_resolved(&task_id);

    let app = make_router(state);

    let body = json!({
        "resolution": "select",
        "selected_proposals": ["answer"],
        "final_output": "the final answer"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/default/tasks/{task_id}/merge"))
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::CONFLICT,
        "re-merging a resolved task must return 409"
    );
}

// ── pareto_weights boundary ───────────────────────────────────────────────────

#[tokio::test]
async fn submit_task_accepts_weights_at_tolerance_boundary() {
    let state = make_state();
    let ts = state.tenant_state(&TenantId::default_tenant());
    *ts.calibration.write().await = Some(synthetic_calibration());

    let app = make_router(state);

    // Weights sum to 1.00009 — within 1e-4 tolerance → must accept.
    let manifest = json!({
        "description": "boundary test",
        "pareto_weights": {"diversity": 0.33337, "containment": 0.33336, "throughput": 0.33336},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2, "tau_min": 0.2, "tau_max": 0.9}
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/default/tasks")
                .header("Content-Type", "application/json")
                .body(Body::from(manifest.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::ACCEPTED,
        "weights within 1e-4 tolerance must be accepted; got: {}",
        response.status()
    );
}
