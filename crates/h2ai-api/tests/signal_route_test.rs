#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Tests for `POST /:tenant_id/tasks/:task_id/signal` — no NATS required.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use h2ai_api::{routes::task_router, state::AppState};
use h2ai_config::H2AIConfig;
use h2ai_orchestrator::task_store::TaskState;
use h2ai_test_utils::DecompositionMockAdapter;
use h2ai_types::identity::{TaskId, TenantId};
use serde_json::{json, Value};
use tower::ServiceExt;

fn make_state() -> AppState {
    let adapter = Arc::new(DecompositionMockAdapter::new("mock".into()));
    AppState::new_for_tests(
        H2AIConfig::default(),
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    )
}

fn make_router(state: AppState) -> Router {
    task_router().with_state(state)
}

async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

// ── 404 for unknown task ──────────────────────────────────────────────────────

#[tokio::test]
async fn signal_returns_404_for_unknown_task() {
    let app = make_router(make_state());
    let unknown = uuid::Uuid::new_v4();
    let body =
        json!({"payload": {"kind": "Approve", "data": {"operator_id": "op1", "approved": true}}});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/default/tasks/{unknown}/signal"))
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── 400 for invalid UUID ──────────────────────────────────────────────────────

#[tokio::test]
async fn signal_returns_400_for_invalid_task_id() {
    let app = make_router(make_state());
    let body =
        json!({"payload": {"kind": "Approve", "data": {"operator_id": "op1", "approved": true}}});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/default/tasks/not-a-uuid/signal")
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── 400 for empty operator_id in Approve ─────────────────────────────────────

#[tokio::test]
async fn signal_returns_400_for_empty_operator_id() {
    let state = make_state();
    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();
    state
        .store
        .insert(task_id.clone(), TaskState::new(task_id.clone(), tenant_id));

    let app = make_router(state);
    let body =
        json!({"payload": {"kind": "Approve", "data": {"operator_id": "  ", "approved": true}}});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/default/tasks/{task_id}/signal"))
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let b = body_json(resp).await;
    assert!(b["message"].as_str().unwrap_or("").contains("operator_id"));
}

// ── 202 for valid WaveContinue signal (no NATS → skipped publish) ─────────────

#[tokio::test]
async fn signal_returns_202_for_active_task_wave_continue() {
    let state = make_state();
    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();
    state
        .store
        .insert(task_id.clone(), TaskState::new(task_id.clone(), tenant_id));

    let app = make_router(state);
    let body = json!({"payload": {"kind": "WaveContinue", "data": {"n_override": null}}});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/default/tasks/{task_id}/signal"))
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let b = body_json(resp).await;
    assert_eq!(b["status"], "signal_queued");
}

// ── 202 for Unknown signal kind (falls through to queued) ────────────────────

#[tokio::test]
async fn signal_returns_202_for_unknown_kind_active_task() {
    let state = make_state();
    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();
    state
        .store
        .insert(task_id.clone(), TaskState::new(task_id.clone(), tenant_id));

    let app = make_router(state);
    // "NotAKnownKind" hits #[serde(other)] → SignalPayloadDto::Unknown → SignalPayload::Unknown
    let body = json!({"payload": {"kind": "NotAKnownKind"}});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/default/tasks/{task_id}/signal"))
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let b = body_json(resp).await;
    assert_eq!(b["status"], "signal_queued");
}

// ── 202 already_resolved for inactive task ────────────────────────────────────

#[tokio::test]
async fn signal_returns_202_already_resolved_for_resolved_task() {
    let state = make_state();
    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();
    state
        .store
        .insert(task_id.clone(), TaskState::new(task_id.clone(), tenant_id));
    state.store.mark_resolved(&task_id);

    let app = make_router(state);
    let body =
        json!({"payload": {"kind": "Approve", "data": {"operator_id": "op1", "approved": true}}});

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/default/tasks/{task_id}/signal"))
                .header("Content-Type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let b = body_json(resp).await;
    assert_eq!(b["status"], "already_resolved");
}
