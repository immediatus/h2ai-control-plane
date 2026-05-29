#![allow(clippy::missing_panics_doc)]
//! Tests for `GET /:tenant_id/tasks/:task_id/recover` route.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use h2ai_api::{routes::task_router, state::AppState};
use h2ai_config::H2AIConfig;
use h2ai_test_utils::decomposition_adapter;
use tower::ServiceExt;

fn make_state() -> AppState {
    let adapter = Arc::new(decomposition_adapter("mock"));
    AppState::new_for_tests(
        H2AIConfig::default(),
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    )
}

fn make_router(state: AppState) -> Router {
    task_router().with_state(state)
}

// ── invalid UUID ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn recover_returns_400_for_invalid_uuid() {
    let app = make_router(make_state());
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/default/tasks/not-a-uuid/recover")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── no journal entries → 404 ──────────────────────────────────────────────────

#[tokio::test]
async fn recover_returns_404_when_no_journal_entries() {
    // No NATS → journal.replay() returns None → TaskNotFound.
    let app = make_router(make_state());
    let task_id = uuid::Uuid::new_v4();
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/default/tasks/{task_id}/recover"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
