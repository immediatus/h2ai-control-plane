#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Tests for `h2ai_api::routes::admin` — `reset_response_body_value` helper
//! and the `POST /:tenant_id/admin/reset-experiment-state` HTTP handler.
//!
//! Uses `AppState::new_for_tests()` and Axum oneshot so no NATS server is required.

use std::sync::Arc;

use axum::{body::Body, http::Request, Router};
use h2ai_api::{routes::admin::reset_response_body_value, routes::admin_router, state::AppState};
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
    admin_router().with_state(state)
}

// ── reset_response_body_value pure-function tests ────────────────────────────

#[test]
fn reset_response_body_value_contains_tenant_id() {
    let body = reset_response_body_value("test-tenant");
    assert_eq!(body["tenant_id"], "test-tenant");
    assert_eq!(body["reset"], true);
}

#[test]
fn reset_response_body_value_fields_reset_is_nonempty() {
    let body = reset_response_body_value("t");
    let arr = body["fields_reset"]
        .as_array()
        .expect("fields_reset must be a JSON array");
    assert!(!arr.is_empty(), "fields_reset must list at least one field");
}

// ── HTTP handler tests ────────────────────────────────────────────────────────

#[tokio::test]
async fn reset_experiment_state_returns_200() {
    let app = make_router(make_state());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/default/admin/reset-experiment-state")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let body: serde_json::Value =
        serde_json::from_slice(&bytes).expect("response must be valid JSON");

    assert_eq!(body["reset"], true);
    assert_eq!(body["tenant_id"], "default");
}

#[tokio::test]
async fn reset_experiment_state_returns_tenant_id_in_body() {
    let app = make_router(make_state());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/acme-corp/admin/reset-experiment-state")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let body: serde_json::Value =
        serde_json::from_slice(&bytes).expect("response must be valid JSON");

    assert_eq!(body["tenant_id"], "acme-corp");
    assert_eq!(body["reset"], true);
    let arr = body["fields_reset"]
        .as_array()
        .expect("fields_reset must be a JSON array");
    assert!(!arr.is_empty());
}
