#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Unit tests for `h2ai_api::routes::health` — liveness, readiness, metrics.
//!
//! Uses `AppState::new_for_tests()` and Axum oneshot so no NATS server is required.

use std::sync::Arc;

use axum::{body::Body, http::Request, Router};
use chrono::Utc;
use h2ai_api::{routes::health_router, state::AppState};
use h2ai_config::H2AIConfig;
use h2ai_test_utils::decomposition_adapter;
use h2ai_types::{
    events::{CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode},
    identity::{TaskId, TenantId},
    sizing::{CoherencyCoefficients, CoordinationThreshold},
};
use serde_json::Value;
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
    health_router().with_state(state)
}

async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

async fn body_text(resp: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    String::from_utf8_lossy(&bytes).into_owned()
}

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

// ── liveness ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn liveness_returns_200_ok() {
    let app = make_router(make_state());
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = body_json(resp).await;
    assert_eq!(body["status"], "ok");
}

// ── readiness ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn readiness_missing_calibration_reports_missing() {
    let app = make_router(make_state());
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = body_json(resp).await;
    assert_eq!(body["status"], "ready");
    assert_eq!(body["calibration"], "missing");
}

#[tokio::test]
async fn readiness_with_calibration_reports_valid() {
    let state = make_state();
    let ts = state.tenant_state(&TenantId::default_tenant());
    *ts.calibration.write().await = Some(synthetic_calibration());

    let app = make_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = body_json(resp).await;
    assert_eq!(body["status"], "ready");
    assert_eq!(body["calibration"], "valid");
}

// ── metrics ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn metrics_returns_text_without_panic() {
    let app = make_router(make_state());
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let text = body_text(resp).await;
    // All non-empty lines must be valid Prometheus format: comment or metric entry.
    for line in text.lines().filter(|l| !l.is_empty()) {
        assert!(
            line.starts_with('#') || line.contains(' '),
            "unexpected prometheus line: {line}"
        );
    }
}
