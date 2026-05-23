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
use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use chrono::Utc;
use h2ai_api::{routes::calibrate_router, state::AppState};
use h2ai_config::{FamilyConstraint, H2AIConfig};
use h2ai_test_utils::{DecompositionMockAdapter, MockAdapter};
use h2ai_types::{
    events::{CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode},
    identity::{TaskId, TenantId},
    sizing::{CoherencyCoefficients, CoordinationThreshold},
};
use serde_json::Value;
use tower::ServiceExt;

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
    let adapter = Arc::new(DecompositionMockAdapter::new("mock response".into()));
    AppState::new_for_tests(
        H2AIConfig::default(),
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    )
}

fn make_router(state: AppState) -> Router {
    calibrate_router().with_state(state)
}

async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

// ── current_calibration ───────────────────────────────────────────────────────

#[tokio::test]
async fn current_calibration_returns_503_when_no_calibration() {
    let state = make_state();
    let app = make_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/calibrate/current")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn current_calibration_returns_200_with_data() {
    let state = make_state();
    let ts = state.tenant_state(&TenantId::default_tenant());
    *ts.calibration.write().await = Some(synthetic_calibration());

    let app = make_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/calibrate/current")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    assert!(body["calibration_id"].is_string());
    assert!(body["alpha"].is_number());
    assert!(body["theta_coord"].is_number());
}

// ── start_calibration ─────────────────────────────────────────────────────────

#[tokio::test]
async fn start_calibration_returns_202_accepted() {
    let state = make_state();
    let app = make_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/calibrate")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let body = body_json(resp).await;
    assert_eq!(body["status"], "accepted");
    assert!(body["calibration_id"].is_string());
    assert!(body["events_url"].is_string());
}

#[tokio::test]
async fn start_calibration_returns_error_for_require_diverse_single_pool() {
    let adapter = Arc::new(MockAdapter::new("mock".into()));
    let mut cfg = H2AIConfig::default();
    cfg.safety.family_constraint = FamilyConstraint::RequireDiverse;
    let state = AppState::new_for_tests(
        cfg,
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    );
    let app = make_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/calibrate")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // Single-family pool with RequireDiverse → error (4xx/5xx)
    assert!(resp.status().is_client_error() || resp.status().is_server_error());
}

// ── run_calibration_core ──────────────────────────────────────────────────────

#[tokio::test]
async fn run_calibration_core_stores_calibration_in_state() {
    use h2ai_api::routes::calibrate::run_calibration_core;

    let adapter = Arc::new(MockAdapter::new("mock calibration response".into()));
    let cfg = H2AIConfig {
        calibration_adapter_count: 1,
        ..H2AIConfig::default()
    };
    let state = AppState::new_for_tests(
        cfg,
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    );

    let ts = state.tenant_state(&TenantId::default_tenant());
    assert!(ts.calibration.read().await.is_none());

    run_calibration_core(state.clone(), false, false, vec!["Mock".into()], None).await;

    assert!(
        ts.calibration.read().await.is_some(),
        "calibration must be stored after run_calibration_core"
    );
}

// ── legacy coefficient tests ──────────────────────────────────────────────────

#[test]
fn calibration_event_has_valid_n_max() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71]).unwrap();
    let n_max = cc.n_max();
    // New formula β_eff = β₀/max(CG,0.05): CG_mean≈0.71, β_eff=0.021/0.71≈0.030 → N_max≈5
    assert!(
        n_max > 1.0 && n_max < 20.0,
        "n_max={n_max} out of expected range"
    );
}

#[test]
fn calibration_theta_coord_bounded() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    assert!(theta.value() <= 0.3);
    assert!(theta.value() >= 0.0);
}
