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
use h2ai_test_utils::{decomposition_adapter, mock_adapter};
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
    let adapter = Arc::new(decomposition_adapter("mock response"));
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
    let adapter = Arc::new(mock_adapter("mock"));
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

    let adapter = Arc::new(mock_adapter("mock calibration response"));
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

// ── start_calibration: FamilyConstraint branches ──────────────────────────────

#[tokio::test]
async fn start_calibration_single_family_ok_returns_202() {
    // FamilyConstraint::SingleFamilyOk (default) with a single-adapter pool must still
    // return 202 — it emits a warning but does not reject the request.
    let adapter = Arc::new(mock_adapter("mock"));
    let mut cfg = H2AIConfig::default();
    cfg.safety.family_constraint = FamilyConstraint::SingleFamilyOk;
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
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let body = body_json(resp).await;
    assert_eq!(body["status"], "accepted");
}

#[tokio::test]
async fn start_calibration_disabled_family_constraint_returns_202() {
    // FamilyConstraint::Disabled with a single-adapter pool: no family gate at all.
    let adapter = Arc::new(mock_adapter("mock"));
    let mut cfg = H2AIConfig::default();
    cfg.safety.family_constraint = FamilyConstraint::Disabled;
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
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn start_calibration_require_diverse_returns_error_with_family_field() {
    // RequireDiverse with a single-family pool must return a 400 body with "family" field.
    let adapter = Arc::new(mock_adapter("mock"));
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
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp).await;
    assert_eq!(body["error"], "SingleFamilyPool");
    assert!(
        body["family"].is_string(),
        "response must include 'family' field"
    );
}

// ── start_calibration: adapter_count in response ──────────────────────────────

#[tokio::test]
async fn start_calibration_response_includes_adapter_count() {
    // calibration_adapter_count > 0 → adapter_count field must match.
    let adapter = Arc::new(decomposition_adapter("mock response"));
    let cfg = H2AIConfig {
        calibration_adapter_count: 2,
        ..Default::default()
    };
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
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let body = body_json(resp).await;
    assert_eq!(body["adapter_count"], 2);
    assert!(body["events_url"]
        .as_str()
        .unwrap_or("")
        .starts_with("/calibrate/"));
}

#[tokio::test]
async fn start_calibration_zero_adapter_count_clamped_to_one() {
    // calibration_adapter_count=0 is clamped to max(1) → adapter_count must be 1.
    let adapter = Arc::new(decomposition_adapter("mock response"));
    let cfg = H2AIConfig {
        calibration_adapter_count: 0,
        ..Default::default()
    };
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
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let body = body_json(resp).await;
    assert_eq!(body["adapter_count"], 1);
}

// ── start_calibration: min_explorer_families gate ────────────────────────────

#[tokio::test]
async fn start_calibration_min_families_gate_satisfied_returns_202() {
    // min_explorer_families=1 with a 1-family pool: gate satisfied, request accepted.
    let adapter = Arc::new(decomposition_adapter("mock response"));
    let mut cfg = H2AIConfig::default();
    cfg.safety.min_explorer_families = 1;
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
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn start_calibration_min_families_gate_degraded_returns_202() {
    // min_explorer_families=2 with a 1-family pool: gate logs a warning but still
    // returns 202 (the gate is advisory, not blocking at the HTTP level).
    let adapter = Arc::new(decomposition_adapter("mock response"));
    let mut cfg = H2AIConfig::default();
    cfg.safety.min_explorer_families = 2;
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
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

// ── current_calibration: full JSON field coverage ─────────────────────────────

#[tokio::test]
async fn current_calibration_returns_all_expected_fields() {
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
    // Verify all JSON keys present in current_calibration handler are present and typed.
    assert!(body["calibration_id"].is_string());
    assert!(body["alpha"].is_number());
    assert!(body["beta_base"].is_number());
    assert!(body["beta_eff"].is_number());
    assert!(body["n_max"].is_number());
    assert!(body["theta_coord"].is_number());
    assert!(body["cg_mean"].is_number());
    assert!(body["cg_std_dev"].is_number());
    assert!(body["n_eff_cosine_prior"].is_number());
}

#[tokio::test]
async fn current_calibration_503_body_has_error_field() {
    // When no calibration exists, the response body must contain an "error" key.
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
    let body = body_json(resp).await;
    assert!(body["error"].is_string(), "missing 'error' key in 503 body");
}

// ── run_calibration_core: notify_cal_id = Some ───────────────────────────────

#[tokio::test]
async fn run_calibration_core_with_notify_cal_id_stores_calibration() {
    use h2ai_api::routes::calibrate::run_calibration_core;
    use h2ai_types::identity::TaskId;

    let adapter = Arc::new(mock_adapter("calibration response"));
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

    let cal_id = TaskId::new();
    run_calibration_core(
        state.clone(),
        false,
        false,
        vec!["Mock".into()],
        Some(cal_id),
    )
    .await;

    // Even without NATS, the calibration result is stored in state.
    assert!(
        ts.calibration.read().await.is_some(),
        "calibration must be stored when notify_cal_id is Some"
    );
}

// ── run_calibration_core: flags propagated to stored event ────────────────────

#[tokio::test]
async fn run_calibration_core_propagates_flags_to_stored_event() {
    use h2ai_api::routes::calibrate::run_calibration_core;

    let adapter = Arc::new(mock_adapter("calibration response"));
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

    run_calibration_core(
        state.clone(),
        true, // single_family_warning
        true, // explorer_verification_family_match
        vec!["FamilyA".into(), "FamilyB".into()],
        None,
    )
    .await;

    let cal = ts.calibration.read().await;
    let event = cal.as_ref().expect("calibration must be stored");
    assert!(
        event.single_family_warning,
        "single_family_warning must propagate"
    );
    assert!(
        event.explorer_verification_family_match,
        "explorer_verification_family_match must propagate"
    );
    assert_eq!(
        event.adapter_families,
        vec!["FamilyA".to_string(), "FamilyB".to_string()]
    );
}

// ── run_calibration_core: failing adapter (network error) no cached cal ───────

#[tokio::test]
async fn run_calibration_core_network_error_no_cached_leaves_state_empty() {
    use h2ai_api::routes::calibrate::run_calibration_core;
    use h2ai_test_utils::failing_adapter;

    let adapter = Arc::new(failing_adapter());
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

    // No pre-seeded calibration; adapter will fail with a network error.
    run_calibration_core(state.clone(), false, false, vec![], None).await;

    // With no cached calibration to re-emit, state must remain empty.
    assert!(
        ts.calibration.read().await.is_none(),
        "state must remain empty when network error and no cached calibration"
    );
}

// ── run_calibration_core: failing adapter (network error) with cached cal ─────

#[tokio::test]
async fn run_calibration_core_network_error_with_cache_updates_cal_id() {
    use h2ai_api::routes::calibrate::run_calibration_core;
    use h2ai_test_utils::failing_adapter;
    use h2ai_types::identity::TaskId;

    let adapter = Arc::new(failing_adapter());
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

    // Pre-seed a cached calibration so the network-error recovery path can re-emit it.
    let old_cal = synthetic_calibration();
    let old_id = old_cal.calibration_id.clone();
    *ts.calibration.write().await = Some(old_cal);

    let new_cal_id = TaskId::new();
    run_calibration_core(
        state.clone(),
        false,
        false,
        vec![],
        Some(new_cal_id.clone()),
    )
    .await;

    // The cached calibration's ID must be replaced with the new cal_id.
    let cal = ts.calibration.read().await;
    let stored = cal
        .as_ref()
        .expect("cached calibration must still be stored");
    assert_ne!(
        stored.calibration_id, old_id,
        "calibration_id must be updated to the new cal_id"
    );
    assert_eq!(
        stored.calibration_id, new_cal_id,
        "calibration_id must match the new cal_id passed to run_calibration_core"
    );
}

// ── calibrate_events: SSE endpoint ───────────────────────────────────────────

#[tokio::test]
async fn calibrate_events_returns_200_sse_content_type() {
    // The SSE endpoint must return 200 with text/event-stream content-type.
    // We seed a calibration so the stream yields immediately without indefinite polling.
    let state = make_state();
    let ts = state.tenant_state(&TenantId::default_tenant());
    *ts.calibration.write().await = Some(synthetic_calibration());

    let app = make_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/calibrate/test-cal-id/events")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "expected text/event-stream, got: {content_type}"
    );
}
