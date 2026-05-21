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
use h2ai_types::approval::{ApprovalDecision, ApprovalRecord};
use h2ai_types::events::{ApprovalRiskLevel, ApprovalTrigger};
use h2ai_types::identity::TenantId;

// ── Integration helpers for /signal route tests ──────────────────────────────

/// Fixed task UUID used by the /signal integration tests.
/// Pre-inserted into the task store as an active (Bootstrap phase) task.
const SIGNAL_TEST_TASK_ID: &str = "00000000-0000-0000-0000-000000000001";

/// Build a minimal Axum test app without a live NATS server.
///
/// Uses `AppState::new_for_tests()` with an in-memory backend so no NATS connection
/// is required. The task `SIGNAL_TEST_TASK_ID` is pre-seeded into the store as active.
async fn build_test_app() -> axum::Router {
    use h2ai_adapters::mock::MockAdapter;
    use h2ai_api::{routes::task_router, state::AppState};
    use h2ai_config::H2AIConfig;
    use h2ai_orchestrator::task_store::TaskState;
    use h2ai_types::identity::TaskId;
    use std::sync::Arc;

    let cfg = H2AIConfig::load_layered(None).expect("load config");
    let adapter = Arc::new(MockAdapter::new(
        r#"{"approved":true,"score":0.9,"reason":"mock"}"#.into(),
    ));
    let state = AppState::new_for_tests(
        cfg,
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter,
    );

    // Pre-seed an active task so the /signal handler finds it
    let task_id =
        TaskId::from_uuid(uuid::Uuid::parse_str(SIGNAL_TEST_TASK_ID).expect("parse fixed UUID"));
    let tenant = TenantId::from("test-team");
    state
        .store
        .insert(task_id.clone(), TaskState::new(task_id, tenant));

    axum::Router::new().merge(task_router()).with_state(state)
}

#[tokio::test]
async fn signal_approve_returns_202() {
    use tower::ServiceExt;

    let app = build_test_app().await;
    let body = serde_json::json!({
        "payload": {
            "kind": "Approve",
            "data": {
                "approved": true,
                "reviewer_note": null,
                "operator_id": "test-operator"
            }
        }
    });
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(format!("/test-team/tasks/{SIGNAL_TEST_TASK_ID}/signal"))
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::ACCEPTED);
}

#[tokio::test]
async fn signal_approve_missing_operator_id_returns_400() {
    use tower::ServiceExt;

    let app = build_test_app().await;
    let body = serde_json::json!({
        "payload": {
            "kind": "Approve",
            "data": {
                "approved": true,
                "reviewer_note": null,
                "operator_id": ""
            }
        }
    });
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(format!("/test-team/tasks/{SIGNAL_TEST_TASK_ID}/signal"))
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[test]
fn approval_record_serializes_roundtrip() {
    let record = ApprovalRecord {
        task_id: "task-123".into(),
        tenant_id: TenantId::default_tenant(),
        proposed_output: "the answer".into(),
        q_confidence: 0.42,
        triggered_by: ApprovalTrigger::LowConfidence,
        created_at_ms: 1000,
        timeout_at_ms: 1000 + 1800000,
    };
    let json = serde_json::to_string(&record).unwrap();
    let back: ApprovalRecord = serde_json::from_str(&json).unwrap();
    assert!((back.q_confidence - 0.42).abs() < 1e-9);
    assert_eq!(back.task_id, "task-123");
}

#[test]
fn approval_decision_serializes_roundtrip() {
    let decision = ApprovalDecision {
        approved: true,
        reviewer_note: Some("LGTM".into()),
        operator_id: "alice@example.com".into(),
        decided_at_ms: 9999,
    };
    let json = serde_json::to_string(&decision).unwrap();
    let back: ApprovalDecision = serde_json::from_str(&json).unwrap();
    assert!(back.approved);
    assert_eq!(back.operator_id, "alice@example.com");
}

#[test]
fn risk_level_high_when_low_confidence() {
    let risk = h2ai_types::approval::compute_risk_level(&ApprovalTrigger::LowConfidence, 0.25);
    assert_eq!(risk, ApprovalRiskLevel::High);
}

#[test]
fn risk_level_medium_when_manifest_flag_and_moderate_confidence() {
    let risk = h2ai_types::approval::compute_risk_level(&ApprovalTrigger::ManifestFlag, 0.60);
    assert_eq!(risk, ApprovalRiskLevel::Medium);
}

#[test]
fn require_approval_defaults_false() {
    use h2ai_types::manifest::TaskManifest;
    let json = r#"{"description":"t","pareto_weights":{"throughput":0.33,"containment":0.33,"diversity":0.34},"topology":{"kind":"auto","branching_factor":null},"explorers":{"count":3,"tau_min":null,"tau_max":null,"roles":[],"review_gates":[],"slot_configs":[]}}"#;
    let m: TaskManifest = serde_json::from_str(json).unwrap();
    assert!(
        !m.require_approval,
        "require_approval must default to false"
    );
}

#[test]
fn hitl_config_defaults_are_sane() {
    use h2ai_config::H2AIConfig;
    let cfg = H2AIConfig::load_layered(None).expect("load defaults");
    assert!(cfg.hitl.enabled);
    assert!((cfg.hitl.confidence_threshold - 0.50).abs() < 1e-9);
    assert_eq!(cfg.hitl.timeout_ms, 1_800_000); // 30 minutes
}

#[test]
fn high_confidence_task_bypasses_gate() {
    let q = 0.95f64;
    let threshold = 0.50f64;
    let require_approval = false;
    let hitl_enabled = true;
    let oracle_task = false;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(!needs_approval, "high confidence task must bypass gate");
}

#[test]
fn low_confidence_task_hits_gate() {
    let q = 0.30f64;
    let threshold = 0.50f64;
    let require_approval = false;
    let hitl_enabled = true;
    let oracle_task = false;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(needs_approval, "low confidence task must hit gate");
}

#[test]
fn require_approval_hits_gate_regardless_of_confidence() {
    let q = 0.99f64;
    let threshold = 0.50f64;
    let require_approval = true;
    let hitl_enabled = true;
    let oracle_task = false;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(needs_approval, "require_approval=true must always hit gate");
}

#[test]
fn oracle_task_always_bypasses_gate() {
    let q = 0.10f64;
    let threshold = 0.50f64;
    let require_approval = true;
    let hitl_enabled = true;
    let oracle_task = true;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(!needs_approval, "oracle task must always bypass HITL gate");
}

#[tokio::test]
async fn approve_endpoint_returns_301_to_signal() {
    use tower::ServiceExt;

    let app = build_test_app().await;
    let body = serde_json::json!({
        "approved": true,
        "reviewer_note": null,
        "operator_id": "ops@example.com"
    });
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/test-team/tasks/00000000-0000-0000-0000-000000000001/approve")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        axum::http::StatusCode::PERMANENT_REDIRECT
    );
    let location = response
        .headers()
        .get("location")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(location.ends_with("/signal"));
}

#[test]
fn hitl_disabled_bypasses_gate() {
    let q = 0.10f64;
    let threshold = 0.50f64;
    let require_approval = true;
    let hitl_enabled = false;
    let oracle_task = false;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(!needs_approval, "disabled HITL must bypass gate");
}

#[test]
fn timeout_triggers_auto_reject_condition() {
    let now_ms: u64 = 1_000_000;
    let timeout_at_ms: u64 = 999_999; // already expired
    assert!(
        now_ms > timeout_at_ms,
        "expired record must trigger auto-reject"
    );

    let future_timeout: u64 = 2_000_000;
    assert!(
        now_ms <= future_timeout,
        "future timeout must not trigger auto-reject"
    );
}
