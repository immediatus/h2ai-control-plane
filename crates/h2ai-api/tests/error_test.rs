#![allow(clippy::missing_panics_doc)]
//! Tests for `ApiError::into_response` — covers every variant.

use axum::{body, http::StatusCode, response::IntoResponse};
use h2ai_api::error::ApiError;
use serde_json::Value;

async fn to_parts(err: ApiError) -> (StatusCode, Value) {
    let resp = err.into_response();
    let status = resp.status();
    let bytes = body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, body)
}

#[tokio::test]
async fn calibration_required_is_503() {
    let (status, body) = to_parts(ApiError::CalibrationRequired).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "CalibrationRequiredError");
}

#[tokio::test]
async fn task_not_found_is_404() {
    let (status, body) = to_parts(ApiError::TaskNotFound("tid-1".into())).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"], "TaskNotFound");
    assert_eq!(body["task_id"], "tid-1");
}

#[tokio::test]
async fn task_already_resolved_is_409() {
    let (status, body) = to_parts(ApiError::TaskAlreadyResolved("tid-2".into())).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"], "TaskAlreadyResolved");
    assert_eq!(body["task_id"], "tid-2");
}

#[tokio::test]
async fn invalid_request_is_400() {
    let (status, body) = to_parts(ApiError::InvalidRequest("bad input".into())).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "InvalidRequest");
}

#[tokio::test]
async fn nats_unavailable_is_503() {
    let (status, body) = to_parts(ApiError::NatsUnavailable("conn refused".into())).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "NatsUnavailable");
}

#[tokio::test]
async fn explorer_budget_exceeded_is_400() {
    let (status, body) = to_parts(ApiError::ExplorerBudgetExceeded {
        requested: 20,
        n_max: 10.0,
    })
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "ExplorerBudgetExceeded");
    assert_eq!(body["requested"], 20);
}

#[tokio::test]
async fn service_unavailable_is_503() {
    let (status, body) = to_parts(ApiError::ServiceUnavailable("at capacity".into())).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "ServiceUnavailable");
}

#[tokio::test]
async fn llm_unavailable_is_503() {
    let (status, body) = to_parts(ApiError::LlmUnavailable("timeout".into())).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"], "LlmUnavailable");
}

#[tokio::test]
async fn single_family_pool_is_400() {
    let (status, body) = to_parts(ApiError::SingleFamilyPool {
        family: "OpenAI".into(),
    })
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "SingleFamilyPool");
    assert_eq!(body["family"], "OpenAI");
    assert!(
        body["message"].as_str().unwrap_or("").contains("OpenAI"),
        "message must mention family name"
    );
}
