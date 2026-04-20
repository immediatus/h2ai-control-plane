use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

#[derive(Debug)]
#[allow(dead_code)]
pub enum ApiError {
    ContextUnderflow { j_eff: f64, threshold: f64 },
    CalibrationRequired,
    TaskNotFound(String),
    TaskAlreadyResolved(String),
    InvalidRequest(String),
    Internal(String),
    NatsUnavailable(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, body) = match self {
            ApiError::ContextUnderflow { j_eff, threshold } => (
                StatusCode::BAD_REQUEST,
                json!({
                    "error": "ContextUnderflowError",
                    "j_eff": j_eff,
                    "threshold": threshold,
                    "message": "Jaccard overlap between submitted context and task requirements is too low."
                }),
            ),
            ApiError::CalibrationRequired => (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({
                    "error": "CalibrationRequiredError",
                    "message": "No calibration data found. POST /calibrate before submitting tasks."
                }),
            ),
            ApiError::TaskNotFound(id) => (
                StatusCode::NOT_FOUND,
                json!({ "error": "TaskNotFound", "task_id": id }),
            ),
            ApiError::TaskAlreadyResolved(id) => (
                StatusCode::CONFLICT,
                json!({ "error": "TaskAlreadyResolved", "task_id": id }),
            ),
            ApiError::InvalidRequest(msg) => (
                StatusCode::BAD_REQUEST,
                json!({ "error": "InvalidRequest", "message": msg }),
            ),
            ApiError::NatsUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({ "error": "NatsUnavailable", "message": msg }),
            ),
            ApiError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({ "error": "InternalError", "message": msg }),
            ),
        };
        (status, Json(body)).into_response()
    }
}
