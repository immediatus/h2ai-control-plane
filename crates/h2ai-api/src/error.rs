use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::json;

#[derive(Debug)]
pub enum ApiError {
    CalibrationRequired,
    TaskNotFound(String),
    TaskAlreadyResolved(String),
    InvalidRequest(String),
    NatsUnavailable(String),
    ExplorerBudgetExceeded {
        requested: usize,
        n_max: f64,
    },
    ServiceUnavailable(String),
    /// All non-Mock adapters belong to the same family; BFT correlated hallucination protection
    /// is degraded. Set `family_constraint = "single_family_ok"` in config to proceed with a warning.
    SingleFamilyPool {
        family: String,
    },
    /// LLM adapter is unreachable (network error, timeout, or server down).
    #[allow(dead_code)]
    LlmUnavailable(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, body) = match self {
            Self::CalibrationRequired => (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({
                    "error": "CalibrationRequiredError",
                    "message": "No calibration data found. POST /calibrate before submitting tasks."
                }),
            ),
            Self::TaskNotFound(id) => (
                StatusCode::NOT_FOUND,
                json!({ "error": "TaskNotFound", "task_id": id }),
            ),
            Self::TaskAlreadyResolved(id) => (
                StatusCode::CONFLICT,
                json!({ "error": "TaskAlreadyResolved", "task_id": id }),
            ),
            Self::InvalidRequest(msg) => (
                StatusCode::BAD_REQUEST,
                json!({ "error": "InvalidRequest", "message": msg }),
            ),
            Self::NatsUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({ "error": "NatsUnavailable", "message": msg }),
            ),
            Self::ExplorerBudgetExceeded { requested, n_max } => (
                StatusCode::BAD_REQUEST,
                json!({
                    "error": "ExplorerBudgetExceeded",
                    "requested": requested,
                    "n_max": n_max,
                    "message": format!("Requested {requested} explorers but N_max={n_max:.1} for current calibration. Reduce explorer count.")
                }),
            ),
            Self::ServiceUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({ "error": "ServiceUnavailable", "message": msg }),
            ),
            Self::LlmUnavailable(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({ "error": "LlmUnavailable", "message": msg }),
            ),
            Self::SingleFamilyPool { family } => (
                StatusCode::BAD_REQUEST,
                json!({
                    "error": "SingleFamilyPool",
                    "family": family,
                    "message": format!(
                        "All non-Mock adapters are from the '{family}' family. \
                         Weiszfeld BFT correlated hallucination protection is degraded. \
                         Add adapters from a different family or set family_constraint = \"single_family_ok\"."
                    )
                }),
            ),
        };
        (status, Json(body)).into_response()
    }
}
