pub mod calibrate;
pub mod health;
pub mod recovery;
pub mod tasks;

use crate::state::AppState;
use axum::Router;

pub fn task_router() -> Router<AppState> {
    use axum::routing::{get, post};
    Router::new()
        .route("/tasks", post(tasks::submit_task))
        .route("/tasks/:task_id/events", get(tasks::task_events))
        .route("/tasks/:task_id", get(tasks::task_status))
        .route("/tasks/:task_id/merge", post(tasks::merge_task))
        .route("/tasks/:task_id/recover", get(recovery::recover_task))
}

pub fn calibrate_router() -> Router<AppState> {
    use axum::routing::{get, post};
    Router::new()
        .route("/calibrate", post(calibrate::start_calibration))
        .route(
            "/calibrate/:cal_id/events",
            get(calibrate::calibrate_events),
        )
        .route("/calibrate/current", get(calibrate::current_calibration))
}

pub fn health_router() -> Router<AppState> {
    use axum::routing::get;
    Router::new()
        .route("/health", get(health::liveness))
        .route("/ready", get(health::readiness))
        .route("/metrics", get(health::metrics))
}
