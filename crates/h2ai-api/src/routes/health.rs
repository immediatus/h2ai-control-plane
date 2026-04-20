use crate::state::AppState;
use axum::{extract::State, Json};
use serde_json::{json, Value};

pub async fn liveness() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

pub async fn readiness(State(state): State<AppState>) -> Json<Value> {
    let cal = state.calibration.read().await;
    let cal_status = if cal.is_some() { "valid" } else { "missing" };
    Json(json!({"status": "ready", "calibration": cal_status}))
}

pub async fn metrics() -> String {
    String::new()
}
