use crate::state::AppState;
use axum::{extract::State, Json};
use h2ai_types::identity::TenantId;
use serde_json::{json, Value};

pub async fn liveness() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

pub async fn readiness(State(state): State<AppState>) -> Json<Value> {
    let ts = state.tenant_state(&TenantId::default_tenant());
    let cal = ts.calibration.read().await;
    let cal_status = if cal.is_some() { "valid" } else { "missing" };
    Json(json!({"status": "ready", "calibration": cal_status}))
}

pub async fn metrics(State(state): State<AppState>) -> String {
    let m = state.metrics.read().await;
    m.to_prometheus_text()
}
