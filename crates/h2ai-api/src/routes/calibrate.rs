use crate::{error::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Json, Sse},
};
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_types::events::H2AIEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::CalibrationAccepted;
use serde_json::{json, Value};
use std::convert::Infallible;

pub async fn start_calibration(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let cal_id = TaskId::new();
    let cal_id_str = cal_id.to_string();
    let events_url = format!("/calibrate/{cal_id_str}/events");

    let m = state.cfg.calibration_adapter_count.max(1);
    if m < 3 {
        tracing::warn!(
            calibration_adapter_count = m,
            "calibration_adapter_count < 3; USL fit will use config fallback values"
        );
    }
    let state_clone = state.clone();
    let cal_id_clone = cal_id.clone();
    tokio::spawn(async move {
        let prompts = vec![
            "Describe a stateless auth approach".into(),
            "Explain CQRS and event sourcing".into(),
            "What is a good API boundary?".into(),
        ];
        let adapter_refs: Vec<&dyn h2ai_types::adapter::IComputeAdapter> = (0..m)
            .map(|_| state_clone.explorer_adapter.as_ref())
            .collect();
        let result = CalibrationHarness::run(CalibrationInput {
            calibration_id: cal_id_clone.clone(),
            task_prompts: prompts,
            adapters: adapter_refs,
            cfg: &state_clone.cfg,
            embedding_model: None,
        })
        .await;

        match result {
            Ok(event) => {
                let mut cal = state_clone.calibration.write().await;
                *cal = Some(event.clone());
                drop(cal);
                if let Err(e) = state_clone.nats.put_calibration(&event).await {
                    tracing::error!("failed to persist calibration: {e}");
                }
                let ev = H2AIEvent::CalibrationCompleted(event);
                let subject = format!("h2ai.calibration.{cal_id_clone}");
                if let Err(e) = state_clone.nats.publish_to(&subject, &ev).await {
                    tracing::error!("failed to publish calibration event: {e}");
                }
            }
            Err(e) => tracing::error!("calibration failed: {e}"),
        }
    });

    let response = CalibrationAccepted {
        calibration_id: cal_id_str,
        status: "accepted".into(),
        events_url,
        adapter_count: m,
    };
    Ok((StatusCode::ACCEPTED, Json(response)))
}

pub async fn calibrate_events(
    Path(_cal_id): Path<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    use axum::response::sse::KeepAlive;

    let cal_cache = state.calibration.clone();
    let stream = async_stream::stream! {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let cal = cal_cache.read().await;
            if let Some(ref event) = *cal {
                let ev = H2AIEvent::CalibrationCompleted(event.clone());
                let data = serde_json::to_string(&ev).unwrap_or_default();
                yield Ok::<Event, Infallible>(Event::default().data(data));
                break;
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub async fn current_calibration(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let cal = state.calibration.read().await;
    match cal.as_ref() {
        Some(c) => Ok(Json(json!({
            "calibration_id": c.calibration_id.to_string(),
            "alpha": c.coefficients.alpha,
            "beta_base": c.coefficients.beta_base,
            "beta_eff": c.coefficients.beta_eff(),
            "n_max": c.coefficients.n_max(),
            "theta_coord": c.coordination_threshold.value(),
            "cg_mean": c.coefficients.cg_mean(),
            "cg_std_dev": c.coefficients.cg_std_dev(),
        }))),
        None => Err(ApiError::CalibrationRequired),
    }
}
