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
        // Cycle all distinct configured adapters so CG_mean reflects inter-adapter
        // coordination cost, not just within-adapter temperature variance.
        let pool: Vec<&dyn h2ai_types::adapter::IComputeAdapter> = {
            use std::collections::HashSet;
            let candidates = [
                state_clone.explorer_adapter.as_ref(),
                state_clone.explorer2_adapter.as_ref(),
                state_clone.verification_adapter.as_ref(),
            ];
            let mut seen: HashSet<*const dyn h2ai_types::adapter::IComputeAdapter> = HashSet::new();
            let mut distinct = Vec::new();
            for a in candidates {
                let ptr = a as *const dyn h2ai_types::adapter::IComputeAdapter;
                if seen.insert(ptr) {
                    distinct.push(a);
                }
            }
            distinct
        };
        let n_distinct = pool.len();
        let adapter_refs: Vec<&dyn h2ai_types::adapter::IComputeAdapter> =
            pool.into_iter().cycle().take(m).collect();
        if n_distinct < 3 {
            tracing::warn!(
                n_distinct,
                "fewer than 3 distinct adapters configured; USL fit will use config fallback values"
            );
        }

        // ── Multi-family enforcement ──────────────────────────────────────────
        use h2ai_types::adapter::AdapterFamily;
        use std::collections::HashSet;

        let families: HashSet<AdapterFamily> = adapter_refs
            .iter()
            .map(|a| a.family())
            .filter(|f| *f != AdapterFamily::Mock)
            .collect();

        let single_family_warning = families.len() == 1;

        if single_family_warning && !state_clone.cfg.allow_single_family {
            let family = families
                .iter()
                .next()
                .map(|f| f.to_string())
                .unwrap_or_default();
            tracing::error!(
                target: "h2ai.calibration",
                family,
                "single-family adapter pool: calibration aborted. \
                 Add adapters from a different family or set allow_single_family=true."
            );
            return;
        }
        if single_family_warning {
            tracing::warn!(
                target: "h2ai.calibration",
                "single-family adapter pool: Weiszfeld BFT correlated hallucination protection \
                 degraded. Set allow_single_family=true to acknowledge."
            );
        }

        let mut adapter_families: Vec<String> = families.iter().map(|f| f.to_string()).collect();
        adapter_families.sort();

        let explorer_verification_family_match = state_clone.explorer_adapter.family()
            == state_clone.verification_adapter.family()
            && state_clone.explorer_adapter.family() != AdapterFamily::Mock;
        // ─────────────────────────────────────────────────────────────────────

        let result = CalibrationHarness::run(CalibrationInput {
            calibration_id: cal_id_clone.clone(),
            task_prompts: prompts,
            adapters: adapter_refs,
            cfg: &state_clone.cfg,
            embedding_model: state_clone.embedding_model.as_deref(),
        })
        .await;

        match result {
            Ok(mut event) => {
                event.adapter_families = adapter_families;
                event.explorer_verification_family_match = explorer_verification_family_match;
                event.single_family_warning = single_family_warning;
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
