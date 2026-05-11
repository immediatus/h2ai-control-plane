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

/// Core calibration runner shared by the HTTP route (via spawn) and startup eager calibration.
///
/// Runs `CalibrationHarness`, stores the result in `state.calibration` and NATS, and
/// optionally publishes a NATS SSE event when `notify_cal_id` is `Some` (HTTP path only).
pub(crate) async fn run_calibration_core(
    state: AppState,
    single_family_warning: bool,
    explorer_verification_family_match: bool,
    adapter_families: Vec<String>,
    notify_cal_id: Option<TaskId>,
) {
    let m = state.cfg.calibration_adapter_count.max(1);
    let prompts = vec![
        "Describe a stateless auth approach".into(),
        "Explain CQRS and event sourcing".into(),
        "What is a good API boundary?".into(),
    ];
    let pool: Vec<&dyn h2ai_types::adapter::IComputeAdapter> = {
        use h2ai_types::adapter::AdapterFamily;
        use std::collections::HashSet;
        let explorer_family = state.explorer_adapter.family();
        let candidates = [
            state.explorer_adapter.as_ref(),
            state.explorer2_adapter.as_ref(),
            state.verification_adapter.as_ref(),
        ];
        let mut seen: HashSet<*const dyn h2ai_types::adapter::IComputeAdapter> = HashSet::new();
        let mut distinct = Vec::new();
        for a in candidates {
            // Skip remote/cloud adapters that belong to a different family than the explorer.
            // USL calibration measures local LLM throughput; probing a remote API distorts
            // the fit and wastes rate-limited quota.
            if a.family() != explorer_family && a.family() != AdapterFamily::Mock {
                continue;
            }
            let ptr = a as *const dyn h2ai_types::adapter::IComputeAdapter;
            if seen.insert(ptr) {
                distinct.push(a);
            }
        }
        distinct
    };
    let n_distinct = pool.len();
    if n_distinct < 3 {
        tracing::warn!(
            n_distinct,
            "fewer than 3 distinct adapters configured; USL fit will use config fallback values"
        );
    }
    let cal_id = notify_cal_id.clone().unwrap_or_default();
    let adapter_refs: Vec<&dyn h2ai_types::adapter::IComputeAdapter> =
        pool.into_iter().cycle().take(m).collect();

    let result = CalibrationHarness::run(CalibrationInput {
        calibration_id: cal_id.clone(),
        task_prompts: prompts,
        adapters: adapter_refs,
        cfg: &state.cfg,
        constraint_corpus: &[],
        embedding_model: state.embedding_model.as_deref(),
    })
    .await;

    match result {
        Ok(mut event) => {
            event.adapter_families = adapter_families;
            event.explorer_verification_family_match = explorer_verification_family_match;
            event.single_family_warning = single_family_warning;
            {
                let mut cal = state.calibration.write().await;
                *cal = Some(event.clone());
            }
            {
                let mut metrics = state.metrics.write().await;
                metrics.n_eff_prior = event.n_eff_cosine_prior;
                metrics.calibration_source_label = match event.calibration_source {
                    h2ai_types::events::CalibrationSource::Measured => "measured",
                    h2ai_types::events::CalibrationSource::PartialFit => "partial_fit",
                    h2ai_types::events::CalibrationSource::SyntheticPriors => "synthetic_priors",
                }
                .to_string();
            }
            if let Err(e) = state.nats.put_calibration(&event).await {
                tracing::error!("failed to persist calibration: {e}");
            }
            if notify_cal_id.is_some() {
                let ev = H2AIEvent::CalibrationCompleted(event);
                let subject = format!("h2ai.calibration.{cal_id}");
                if let Err(e) = state.nats.publish_to(&subject, &ev).await {
                    tracing::error!("failed to publish calibration event: {e}");
                }
            }
        }
        Err(e) => {
            let is_network = e.to_string().contains("network error")
                || e.to_string().contains("connection refused")
                || e.to_string().contains("timed out");
            if is_network {
                tracing::warn!("calibration skipped — LLM adapter unreachable: {e}");
            } else {
                tracing::error!("calibration failed: {e}");
            }
            if let Some(cid) = notify_cal_id {
                let subject = format!("h2ai.calibration.{cid}");
                let _ = state
                    .nats
                    .publish_to(
                        &subject,
                        &H2AIEvent::CalibrationFailed {
                            calibration_id: cid.to_string(),
                            reason: e.to_string(),
                        },
                    )
                    .await;
            }
        }
    }
}

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

    // ── Multi-family enforcement (before spawn so we can return an HTTP error) ─
    use h2ai_types::adapter::AdapterFamily;
    use std::collections::HashSet;
    let pre_families: HashSet<AdapterFamily> = [
        state.explorer_adapter.family(),
        state.explorer2_adapter.family(),
        state.verification_adapter.family(),
    ]
    .into_iter()
    .filter(|f| *f != AdapterFamily::Mock)
    .collect();
    let single_family_warning = pre_families.len() == 1;
    if single_family_warning {
        use h2ai_config::FamilyConstraint;
        match state.cfg.safety.family_constraint {
            FamilyConstraint::RequireDiverse => {
                let family = pre_families
                    .iter()
                    .next()
                    .map(|f| f.to_string())
                    .unwrap_or_default();
                return Err(ApiError::SingleFamilyPool { family });
            }
            FamilyConstraint::SingleFamilyOk => {
                tracing::warn!(
                    target: "h2ai.calibration",
                    "single-family adapter pool: correlated hallucination protection degraded"
                );
            }
            FamilyConstraint::Disabled => {}
        }
    }
    let mut adapter_families: Vec<String> = pre_families.iter().map(|f| f.to_string()).collect();
    adapter_families.sort();
    let explorer_verification_family_match = state.explorer_adapter.family()
        == state.verification_adapter.family()
        && state.explorer_adapter.family() != AdapterFamily::Mock;
    // ──────────────────────────────────────────────────────────────────────────

    tokio::spawn(run_calibration_core(
        state.clone(),
        single_family_warning,
        explorer_verification_family_match,
        adapter_families,
        Some(cal_id.clone()),
    ));

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
            "n_eff_cosine_prior": c.n_eff_cosine_prior,
        }))),
        None => Err(ApiError::CalibrationRequired),
    }
}
