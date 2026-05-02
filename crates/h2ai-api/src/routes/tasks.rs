use crate::{error::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Json, Sse},
};
use h2ai_constraints::loader::load_corpus;
use h2ai_orchestrator::engine::{EngineInput, ExecutionEngine};
use h2ai_types::config::{TaoConfig, VerificationConfig};
use h2ai_types::events::{H2AIEvent, TaskAttributionEvent};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{MergeRequest, TaskAccepted, TaskManifest};
use serde_json::{json, Value};
use std::convert::Infallible;

/// Accept a [`TaskManifest`] and begin async execution, returning `202 Accepted` immediately.
///
/// Performs the following validation before spawning:
/// - Pareto weights (`diversity + containment + throughput`) must sum to 1.0 (±1e-4).
/// - A completed [`CalibrationCompletedEvent`] must be present; returns
///   `ApiError::CalibrationRequired` otherwise.
/// - Manifest description must pass the J_eff gate (`j_eff >= cfg.j_eff_gate`);
///   returns `ApiError::ContextUnderflow` when the token-level Jaccard score is too low.
/// - `manifest.explorers.count` must not exceed `calibration.coefficients.n_max()`;
///   returns `ApiError::ExplorerBudgetExceeded` otherwise.
/// - A semaphore permit must be available (`cfg.max_concurrent_tasks`); returns
///   `ApiError::ServiceUnavailable` when the server is at capacity.
///
/// On success the handler inserts the task into the store, spawns a Tokio task that runs
/// [`ExecutionEngine::run_offline`], and returns `202 Accepted` with a [`TaskAccepted`]
/// body containing the task ID, status URL, J_eff score, and topology kind.
/// When the engine finishes it publishes `H2AIEvent::VerificationScored` events to NATS
/// for each scored proposal, followed by a single `H2AIEvent::TaskAttribution` event
/// with quality metrics and waste analysis, then marks the task resolved in the store.
pub async fn submit_task(
    State(state): State<AppState>,
    Json(manifest): Json<TaskManifest>,
) -> Result<impl IntoResponse, ApiError> {
    if (manifest.pareto_weights.diversity
        + manifest.pareto_weights.containment
        + manifest.pareto_weights.throughput
        - 1.0)
        .abs()
        > 1e-4
    {
        return Err(ApiError::InvalidRequest(
            "pareto_weights must sum to 1.0".into(),
        ));
    }

    let calibration = {
        let cal = state.calibration.read().await;
        cal.clone().ok_or(ApiError::CalibrationRequired)?
    };

    let corpus_path =
        std::env::var("H2AI_CONSTRAINT_CORPUS_PATH").unwrap_or_else(|_| "/constraints".into());
    let corpus = load_corpus(&corpus_path)
        .map_err(|e| ApiError::Internal(format!("constraint corpus load failed: {e}")))?;

    use h2ai_context::jaccard::{jaccard, tokenize};
    let required_kw = corpus
        .iter()
        .flat_map(|d| d.vocabulary().into_iter())
        .chain(manifest.constraints.iter().cloned())
        .collect::<Vec<_>>()
        .join(" ");
    let required_kw = if required_kw.is_empty() {
        manifest.description.clone()
    } else {
        required_kw
    };
    let j_eff = jaccard(&tokenize(&manifest.description), &tokenize(&required_kw));
    if j_eff < state.cfg.j_eff_gate {
        return Err(ApiError::ContextUnderflow {
            j_eff,
            threshold: state.cfg.j_eff_gate,
        });
    }

    let topology_kind_str = manifest.topology.kind.clone();
    let n_max = calibration.coefficients.n_max();

    if manifest.explorers.count as f64 > n_max {
        return Err(ApiError::ExplorerBudgetExceeded {
            requested: manifest.explorers.count,
            n_max,
        });
    }

    let permit = state
        .task_semaphore
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::ServiceUnavailable(format!(
                "server at capacity ({} concurrent tasks)",
                state.cfg.max_concurrent_tasks
            ))
        })?;

    // Generate the task_id here so the response and the engine share the same identity.
    let task_id = TaskId::new();
    let task_id_str = task_id.to_string();

    // Pre-insert so GET /tasks/{id}/status succeeds immediately after this response.
    use h2ai_orchestrator::task_store::TaskState;
    state
        .store
        .insert(task_id.clone(), TaskState::new(task_id.clone()));

    let explorer = state.explorer_adapter.clone();
    let explorer2 = state.explorer2_adapter.clone();
    let verifier = state.verification_adapter.clone();
    let auditor = state.auditor_adapter.clone();
    let registry = state.registry();

    let state_clone = state.clone();
    let manifest_clone = manifest.clone();
    let calibration_clone = calibration.clone();
    let store_clone = state.store.clone();
    let task_id_clone = task_id.clone();

    tokio::spawn(async move {
        let _permit = permit; // dropped when this task completes, freeing semaphore slot
        let tao_multiplier = state_clone
            .tao_multiplier_estimator
            .read()
            .await
            .multiplier();
        let tao_multiplier_estimator = std::sync::Arc::clone(&state_clone.tao_multiplier_estimator);
        let input = EngineInput {
            task_id: task_id_clone,
            manifest: manifest_clone,
            calibration: calibration_clone,
            explorer_adapters: vec![explorer.as_ref(), explorer2.as_ref(), explorer.as_ref()],
            verification_adapter: verifier.as_ref(),
            auditor_adapter: auditor.as_ref(),
            auditor_config: h2ai_types::config::AuditorConfig {
                adapter: auditor.kind().clone(),
                ..Default::default()
            },
            tao_config: TaoConfig::default(),
            verification_config: VerificationConfig::default(),
            constraint_corpus: corpus,
            cfg: &state_clone.cfg,
            store: store_clone,
            nats_dispatch: None,
            registry: &registry,
            embedding_model: state_clone.embedding_model.as_deref(),
            tao_multiplier,
            tao_estimator: tao_multiplier_estimator,
        };

        match ExecutionEngine::run_offline(input).await {
            Ok(output) => {
                for event in output.verification_events {
                    let h2ai_ev = H2AIEvent::VerificationScored(event);
                    match state_clone
                        .nats
                        .publish_event_seq(&output.task_id, &h2ai_ev)
                        .await
                    {
                        Ok(seq) => {
                            if let Some(ts) = state_clone.store.get(&output.task_id) {
                                state_clone.journal.note_event(&output.task_id, seq, &ts);
                            }
                        }
                        Err(e) => tracing::warn!("failed to publish VerificationScoredEvent: {e}"),
                    }
                }

                // Apply τ-spread EMA update when the engine detected waste.
                if !output.applied_optimizations.is_empty() {
                    use h2ai_types::events::OptimizationKind;
                    for opt in &output.applied_optimizations {
                        if opt.kind == OptimizationKind::TauSpreadAdjusted {
                            if let (Ok(before), Ok(after)) =
                                (opt.before.parse::<f64>(), opt.after.parse::<f64>())
                            {
                                let mut est = state_clone.tau_spread_estimator.write().await;
                                // Use the verify_threshold change as a proxy for τ spread shift.
                                // before/after are verify_threshold values scaled to [0,1].
                                est.update(before.min(after), before.max(after));
                            }
                        }
                    }
                }

                let attr_ev = H2AIEvent::TaskAttribution(TaskAttributionEvent {
                    task_id: output.task_id.clone(),
                    q_predicted: output.attribution.total_quality,
                    q_measured: output.attribution.q_measured,
                    q_interval_lo: output.attribution_interval.as_ref().map(|iv| iv.q_total_lo),
                    q_interval_hi: output.attribution_interval.as_ref().map(|iv| iv.q_total_hi),
                    prediction_basis: output.attribution.prediction_basis,
                    waste_ratio: output.waste_ratio,
                    applied_optimizations: output.applied_optimizations,
                    timestamp: chrono::Utc::now(),
                });
                match state_clone
                    .nats
                    .publish_event_seq(&output.task_id, &attr_ev)
                    .await
                {
                    Ok(seq) => {
                        if let Some(ts) = state_clone.store.get(&output.task_id) {
                            state_clone.journal.note_event(&output.task_id, seq, &ts);
                        }
                    }
                    Err(e) => tracing::warn!("failed to publish TaskAttributionEvent: {e}"),
                }
                state_clone.store.mark_resolved(&output.task_id);
            }
            Err(e) => {
                tracing::error!("engine error: {e}");
            }
        }

        // Persist estimator state to NATS — fire-and-forget.
        if let Some((ema, count)) = state_clone
            .tao_multiplier_estimator
            .read()
            .await
            .persist_state()
        {
            if let Err(e) = state_clone.nats.put_tao_estimator_state(ema, count).await {
                tracing::warn!("failed to persist tao_estimator: {e}");
            }
        }
    });
    let events_url = format!("/tasks/{task_id_str}/events");

    let response = TaskAccepted {
        task_id: task_id_str,
        status: "accepted".into(),
        events_url,
        j_eff,
        topology_kind: topology_kind_str,
        n_max,
        interface_n_max: None,
    };

    Ok((StatusCode::ACCEPTED, Json(response)))
}

pub async fn task_events(
    Path(task_id): Path<String>,
    State(state): State<AppState>,
) -> axum::response::Response {
    use axum::response::sse::KeepAlive;
    use futures::StreamExt;

    let tid_uuid = match uuid::Uuid::parse_str(&task_id) {
        Ok(u) => u,
        Err(_) => {
            return Sse::new(tokio_stream::empty::<Result<Event, Infallible>>())
                .keep_alive(KeepAlive::default())
                .into_response();
        }
    };
    let tid = TaskId::from_uuid(tid_uuid);
    let from_seq: u64 = 0;

    let nats = state.nats.clone();
    let stream = async_stream::stream! {
        match nats.tail_task_events(&tid, from_seq).await {
            Err(e) => {
                tracing::error!("tail error: {e}");
            }
            Ok(mut events) => {
                while let Some(item) = events.next().await {
                    match item {
                        Ok((seq, event)) => {
                            let data = serde_json::to_string(&event).unwrap_or_default();
                            yield Ok::<Event, Infallible>(
                                Event::default().id(seq.to_string()).data(data)
                            );
                            if matches!(event, H2AIEvent::MergeResolved(_) | H2AIEvent::TaskFailed(_)) {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!("stream error: {e}");
                            break;
                        }
                    }
                }
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

pub async fn task_status(
    Path(task_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    let tid_uuid = uuid::Uuid::parse_str(&task_id)
        .map_err(|_| ApiError::InvalidRequest(format!("invalid task_id: {task_id}")))?;
    let tid = TaskId::from_uuid(tid_uuid);
    let ts = state
        .store
        .get(&tid)
        .ok_or_else(|| ApiError::TaskNotFound(task_id.clone()))?;
    Ok(Json(json!({
        "task_id": ts.task_id.to_string(),
        "status": ts.status,
        "phase": ts.phase,
        "phase_name": ts.phase_name,
        "explorers_completed": ts.explorers_completed,
        "explorers_total": ts.explorers_total,
        "proposals_valid": ts.proposals_valid,
        "proposals_pruned": ts.proposals_pruned,
        "autonomic_retries": ts.autonomic_retries,
    })))
}

pub async fn merge_task(
    Path(task_id): Path<String>,
    State(state): State<AppState>,
    Json(body): Json<MergeRequest>,
) -> Result<Json<Value>, ApiError> {
    let tid_uuid = uuid::Uuid::parse_str(&task_id)
        .map_err(|_| ApiError::InvalidRequest(format!("invalid task_id: {task_id}")))?;
    let tid = TaskId::from_uuid(tid_uuid);
    let ts = state
        .store
        .get(&tid)
        .ok_or_else(|| ApiError::TaskNotFound(task_id.clone()))?;
    if ts.status == "resolved" {
        return Err(ApiError::TaskAlreadyResolved(task_id));
    }
    let event = H2AIEvent::MergeResolved(h2ai_types::events::MergeResolvedEvent {
        task_id: tid.clone(),
        resolved_output: body
            .final_output
            .unwrap_or_else(|| body.selected_proposals.join(", ")),
        timestamp: chrono::Utc::now(),
    });
    state
        .nats
        .publish_event(&tid, &event)
        .await
        .map_err(|e| ApiError::NatsUnavailable(e.to_string()))?;
    state.store.mark_resolved(&tid);
    Ok(Json(json!({"status": "resolved", "task_id": task_id})))
}
