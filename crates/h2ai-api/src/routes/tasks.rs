use crate::{error::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Json, Sse},
};
use h2ai_context::adr::load_corpus;
use h2ai_orchestrator::engine::{EngineInput, ExecutionEngine};
use h2ai_types::config::{TaoConfig, VerificationConfig};
use h2ai_types::events::H2AIEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{MergeRequest, TaskAccepted, TaskManifest};
use serde_json::{json, Value};
use std::convert::Infallible;

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

    let adr_path = std::env::var("H2AI_ADR_CORPUS_PATH").unwrap_or_else(|_| "/adr".into());
    let corpus = load_corpus(&adr_path).unwrap_or_default();

    use h2ai_context::jaccard::{jaccard, tokenize};
    let required_kw = corpus
        .iter()
        .flat_map(|a: &h2ai_context::adr::AdrConstraints| a.keywords.iter().cloned())
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

    let adapter = h2ai_adapters::mock::MockAdapter::new(
        "mock output — replace with real adapter in production".into(),
    );
    let auditor = h2ai_adapters::mock::MockAdapter::new(
        "mock auditor — replace with real adapter in production".into(),
    );

    let state_clone = state.clone();
    let manifest_clone = manifest.clone();
    let calibration_clone = calibration.clone();
    let store_clone = state.store.clone();

    tokio::spawn(async move {
        let input = EngineInput {
            manifest: manifest_clone,
            calibration: calibration_clone,
            explorer_adapters: vec![
                &adapter as &dyn h2ai_types::adapter::IComputeAdapter,
                &adapter,
            ],
            auditor_adapter: &auditor as &dyn h2ai_types::adapter::IComputeAdapter,
            auditor_config: h2ai_types::config::AuditorConfig {
                adapter: h2ai_types::config::AdapterKind::CloudGeneric {
                    endpoint: "mock".into(),
                    api_key_env: "NONE".into(),
                },
                ..Default::default()
            },
            tao_config: TaoConfig::default(),
            verification_config: VerificationConfig::default(),
            adr_corpus: corpus,
            cfg: &state_clone.cfg,
            store: store_clone,
        };

        match ExecutionEngine::run_offline(input).await {
            Ok(output) => {
                for event in output.verification_events {
                    let h2ai_ev = H2AIEvent::VerificationScored(event);
                    if let Err(e) = state_clone
                        .nats
                        .publish_event(&output.task_id, &h2ai_ev)
                        .await
                    {
                        tracing::warn!("failed to publish VerificationScoredEvent: {e}");
                    }
                }
                state_clone.store.mark_resolved(&output.task_id);
            }
            Err(e) => {
                tracing::error!("engine error: {e}");
            }
        }
    });

    let task_id = TaskId::new();
    let task_id_str = task_id.to_string();
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
