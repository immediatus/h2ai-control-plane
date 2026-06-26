use crate::task_pipeline::{run_task_pipeline, TaskPipelineInput};
use crate::{error::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Json, Sse},
};
use h2ai_orchestrator::engine::{NatsDispatchConfig, ShadowAuditCtx};
use h2ai_types::agent::{AgentDescriptor, AgentTool, CostTier, TaskRequirements};
use h2ai_types::events::H2AIEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{MergeRequest, TaskAccepted, TaskManifest};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

pub fn compute_j_eff_raw(
    n_valid: usize,
    n_agents: usize,
    p_mean: f64,
    rho_mean: f64,
) -> Option<f64> {
    use h2ai_types::sizing::condorcet_quality;
    let filter_ratio = if n_agents > 0 {
        n_valid as f64 / n_agents as f64
    } else {
        0.0
    };
    let q_realized = condorcet_quality(n_valid, filter_ratio, rho_mean);
    let q_ceiling = condorcet_quality(n_agents, p_mean, 0.0);
    if q_ceiling > 0.0 {
        Some((q_realized / q_ceiling).clamp(0.0, 1.0))
    } else {
        None
    }
}

pub fn compute_j_eff(
    n_valid: usize,
    n_agents: usize,
    calibration: &h2ai_types::events::CalibrationCompletedEvent,
) -> Option<f64> {
    let (p_mean, rho_mean) = calibration
        .ensemble
        .as_ref()
        .map_or((0.5, 0.0), |e| (e.p_mean, e.rho_mean));
    compute_j_eff_raw(n_valid, n_agents, p_mean, rho_mean)
}

pub async fn submit_task(
    Path(tenant_id): Path<String>,
    State(state): State<AppState>,
    Json(mut manifest): Json<TaskManifest>,
) -> Result<impl IntoResponse, ApiError> {
    use h2ai_orchestrator::task_store::TaskState;

    manifest.tenant_id = h2ai_types::identity::TenantId::from(tenant_id.as_str());
    validate_pareto_weights(&manifest)?;

    let task_tenant_id = manifest.tenant_id.clone();
    state
        .seed_calibration_from_default_if_needed(&task_tenant_id)
        .await;
    let ts = state.tenant_state(&task_tenant_id);
    let calibration = {
        let cal = ts.calibration.read().await;
        cal.clone().ok_or(ApiError::CalibrationRequired)?
    };

    let task_tags = manifest.constraint_tags.clone();
    let explicit_ids = manifest.constraints.clone();
    let corpus = state
        .constraint_resolver
        .resolve(&explicit_ids, &task_tags, &manifest.description)
        .await;
    let wiki_revision = 0u64;
    let resolved_ids: Vec<String> = corpus.iter().map(|d| d.id.clone()).collect();
    tracing::info!(
        target: "h2ai.tasks",
        n_constraints = corpus.len(),
        constraint_ids = ?resolved_ids,
        "resolved constraints for task"
    );

    let topology_kind_str = manifest.topology.kind.clone();
    let n_max = calibration.coefficients.n_max();
    validate_explorer_budget(&manifest, n_max)?;
    let permit = acquire_permit(&state)?;

    let task_id = TaskId::new();
    let task_id_str = task_id.to_string();

    state.store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), manifest.tenant_id.clone()),
    );

    let manifest_json = serde_json::to_string(&manifest).unwrap_or_default();
    let oracle_spec = manifest.oracle.clone();
    let nats_dispatch = state
        .agent_provider
        .as_ref()
        .map(|provider| NatsDispatchConfig {
            nats: state
                .task_dispatch_nats
                .clone()
                .expect("NATS required for agent dispatch"),
            provider: Arc::clone(provider),
            agent_descriptor: AgentDescriptor {
                model: state.cfg.nats_agent_model.clone(),
                tools: vec![AgentTool::Shell, AgentTool::FileSystem],
                cost_tier: CostTier::Mid,
            },
            task_requirements: TaskRequirements {
                max_cost_tier: CostTier::High,
                required_tools: vec![AgentTool::Shell, AgentTool::FileSystem],
            },
            task_timeout: Duration::from_secs(state.cfg.nats_agent_timeout_secs),
            payload_store: state.payload_store.clone(),
            offload_threshold_bytes: 8 * 1024,
        });

    let shadow_audit_ctx = if let Some(ref adapter) = state.shadow_auditor_adapter {
        let promoted_snap = state.promoted_audit_domains.read().await.clone();
        Some(ShadowAuditCtx {
            adapter: adapter.clone(),
            promoted_domains: promoted_snap,
            strict: state.cfg.safety.shadow_auditor.strict,
        })
    } else {
        None
    };

    let pipeline_input = TaskPipelineInput {
        task_id: task_id.clone(),
        tenant_id: task_tenant_id,
        manifest: manifest.clone(),
        calibration,
        corpus,
        wiki_revision,
        manifest_json,
        resolved_ids,
        thinking_loop_runner: state.thinking_loop_runner.clone(),
        decomposer: state.decomposer.clone(),
        engine_runner: state.engine_runner.clone(),
        nats: state.nats.clone(),
        nats_raw_client: state.nats_raw_client.clone(),
        store: state.store.clone(),
        journal: state.journal.clone(),
        cfg: Arc::clone(&state.cfg),
        metrics: state.metrics.clone(),
        drift_monitor: state.drift_monitor.clone(),
        adapter_pool: state.adapter_pool.clone(),
        verification_adapter: state.verification_adapter.clone(),
        auditor_adapter: state.auditor_adapter.clone(),
        embedding_model: state.embedding_model.clone(),
        researcher_adapter: state.researcher_adapter.clone(),
        knowledge_provider: state.knowledge_provider.clone(),
        tenant_state: Arc::clone(&ts),
        nats_dispatch,
        gap_research_chain: state.gap_research_chain.clone(),
        shadow_audit_ctx,
        shadow_accumulator: state.shadow_accumulator.clone(),
        registry: state.registry(),
        oracle_spec,
        debug_log_path: state.cfg.debug_log_path.clone(),
        skill_provider: Arc::clone(&state.skill_provider),
    };

    tokio::spawn(async move {
        let _permit = permit;
        run_task_pipeline(pipeline_input).await;
    });

    let events_url = format!("/tenants/{}/tasks/{task_id_str}/events", manifest.tenant_id);
    let response = TaskAccepted {
        task_id: task_id_str,
        status: "accepted".into(),
        events_url,
        topology_kind: topology_kind_str,
        n_max,
        interface_n_max: None,
    };

    Ok((StatusCode::ACCEPTED, Json(response)))
}

fn validate_pareto_weights(manifest: &TaskManifest) -> Result<(), ApiError> {
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
    Ok(())
}

fn validate_explorer_budget(manifest: &TaskManifest, n_max: f64) -> Result<(), ApiError> {
    if manifest.explorers.count as f64 > n_max {
        return Err(ApiError::ExplorerBudgetExceeded {
            requested: manifest.explorers.count,
            n_max,
        });
    }
    Ok(())
}

fn acquire_permit(state: &AppState) -> Result<tokio::sync::OwnedSemaphorePermit, ApiError> {
    state
        .task_semaphore
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::ServiceUnavailable(format!(
                "server at capacity ({} concurrent tasks)",
                state.cfg.max_concurrent_tasks
            ))
        })
}

pub async fn task_events(
    Path((tenant_id, task_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> axum::response::Response {
    use axum::response::sse::KeepAlive;
    use futures::StreamExt;
    use h2ai_types::identity::TenantId;

    let tid_uuid = match uuid::Uuid::parse_str(&task_id) {
        Ok(u) => u,
        Err(_) => {
            return Sse::new(tokio_stream::empty::<Result<Event, Infallible>>())
                .keep_alive(KeepAlive::default())
                .into_response();
        }
    };
    let tid = TaskId::from_uuid(tid_uuid);
    let tenant = TenantId::from(tenant_id.as_str());

    // Validate ownership before streaming
    if state.store.get_for_tenant(&tid, &tenant).is_none() {
        return (StatusCode::NOT_FOUND, "task not found for this tenant").into_response();
    }
    let from_seq: u64 = 0;

    let nats = state.nats.clone().expect("NATS required for event stream");
    let stream = async_stream::stream! {
        match nats.tail_task_events_boxed(&tid, from_seq).await {
            Err(e) => {
                tracing::error!("tail error: {e}");
            }
            Ok(mut events) => {
                while let Some(item) = events.next().await {
                    match item {
                        Ok((seq, event)) => {
                            // Update local TaskStore for cross-node consistency:
                            // When approval is processed on Node B, Node A's store must converge.
                            match &event {
                                H2AIEvent::MergeResolved(ev) => {
                                    state.store.mark_resolved(&ev.task_id);
                                }
                                H2AIEvent::TaskFailed(ev) => {
                                    state.store.mark_failed(&ev.task_id);
                                }
                                _ => {}
                            }

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
    Path((tenant_id, task_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<Value>, ApiError> {
    use h2ai_types::identity::TenantId;
    let tid_uuid = uuid::Uuid::parse_str(&task_id)
        .map_err(|_| ApiError::InvalidRequest(format!("invalid task_id: {task_id}")))?;
    let tid = TaskId::from_uuid(tid_uuid);
    let tenant = TenantId::from(tenant_id.as_str());
    let ts = state
        .store
        .get_for_tenant(&tid, &tenant)
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
    Path((tenant_id, task_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(body): Json<MergeRequest>,
) -> Result<Json<Value>, ApiError> {
    use h2ai_types::identity::TenantId;
    let tid_uuid = uuid::Uuid::parse_str(&task_id)
        .map_err(|_| ApiError::InvalidRequest(format!("invalid task_id: {task_id}")))?;
    let tid = TaskId::from_uuid(tid_uuid);
    let tenant = TenantId::from(tenant_id.as_str());
    let ts = state
        .store
        .get_for_tenant(&tid, &tenant)
        .ok_or_else(|| ApiError::TaskNotFound(task_id.clone()))?;
    if ts.status == "resolved" {
        return Err(ApiError::TaskAlreadyResolved(task_id));
    }
    let event = H2AIEvent::MergeResolved(h2ai_types::events::MergeResolvedEvent {
        task_id: tid.clone(),
        resolved_output: body
            .final_output
            .unwrap_or_else(|| body.selected_proposals.join(", ")),
        j_eff: None,
        timestamp: chrono::Utc::now(),
        oracle_gate_passed: None,
        zone3_hints: None,
        contradiction_analysis: None,
    });
    state
        .nats
        .as_ref()
        .expect("NATS required")
        .publish_event(&tid, &event)
        .await
        .map_err(|e| ApiError::NatsUnavailable(e.to_string()))?;
    state.store.mark_resolved(&tid);
    Ok(Json(json!({"status": "resolved", "task_id": task_id})))
}

#[derive(Deserialize)]
pub struct ClarifyRequest {
    pub answer: String,
}

pub async fn clarify_task(
    Path((tenant_id, task_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(body): Json<ClarifyRequest>,
) -> impl IntoResponse {
    use h2ai_types::identity::TenantId;
    let tenant = TenantId::from(tenant_id.as_str());
    let tid_uuid = match uuid::Uuid::parse_str(&task_id) {
        Ok(u) => u,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid task_id"})),
            )
                .into_response();
        }
    };
    let tid = TaskId::from_uuid(tid_uuid);
    // Validate ownership before accepting clarification
    if state.store.get_for_tenant(&tid, &tenant).is_none() {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "task not found for this tenant"})),
        )
            .into_response();
    }
    let waiters = state.clarification_waiters.lock().unwrap();
    if let Some((notify, answer_slot)) = waiters.get(&task_id) {
        *answer_slot.lock().unwrap() = Some(body.answer);
        notify.notify_one();
        (
            StatusCode::OK,
            Json(serde_json::json!({"status": "clarification received"})),
        )
            .into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "no pending clarification for this task"})),
        )
            .into_response()
    }
}
