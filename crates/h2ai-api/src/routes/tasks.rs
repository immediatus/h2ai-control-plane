use crate::{error::ApiError, state::AppState};
use crate::task_pipeline::{run_task_pipeline, TaskPipelineInput};
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

fn compute_j_eff_raw(n_valid: usize, n_agents: usize, p_mean: f64, rho_mean: f64) -> Option<f64> {
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

pub(crate) fn compute_j_eff(
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
    let (srani_ema_cfi, srani_count) = *ts.srani_state.read().await;

    let nats_dispatch = state.agent_provider.as_ref().map(|provider| NatsDispatchConfig {
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
        srani_ema_cfi,
        srani_count,
        srani_grounding_chain: state.srani_grounding_chain.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::{get, post},
        Router,
    };
    use h2ai_config::H2AIConfig;
    use h2ai_test_utils::{mock_adapter, MockNatsBackend};
    use h2ai_types::config::ParetoWeights;
    use h2ai_types::events::CalibrationCompletedEvent;
    use h2ai_types::identity::{TaskId, TenantId};
    use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
    use h2ai_types::sizing::CoherencyCoefficients;
    use serde_json::json;
    use std::sync::Arc;
    use tower::ServiceExt;

    // ── Shared helpers ───────────────────────────────────────────────────────

    fn test_cfg() -> H2AIConfig {
        H2AIConfig::default()
    }

    fn test_state() -> crate::state::AppState {
        let adapter = Arc::new(mock_adapter("answer"));
        let auditor = Arc::new(mock_adapter(r#"{"approved":true,"reason":"ok"}"#));
        crate::state::AppState::new_for_tests(test_cfg(), vec![adapter], auditor)
    }

    fn task_app(state: crate::state::AppState) -> Router {
        Router::new()
            .route("/:tenant_id/tasks", post(submit_task))
            .route("/:tenant_id/tasks/:task_id", get(task_status))
            .route("/:tenant_id/tasks/:task_id/merge", post(merge_task))
            .route("/:tenant_id/tasks/:task_id/clarify", post(clarify_task))
            .with_state(state)
    }

    /// Minimal CalibrationCompletedEvent — n_max ≈ 13.
    /// n_max = sqrt((1 − 0.1) / (0.01 × (1 − 0.5))) = sqrt(180) ≈ 13.
    fn minimal_calibration() -> CalibrationCompletedEvent {
        use h2ai_types::sizing::CoordinationThreshold;
        let coefficients = CoherencyCoefficients {
            alpha: 0.1,
            beta_base: 0.01,
            beta_quality: None,
            cg_samples: vec![0.5],
            sample_timestamps: vec![],
        };
        let threshold = CoordinationThreshold::from_calibration(&coefficients, 1.0);
        CalibrationCompletedEvent {
            calibration_id: TaskId::new(),
            coefficients,
            coordination_threshold: threshold,
            ensemble: None,
            eigen: None,
            timestamp: chrono::Utc::now(),
            pairwise_beta: None,
            cg_mode: Default::default(),
            adapter_families: vec![],
            explorer_verification_family_match: false,
            single_family_warning: false,
            n_max_lo: 0.0,
            n_max_hi: 0.0,
            n_eff_cosine_prior: 0.0,
            calibration_quality: Default::default(),
            calibration_source: Default::default(),
            beta_quality: None,
        }
    }

    async fn seed_calibration(app_state: &crate::state::AppState) {
        let ts = app_state.tenant_state(&TenantId::default_tenant());
        *ts.calibration.write().await = Some(minimal_calibration());
    }

    fn valid_manifest(explorer_count: usize) -> TaskManifest {
        TaskManifest {
            description: "test task".into(),
            pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
            topology: TopologyRequest { kind: "ensemble".into(), branching_factor: None },
            explorers: ExplorerRequest {
                count: explorer_count,
                tau_min: Some(0.3),
                tau_max: Some(0.7),
                roles: vec![],
                review_gates: vec![],
                slot_configs: vec![],
                diversity_ids: vec![],
            },
            constraints: vec![],
            context: None,
            oracle: None,
            require_approval: false,
            constraint_tags: vec![],
            measure_verifier_ab: false,
            tenant_id: TenantId::default_tenant(),
        }
    }

    fn post_json(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    fn get_req(uri: &str) -> Request<Body> {
        Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap()
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ── Task 1: pure functions ───────────────────────────────────────────────

    #[test]
    fn compute_j_eff_raw_returns_none_when_n_agents_zero() {
        // q_ceiling = condorcet_quality(0, 0.8, 0.0) = 0 → None
        let result = compute_j_eff_raw(0, 0, 0.8, 0.1);
        assert!(result.is_none());
    }

    #[test]
    fn compute_j_eff_raw_returns_one_when_all_valid() {
        // n_valid == n_agents → filter_ratio = 1.0 → q_realized == q_ceiling → 1.0
        let result = compute_j_eff_raw(5, 5, 0.7, 0.2);
        assert!(result.is_some());
        let v = result.unwrap();
        assert!((v - 1.0).abs() < 1e-9, "expected 1.0, got {v}");
    }

    #[test]
    fn compute_j_eff_raw_clamps_ratio_to_zero_when_no_valid() {
        // 0 valid out of 5 → q_realized = 0, q_ceiling > 0 → result = 0.0
        let result = compute_j_eff_raw(0, 5, 0.7, 0.0);
        assert!(result.is_some());
        let v = result.unwrap();
        assert!(v >= 0.0 && v <= 1.0, "out of [0,1]: {v}");
        assert_eq!(v, 0.0);
    }

    #[test]
    fn compute_j_eff_uses_ensemble_p_and_rho_when_present() {
        use h2ai_types::sizing::EnsembleCalibration;
        let mut cal = minimal_calibration();
        cal.ensemble = Some(EnsembleCalibration {
            p_mean: 0.8,
            rho_mean: 0.1,
            n_optimal: 3,
            q_optimal: 0.9,
            prediction_basis: Default::default(),
        });
        let result = compute_j_eff(3, 5, &cal);
        assert!(result.is_some());
    }

    #[test]
    fn compute_j_eff_falls_back_to_defaults_when_no_ensemble() {
        // ensemble = None → p_mean=0.5, rho_mean=0.0
        let cal = minimal_calibration();
        let result = compute_j_eff(2, 3, &cal);
        assert!(result.is_some());
    }

    // ── Task 2: task_status ──────────────────────────────────────────────────

    #[tokio::test]
    async fn task_status_returns_400_for_invalid_uuid() {
        let state = test_state();
        let app = task_app(state);
        let resp = app.oneshot(get_req("/default/tasks/not-a-uuid")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn task_status_returns_404_for_unknown_task() {
        let state = test_state();
        let app = task_app(state);
        let fake_id = TaskId::new().to_string();
        let resp = app
            .oneshot(get_req(&format!("/default/tasks/{fake_id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn task_status_returns_200_with_phase_for_known_task() {
        use h2ai_orchestrator::task_store::TaskState;
        let state = test_state();
        let task_id = TaskId::new();
        state.store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), TenantId::default_tenant()),
        );
        let app = task_app(state);
        let resp = app
            .oneshot(get_req(&format!("/default/tasks/{task_id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["task_id"], task_id.to_string());
        assert_eq!(v["phase"], 1);
    }

    #[tokio::test]
    async fn task_status_returns_404_when_tenant_mismatch() {
        use h2ai_orchestrator::task_store::TaskState;
        let state = test_state();
        let task_id = TaskId::new();
        state.store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), TenantId::from("tenant-a")),
        );
        let app = task_app(state);
        // query under "tenant-b" — different tenant
        let resp = app
            .oneshot(get_req(&format!("/tenant-b/tasks/{task_id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── Task 3: clarify_task ─────────────────────────────────────────────────

    #[tokio::test]
    async fn clarify_task_returns_400_for_invalid_uuid() {
        let state = test_state();
        let app = task_app(state);
        let resp = app
            .oneshot(post_json("/default/tasks/bad-uuid/clarify", json!({"answer": "yes"})))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn clarify_task_returns_404_when_task_not_in_store() {
        let state = test_state();
        let fake_id = TaskId::new().to_string();
        let app = task_app(state);
        let resp = app
            .oneshot(post_json(
                &format!("/default/tasks/{fake_id}/clarify"),
                json!({"answer": "yes"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn clarify_task_returns_404_when_no_pending_waiter() {
        use h2ai_orchestrator::task_store::TaskState;
        let state = test_state();
        let task_id = TaskId::new();
        state.store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), TenantId::default_tenant()),
        );
        let app = task_app(state);
        let resp = app
            .oneshot(post_json(
                &format!("/default/tasks/{task_id}/clarify"),
                json!({"answer": "42"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let v = body_json(resp).await;
        assert_eq!(v["error"], "no pending clarification for this task");
    }

    #[tokio::test]
    async fn clarify_task_returns_200_and_notifies_waiter() {
        use h2ai_orchestrator::task_store::TaskState;
        use tokio::sync::Notify;
        let state = test_state();
        let task_id = TaskId::new();
        state.store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), TenantId::default_tenant()),
        );
        let notify = Arc::new(Notify::new());
        let slot: Arc<std::sync::Mutex<Option<String>>> = Arc::new(std::sync::Mutex::new(None));
        {
            let mut waiters = state.clarification_waiters.lock().unwrap();
            waiters.insert(task_id.to_string(), (notify.clone(), slot.clone()));
        }
        let app = task_app(state);
        let resp = app
            .oneshot(post_json(
                &format!("/default/tasks/{task_id}/clarify"),
                json!({"answer": "my answer"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(slot.lock().unwrap().as_deref(), Some("my answer"));
    }

    // ── Task 4: submit_task validation ───────────────────────────────────────

    #[tokio::test]
    async fn submit_task_rejects_pareto_weights_not_summing_to_one() {
        let state = test_state();
        let app = task_app(state);
        let bad = json!({
            "description": "test",
            "pareto_weights": {"diversity": 0.5, "containment": 0.5, "throughput": 0.5},
            "topology": {"kind": "ensemble"},
            "explorers": {"count": 2, "tau_min": 0.3, "tau_max": 0.7}
        });
        let resp = app.oneshot(post_json("/default/tasks", bad)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"], "InvalidRequest");
        assert!(v["message"].as_str().unwrap().contains("pareto_weights"));
    }

    #[tokio::test]
    async fn submit_task_returns_503_when_calibration_missing() {
        let state = test_state();
        let app = task_app(state);
        let manifest = serde_json::to_value(valid_manifest(2)).unwrap();
        let resp = app.oneshot(post_json("/default/tasks", manifest)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let v = body_json(resp).await;
        assert_eq!(v["error"], "CalibrationRequiredError");
    }

    #[tokio::test]
    async fn submit_task_returns_400_when_explorer_count_exceeds_n_max() {
        let state = test_state();
        seed_calibration(&state).await; // n_max ≈ 13
        let app = task_app(state);
        let manifest = serde_json::to_value(valid_manifest(20)).unwrap();
        let resp = app.oneshot(post_json("/default/tasks", manifest)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"], "ExplorerBudgetExceeded");
        assert_eq!(v["requested"], 20);
    }

    #[tokio::test]
    async fn submit_task_returns_503_when_semaphore_exhausted() {
        use tokio::sync::Semaphore;
        let mut state = test_state();
        seed_calibration(&state).await;
        state.task_semaphore = Arc::new(Semaphore::new(0));
        let app = task_app(state);
        let manifest = serde_json::to_value(valid_manifest(2)).unwrap();
        let resp = app.oneshot(post_json("/default/tasks", manifest)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let v = body_json(resp).await;
        assert_eq!(v["error"], "ServiceUnavailable");
    }

    #[tokio::test]
    async fn submit_task_returns_202_with_accepted_body() {
        let state = test_state();
        seed_calibration(&state).await;
        let app = task_app(state);
        let manifest = serde_json::to_value(valid_manifest(2)).unwrap();
        let resp = app.oneshot(post_json("/default/tasks", manifest)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);
        let v = body_json(resp).await;
        assert!(v["task_id"].is_string(), "task_id missing");
        assert_eq!(v["status"], "accepted");
        assert!(
            v["events_url"].as_str().unwrap().ends_with("/events"),
            "events_url should end with /events"
        );
        assert_eq!(v["topology_kind"], "ensemble");
    }

    // ── Task 5: merge_task ───────────────────────────────────────────────────

    #[tokio::test]
    async fn merge_task_returns_400_for_invalid_task_id_uuid() {
        let state = test_state();
        let app = task_app(state);
        let resp = app
            .oneshot(post_json(
                "/default/tasks/not-a-uuid/merge",
                json!({"resolution": "select", "selected_proposals": ["p1"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn merge_task_returns_404_for_unknown_task() {
        let state = test_state();
        let app = task_app(state);
        let fake_id = TaskId::new().to_string();
        let resp = app
            .oneshot(post_json(
                &format!("/default/tasks/{fake_id}/merge"),
                json!({"resolution": "select", "selected_proposals": ["p1"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn merge_task_returns_409_when_already_resolved() {
        use h2ai_orchestrator::task_store::TaskState;
        let mut state = test_state();
        // Wire a no-expectation mock — publish_event must NOT be called
        let mock = MockNatsBackend::new();
        state.nats = Some(Arc::new(mock));
        let task_id = TaskId::new();
        let mut ts = TaskState::new(task_id.clone(), TenantId::default_tenant());
        ts.status = "resolved".into();
        state.store.insert(task_id.clone(), ts);
        let app = task_app(state);
        let resp = app
            .oneshot(post_json(
                &format!("/default/tasks/{task_id}/merge"),
                json!({"resolution": "select", "selected_proposals": ["p1"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn merge_task_publishes_event_and_returns_resolved() {
        use h2ai_orchestrator::task_store::TaskState;
        let mut state = test_state();
        let mut mock = MockNatsBackend::new();
        mock.expect_publish_event().once().returning(|_, _| Ok(()));
        state.nats = Some(Arc::new(mock));
        let task_id = TaskId::new();
        state.store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), TenantId::default_tenant()),
        );
        let app = task_app(state);
        let resp = app
            .oneshot(post_json(
                &format!("/default/tasks/{task_id}/merge"),
                json!({"resolution": "select", "selected_proposals": ["Proposal A"], "final_output": "Proposal A"}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["status"], "resolved");
        assert_eq!(v["task_id"], task_id.to_string());
    }

    #[tokio::test]
    async fn merge_task_returns_503_when_nats_publish_fails() {
        use h2ai_orchestrator::task_store::TaskState;
        use h2ai_state::nats::NatsError;
        let mut state = test_state();
        let mut mock = MockNatsBackend::new();
        mock.expect_publish_event()
            .once()
            .returning(|_, _| Err(NatsError::KvError("mock nats failure".into())));
        state.nats = Some(Arc::new(mock));
        let task_id = TaskId::new();
        state.store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), TenantId::default_tenant()),
        );
        let app = task_app(state);
        let resp = app
            .oneshot(post_json(
                &format!("/default/tasks/{task_id}/merge"),
                json!({"resolution": "select", "selected_proposals": ["p1"]}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let v = body_json(resp).await;
        assert_eq!(v["error"], "NatsUnavailable");
    }
}
