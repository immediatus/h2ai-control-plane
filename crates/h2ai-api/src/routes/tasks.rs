use crate::{error::ApiError, state::AppState};

const THINKING_LOOP_SECTION: &str = "## Thinking Loop Analysis";
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::Event,
    response::{IntoResponse, Json, Sse},
};
use h2ai_orchestrator::decomposition::{
    compute_role_diversity, prune_by_orthogonality, run_decomposition_agent,
};
use h2ai_orchestrator::engine::{EngineError, EngineInput, ExecutionEngine, NatsDispatchConfig};
use h2ai_orchestrator::thinking_loop::{self, ThinkingLoopInput};
use h2ai_types::agent::{AgentDescriptor, AgentTool, CostTier, TaskRequirements};
use h2ai_types::checkpoint::ConstraintSnapshot;
use h2ai_types::config::{TaoConfig, VerificationConfig};
use h2ai_types::events::{
    CoherenceIncompleteEvent, H2AIEvent, MergeResolvedEvent, TaskAttributionEvent, TaskFailedEvent,
    ThinkingLoopCompletedEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{MergeRequest, TaskAccepted, TaskManifest};
use h2ai_types::prompts::ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT;
use h2ai_types::sizing::TaskQuadrant;
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
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

fn compute_j_eff(
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

/// Accept a [`TaskManifest`] and begin async execution, returning `202 Accepted` immediately.
///
/// Performs the following validation before spawning:
/// - Pareto weights (`diversity + containment + throughput`) must sum to 1.0 (±1e-4).
/// - A completed [`CalibrationCompletedEvent`] must be present; returns
///   `ApiError::CalibrationRequired` otherwise.
/// - `manifest.explorers.count` must not exceed `calibration.coefficients.n_max()`;
///   returns `ApiError::ExplorerBudgetExceeded` otherwise.
/// - A semaphore permit must be available (`cfg.max_concurrent_tasks`); returns
///   `ApiError::ServiceUnavailable` when the server is at capacity.
///
/// On success the handler inserts the task into the store, spawns a Tokio task that runs
/// [`ExecutionEngine::run_offline`], and returns `202 Accepted` with a [`TaskAccepted`]
/// body containing the task ID, status URL, `J_eff` score, and topology kind.
/// When the engine finishes it publishes `H2AIEvent::VerificationScored` events to NATS
/// for each scored proposal, followed by a single `H2AIEvent::TaskAttribution` event
/// with quality metrics and waste analysis, then marks the task resolved in the store.
pub async fn submit_task(
    Path(tenant_id): Path<String>,
    State(state): State<AppState>,
    Json(mut manifest): Json<TaskManifest>,
) -> Result<impl IntoResponse, ApiError> {
    // Path tenant_id is authoritative — override whatever the body may contain.
    manifest.tenant_id = h2ai_types::identity::TenantId::from(tenant_id.as_str());
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
    state.store.insert(
        task_id.clone(),
        TaskState::new(task_id.clone(), manifest.tenant_id.clone()),
    );

    let adapter_pool = state.adapter_pool.clone();
    let verifier = state.verification_adapter.clone();
    let auditor = state.auditor_adapter.clone();
    let shadow_auditor_adapter = state.shadow_auditor_adapter.clone();
    let shadow_accumulator = state.shadow_accumulator.clone();
    let registry = state.registry();

    let resolved_ids_for_checkpoint = resolved_ids;
    let wiki_revision_for_checkpoint = wiki_revision;
    let evaluated_ids_for_checkpoint: Vec<String> = corpus.iter().map(|d| d.id.clone()).collect();

    let state_clone = state.clone();
    let ts_clone = std::sync::Arc::clone(&ts);
    let task_tenant_id_clone = task_tenant_id.clone();
    let manifest_clone = manifest.clone();
    let calibration_clone = calibration.clone();
    let store_clone = state.store.clone();
    let task_id_clone = task_id;

    tokio::spawn(async move {
        let _permit = permit; // dropped when this task completes, freeing semaphore slot
        let ts = ts_clone;
        let task_tenant_id = task_tenant_id_clone;
        let tao_multiplier = ts.tao_multiplier_estimator.read().await.multiplier();
        let tao_multiplier_estimator = std::sync::Arc::clone(&ts.tao_multiplier_estimator);
        let bandit = std::sync::Arc::clone(&ts.bandit_state);
        let task_id_for_failure = task_id_clone.clone();
        let oracle_spec_clone = manifest_clone.oracle.clone();
        // Pre-serialize manifest for checkpoint (manifest_clone is moved into input below).
        let manifest_json_for_checkpoint =
            serde_json::to_string(&manifest_clone).unwrap_or_default();
        let nats_dispatch =
            state_clone
                .agent_provider
                .as_ref()
                .map(|provider| NatsDispatchConfig {
                    nats: state_clone
                        .nats
                        .clone()
                        .expect("NATS required for agent dispatch"),
                    provider: std::sync::Arc::clone(provider),
                    agent_descriptor: AgentDescriptor {
                        model: state_clone.cfg.nats_agent_model.clone(),
                        tools: vec![AgentTool::Shell, AgentTool::FileSystem],
                        cost_tier: CostTier::Mid,
                    },
                    task_requirements: TaskRequirements {
                        max_cost_tier: CostTier::High,
                        required_tools: vec![AgentTool::Shell, AgentTool::FileSystem],
                    },
                    task_timeout: Duration::from_secs(state_clone.cfg.nats_agent_timeout_secs),
                    payload_store: state_clone.payload_store.clone(),
                    offload_threshold_bytes: 8 * 1024,
                });
        // Phase 0: LLM decomposition always runs.
        // Operator slot_configs (if any) are appended after and the combined set is
        // Manifest count is a hard upper bound — submitter chose it deliberately.
        // USL n_max() is a throughput ceiling but can exceed the task's stated count.
        let n_max = (calibration_clone.coefficients.n_max() as usize)
            .min(manifest_clone.explorers.count)
            .max(1);
        // Quality-driven target: one slot per constraint domain + one integration slot.
        // n_max (USL throughput ceiling) is the hard cap; n_target drives the prompt.
        let n_domains = corpus
            .iter()
            .flat_map(|d| d.domains.iter())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let n_target = (n_domains + 1).max(2).min(n_max.max(1));
        let thinking_task_id = task_id_clone.to_string();
        let thinking_nats_client = state_clone.nats.as_ref().map(|n| n.client.clone());
        let thinking_constraint_tags: Vec<String> = corpus
            .iter()
            .flat_map(|d| d.domains.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let thinking_report = thinking_loop::run(ThinkingLoopInput {
            task_description: &manifest_clone.description,
            constraint_ids: &corpus.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
            constraint_tags: &thinking_constraint_tags,
            research_context: "",
            knowledge_provider: Some(state_clone.knowledge_provider.clone()),
            n_archetypes: state_clone.cfg.thinking_loop.max_archetypes,
            cfg: &state_clone.cfg.thinking_loop,
            adapter: adapter_pool[0].as_ref(),
            embedding_model: state_clone.embedding_model.as_deref(),
            nats_client: thinking_nats_client,
            task_id: &thinking_task_id,
        })
        .await;
        {
            let tl_ev = H2AIEvent::ThinkingLoopCompleted(ThinkingLoopCompletedEvent {
                task_id: task_id_clone.clone(),
                enabled: state_clone.cfg.thinking_loop.enabled,
                iterations_run: thinking_report.iteration,
                coverage_score: thinking_report.coverage_score,
                shared_understanding_len: thinking_report.shared_understanding.len(),
                archetypes: vec![], // archetype names not carried on ThinkingReport
                timestamp: chrono::Utc::now(),
            });
            if let Some(nats) = &state_clone.nats {
                if let Err(e) = nats.publish_event(&task_id_clone, &tl_ev).await {
                    tracing::warn!(task_id = %task_id_clone, "failed to publish ThinkingLoopCompletedEvent: {e}");
                }
            }
        }
        let thinking_context = if thinking_report.shared_understanding.is_empty() {
            String::new()
        } else {
            thinking_report.shared_understanding.clone()
        };
        let slot_configs = match run_decomposition_agent(
            &manifest_clone.description,
            &corpus,
            &manifest_clone.pareto_weights,
            n_target,
            n_max.max(1),
            adapter_pool[0].as_ref(),
            state_clone.embedding_model.as_deref(),
            state_clone.cfg.decomposition_step_max_tokens,
            state_clone.cfg.decomposition_json_max_tokens,
            &thinking_context,
        )
        .await
        {
            Ok(mut slots) => {
                // Append operator-specified extra slots, then re-prune.
                let operator_extra = manifest_clone.explorers.slot_configs.clone();
                if !operator_extra.is_empty() {
                    slots.extend(operator_extra);
                    if let Some(model) = state_clone.embedding_model.as_deref() {
                        slots = prune_by_orthogonality(slots, n_max.max(1), model);
                    } else {
                        slots.truncate(n_max.max(1));
                    }
                }
                let role_diversity =
                    compute_role_diversity(&slots, state_clone.embedding_model.as_deref());
                tracing::info!(
                    target: "h2ai.decomposition",
                    n_slots = slots.len(),
                    n_eff_cosine_roles = role_diversity,
                    embedding_blind = state_clone.embedding_model.is_none(),
                    "decomposition produced slots"
                );
                for (i, s) in slots.iter().enumerate() {
                    let mandate = if s.focus_mandate.chars().count() > 60 {
                        format!(
                            "[truncated] {}…",
                            s.focus_mandate.chars().take(60).collect::<String>()
                        )
                    } else {
                        s.focus_mandate.clone()
                    };
                    let role = if s.role_frame.chars().count() > 120 {
                        format!(
                            "[truncated] {}…",
                            s.role_frame.chars().take(120).collect::<String>()
                        )
                    } else {
                        s.role_frame.clone()
                    };
                    tracing::info!(
                        target: "h2ai.decomposition",
                        slot = i,
                        cot_style = ?s.cot_style,
                        mandate = %mandate,
                        role = %role,
                        "slot config"
                    );
                }
                slots
            }
            Err(e) => {
                tracing::error!(
                    target: "h2ai.decomposition",
                    error = %e,
                    "decomposition failed — task cannot proceed without an epistemic committee"
                );
                let failed_ev = H2AIEvent::TaskFailed(TaskFailedEvent {
                    task_id: task_id_clone.clone(),
                    pruned_events: vec![],
                    topologies_tried: vec![],
                    tau_values_tried: vec![],
                    multiplication_condition_failure: None,
                    timestamp: chrono::Utc::now(),
                });
                if let Err(pub_err) = state_clone
                    .nats
                    .as_ref()
                    .expect("NATS required")
                    .publish_event(&task_id_clone, &failed_ev)
                    .await
                {
                    tracing::warn!(
                        "failed to publish TaskFailedEvent after decomposition failure: {pub_err}"
                    );
                }
                state_clone.store.mark_failed(&task_id_clone);
                return;
            }
        };
        let use_adversarial_verifier = slot_configs
            .iter()
            .any(|s| !s.rejection_criteria.is_empty());
        let calibration_source_for_attr = calibration_clone.calibration_source;

        // Build shadow audit context: clone the promoted-domains snapshot at task start.
        let shadow_ctx = shadow_auditor_adapter.as_ref().map(|adapter| {
            let promoted_snap = {
                // Synchronous snapshot — we can't await here inside a non-async let,
                // so we take a blocking read by temporarily blocking on the async lock.
                // This is cheap (HashSet clone) and happens once per task submission.
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current()
                        .block_on(state_clone.promoted_audit_domains.read())
                })
                .clone()
            };
            h2ai_orchestrator::engine::ShadowAuditCtx {
                adapter: adapter.clone(),
                promoted_domains: promoted_snap,
            }
        });

        let (srani_ema_cfi, srani_count) = *ts.srani_state.read().await;

        let calibration_for_merge = calibration_clone.clone();
        let diversity_ids: Vec<u32> = if manifest_clone.explorers.diversity_ids.is_empty() {
            (0..manifest_clone.explorers.count as u32).collect()
        } else {
            manifest_clone.explorers.diversity_ids.clone()
        };
        let pool_len = adapter_pool.len().max(1);
        // Resolve Arc refs for each diversity ID so they outlive the EngineInput borrow.
        let explorer_arcs: Vec<std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>> =
            diversity_ids
                .iter()
                .map(|id| adapter_pool[*id as usize % pool_len].clone())
                .collect();
        let input = EngineInput {
            task_id: task_id_clone,
            manifest: {
                let mut m = manifest_clone.clone();
                m.explorers.slot_configs = slot_configs;
                // Inject thinking loop shared_understanding into context so explorers address
                // the specific requirements the thinking loop identified.
                if !thinking_context.is_empty() {
                    m.context = Some(match m.context.as_deref() {
                        Some(ctx) if !ctx.is_empty() => {
                            format!("{ctx}\n\n{THINKING_LOOP_SECTION}\n{thinking_context}")
                        }
                        _ => format!("{THINKING_LOOP_SECTION}\n{thinking_context}"),
                    });
                }
                m
            },
            calibration: calibration_clone,
            explorer_adapters: explorer_arcs
                .iter()
                .map(std::convert::AsRef::as_ref)
                .collect(),
            verification_adapter: verifier.as_ref(),
            auditor_adapter: auditor.as_ref(),
            auditor_config: h2ai_types::config::AuditorConfig {
                adapter: auditor.kind().clone(),
                ..Default::default()
            },
            tao_config: TaoConfig::default(),
            verification_config: if use_adversarial_verifier {
                VerificationConfig {
                    threshold: state_clone.cfg.verify_threshold,
                    evaluator_system_prompt: ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT.into(),
                    record_adversarial_comparison: manifest_clone.measure_verifier_ab,
                    ..VerificationConfig::default()
                }
            } else {
                VerificationConfig {
                    threshold: state_clone.cfg.verify_threshold,
                    record_adversarial_comparison: manifest_clone.measure_verifier_ab,
                    ..VerificationConfig::default()
                }
            },
            constraint_corpus: corpus,
            cfg: &state_clone.cfg,
            store: store_clone,
            nats_dispatch,
            registry: &registry,
            embedding_model: state_clone.embedding_model.as_deref(),
            tao_multiplier,
            tao_estimator: tao_multiplier_estimator,
            synthesis_adapter: None,
            bandit_state: Some(bandit),
            shadow_audit_ctx: shadow_ctx,
            researcher_adapter: state_clone.researcher_adapter.clone(),
            srani_ema_cfi,
            srani_count,
            srani_grounding_chain: state_clone.srani_grounding_chain.clone(),
            nats_raw: None,
            tenant_id: manifest_clone.tenant_id.clone(),
            nats: state_clone.nats.clone(),
            prev_assembled_contexts: Vec::new(),
            compression_adapter: None,
            stable_cache: None,
            knowledge_provider: Some(state_clone.knowledge_provider.clone()),
            induction_store: None,
        };

        match ExecutionEngine::run_offline(input).await {
            Ok(output) => {
                // Phase checkpoint: save resolved output before publishing events (best-effort).
                {
                    use h2ai_types::checkpoint::TaskCheckpoint;
                    let node_id = format!(
                        "{}:{}",
                        hostname::get()
                            .map_or_else(|_| "unknown".into(), |h| h.to_string_lossy().to_string()),
                        std::process::id()
                    );
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    let checkpoint = TaskCheckpoint {
                        task_id: output.task_id.to_string(),
                        phase: "Merging".into(),
                        node_id,
                        lease_seq: 0,
                        proposals: vec![],
                        auditor_survivors: vec![],
                        resolved_output: Some(output.resolved_output.clone()),
                        manifest_json: manifest_json_for_checkpoint.clone(),
                        object_store_ref: None,
                        created_at_ms: now_ms,
                        updated_at_ms: now_ms,
                        constraint_snapshot: Some(ConstraintSnapshot {
                            wiki_revision: wiki_revision_for_checkpoint,
                            resolved_ids: resolved_ids_for_checkpoint,
                            evaluated_ids: evaluated_ids_for_checkpoint,
                            violation_ids: vec![],
                        }),
                        j_eff: compute_j_eff(
                            output.selection_resolved.valid_proposals.len(),
                            manifest_clone.explorers.count,
                            &calibration_for_merge,
                        ),
                    };
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .put_task_checkpoint(&checkpoint, None)
                        .await
                    {
                        tracing::warn!(task_id = %output.task_id, "checkpoint write failed (best-effort): {e}");
                    }
                }
                // Debug NDJSON log: append one JSON line when debug_log_path is set.
                // Must run before partial moves from output (e.g. applied_optimizations).
                if let Some(ref log_path) = state_clone.cfg.debug_log_path {
                    let record = crate::debug_record::TaskDebugRecord::build(
                        &manifest_clone.description,
                        srani_ema_cfi,
                        srani_count,
                        &output,
                        &state_clone.cfg,
                    );
                    crate::debug_record::append_debug_record(log_path, &record);
                }
                // Update Prometheus metrics from engine output
                {
                    let mut metrics = state_clone.metrics.write().await;
                    metrics.mapek_mode_collapse_count += output.mode_collapse_count as u64;
                    let constrained = output
                        .topology_retry_events
                        .len()
                        .saturating_sub(output.mode_collapse_count);
                    metrics.mapek_constrained_exploration_count += constrained as u64;
                    // Phase 1.5 quadrant distribution — used to validate θ_tcc before
                    // shadow_mode can be disabled (see gap-a1-solution.md §4.3).
                    match output.complexity_event.task_quadrant {
                        TaskQuadrant::Precision => metrics.phase15_quadrant_precision += 1,
                        TaskQuadrant::Coverage => metrics.phase15_quadrant_coverage += 1,
                        TaskQuadrant::Complex => metrics.phase15_quadrant_complex += 1,
                        TaskQuadrant::Degenerate => metrics.phase15_quadrant_degenerate += 1,
                    }
                    metrics.oracle_tasks_total += 1;
                    if oracle_spec_clone.is_some() {
                        metrics.oracle_tasks_with_spec += 1;
                    }
                    metrics.oracle_coverage_rate = if metrics.oracle_tasks_total > 0 {
                        metrics.oracle_tasks_with_spec as f64 / metrics.oracle_tasks_total as f64
                    } else {
                        0.0
                    };
                }
                let complexity_ev =
                    H2AIEvent::TaskComplexityAssessed(output.complexity_event.clone());
                match state_clone
                    .nats
                    .as_ref()
                    .expect("NATS required")
                    .publish_event_seq(&output.task_id, &complexity_ev)
                    .await
                {
                    Ok(seq) => {
                        if let Some(task_state) = state_clone.store.get(&output.task_id) {
                            state_clone
                                .journal
                                .note_event(&output.task_id, seq, &task_state);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("failed to publish TaskComplexityAssessedEvent: {e}");
                    }
                }

                if let Some(ref frontier_ev) = output.frontier_event {
                    let h2ai_ev = H2AIEvent::ConstraintFrontier(frontier_ev.clone());
                    match state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &h2ai_ev)
                        .await
                    {
                        Ok(seq) => {
                            if let Some(task_state) = state_clone.store.get(&output.task_id) {
                                state_clone
                                    .journal
                                    .note_event(&output.task_id, seq, &task_state);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("failed to publish ConstraintFrontierEvent: {e}");
                        }
                    }
                }

                // INNOVATION-3 (GAP-A3): update online ρ EMA from this task's verification scores.
                {
                    let scores: Vec<(String, f64)> = output
                        .verification_events
                        .iter()
                        .map(|e| (e.explorer_id.to_string(), e.score))
                        .collect();

                    if scores.len() >= 2 {
                        let p_mean = {
                            let cal = ts.calibration.read().await;
                            cal.as_ref()
                                .and_then(|c| c.ensemble.as_ref())
                                .map_or(0.7_f64, |e| e.p_mean)
                        };
                        let variance = (p_mean * (1.0 - p_mean)).max(0.01);
                        let mut pairs: Vec<(String, String, f64)> = Vec::new();
                        for i in 0..scores.len() {
                            for j in (i + 1)..scores.len() {
                                let (id_a, s_a) = &scores[i];
                                let (id_b, s_b) = &scores[j];
                                let product =
                                    ((s_a - p_mean) * (s_b - p_mean) / variance).clamp(-1.0, 1.0);
                                pairs.push((id_a.clone(), id_b.clone(), product));
                            }
                        }

                        let n_obs = {
                            let mut rho_ema = ts.rho_ema.write().await;
                            rho_ema.update(&pairs, 0.10);
                            rho_ema.n_observations
                        };

                        if n_obs >= 30 {
                            let rho_empirical = ts.rho_ema.read().await.rho_mean();
                            let mut cal = ts.calibration.write().await;
                            if let Some(ref mut event) = *cal {
                                if let Some(ref existing_ec) = event.ensemble {
                                    use h2ai_types::sizing::EnsembleCalibration;
                                    event.ensemble = Some(EnsembleCalibration::from_empirical(
                                        existing_ec.p_mean,
                                        rho_empirical,
                                        state_clone.cfg.calibration_max_ensemble_size,
                                    ));
                                }
                            }
                        }
                    }
                }

                for event in output.verification_events {
                    let h2ai_ev = H2AIEvent::VerificationScored(event);
                    match state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &h2ai_ev)
                        .await
                    {
                        Ok(seq) => {
                            if let Some(task_state) = state_clone.store.get(&output.task_id) {
                                state_clone
                                    .journal
                                    .note_event(&output.task_id, seq, &task_state);
                            }
                        }
                        Err(e) => tracing::warn!("failed to publish VerificationScoredEvent: {e}"),
                    }
                }

                for event in output.failed_proposals {
                    let h2ai_ev = H2AIEvent::ProposalFailed(event);
                    match state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &h2ai_ev)
                        .await
                    {
                        Ok(seq) => {
                            if let Some(task_state) = state_clone.store.get(&output.task_id) {
                                state_clone
                                    .journal
                                    .note_event(&output.task_id, seq, &task_state);
                            }
                        }
                        Err(e) => tracing::warn!("failed to publish ProposalFailedEvent: {e}"),
                    }
                }

                let selection_ev = H2AIEvent::SelectionResolved(output.selection_resolved.clone());
                match state_clone
                    .nats
                    .as_ref()
                    .expect("NATS required")
                    .publish_event_seq(&output.task_id, &selection_ev)
                    .await
                {
                    Ok(seq) => {
                        if let Some(task_state) = state_clone.store.get(&output.task_id) {
                            state_clone
                                .journal
                                .note_event(&output.task_id, seq, &task_state);
                        }
                    }
                    Err(e) => tracing::warn!("failed to publish SelectionResolvedEvent: {e}"),
                }

                // Apply τ-spread EMA update when the engine detected waste.
                if !output.applied_optimizations.is_empty() {
                    use h2ai_types::events::OptimizationKind;
                    for opt in &output.applied_optimizations {
                        if opt.kind == OptimizationKind::TauSpreadAdjusted {
                            if let (Ok(before), Ok(after)) =
                                (opt.before.parse::<f64>(), opt.after.parse::<f64>())
                            {
                                let mut est = ts.tau_spread_estimator.write().await;
                                // Use the verify_threshold change as a proxy for τ spread shift.
                                // before/after are verify_threshold values scaled to [0,1].
                                est.update(before.min(after), before.max(after));
                            }
                        }
                    }
                }

                let attr_ev = H2AIEvent::TaskAttribution(TaskAttributionEvent {
                    task_id: output.task_id.clone(),
                    q_confidence: output.attribution.q_confidence,
                    q_measured: output.attribution.q_measured,
                    q_interval_lo: output
                        .attribution_interval
                        .as_ref()
                        .map(|iv| iv.q_confidence_lo),
                    q_interval_hi: output
                        .attribution_interval
                        .as_ref()
                        .map(|iv| iv.q_confidence_hi),
                    prediction_basis: output.attribution.prediction_basis,
                    waste_ratio: output.waste_ratio,
                    applied_optimizations: output.applied_optimizations,
                    timestamp: chrono::Utc::now(),
                    approval_decision: None,
                    calibration_source: calibration_source_for_attr,
                });
                match state_clone
                    .nats
                    .as_ref()
                    .expect("NATS required")
                    .publish_event_seq(&output.task_id, &attr_ev)
                    .await
                {
                    Ok(seq) => {
                        if let Some(task_state) = state_clone.store.get(&output.task_id) {
                            state_clone
                                .journal
                                .note_event(&output.task_id, seq, &task_state);
                        }
                    }
                    Err(e) => tracing::warn!("failed to publish TaskAttributionEvent: {e}"),
                }
                for comp_ev in &output.comparison_events {
                    let ev = H2AIEvent::VerifierComparison(comp_ev.clone());
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &ev)
                        .await
                    {
                        tracing::warn!("failed to publish VerifierComparisonEvent: {e}");
                    }
                }
                // Publish shadow audit events and feed accumulator.
                if !output.shadow_audit_events.is_empty() {
                    for shadow_ev in &output.shadow_audit_events {
                        let ev = H2AIEvent::ShadowAudit(shadow_ev.clone());
                        if let Err(e) = state_clone
                            .nats
                            .as_ref()
                            .expect("NATS required")
                            .publish_event_seq(&output.task_id, &ev)
                            .await
                        {
                            tracing::warn!("failed to publish ShadowAuditorResultEvent: {e}");
                        }
                    }
                    if let Some(ref acc) = shadow_accumulator {
                        acc.lock()
                            .await
                            .process(output.shadow_audit_events.clone())
                            .await;
                    }
                }
                // Publish C1 correlated ensemble warnings
                for warning in &output.correlated_warnings {
                    let ev = H2AIEvent::CorrelatedEnsemble(warning.clone());
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &ev)
                        .await
                    {
                        tracing::warn!("failed to publish CorrelatedEnsembleWarning: {e}");
                    }
                }
                // Publish SRANI correlated fabrication events
                for srani_ev in &output.srani_events {
                    let ev = H2AIEvent::CorrelatedFabrication(srani_ev.clone());
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &ev)
                        .await
                    {
                        tracing::warn!("failed to publish CorrelatedFabricationEvent: {e}");
                    }
                }
                // Persist updated SRANI adaptive EMA state.
                if output.srani_count_updated != srani_count {
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .put_srani_state(
                            &task_tenant_id,
                            output.srani_ema_cfi_updated,
                            output.srani_count_updated,
                        )
                        .await
                    {
                        tracing::warn!("failed to persist srani state: {e}");
                    }
                    *ts.srani_state.write().await =
                        (output.srani_ema_cfi_updated, output.srani_count_updated);
                }
                // Publish researcher grounding events
                for grounding in &output.researcher_grounding_events {
                    let ev = H2AIEvent::ResearcherGrounding(grounding.clone());
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &ev)
                        .await
                    {
                        tracing::warn!("failed to publish ResearcherGroundingEvent: {e}");
                    }
                }
                // Publish C3 diversity guard degraded event
                if let Some(ref degraded) = output.diversity_degraded_event {
                    let ev = H2AIEvent::DiversityGuardDegraded(degraded.clone());
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &ev)
                        .await
                    {
                        tracing::warn!("failed to publish DiversityGuardDegradedEvent: {e}");
                    }
                }
                if !output.coherence_state.is_closed() {
                    let coh_ev = H2AIEvent::CoherenceIncomplete(CoherenceIncompleteEvent {
                        task_id: output.task_id.clone(),
                        uncovered_domains: output.coherence_state.uncovered_domains.clone(),
                        active_contradictions: output
                            .coherence_state
                            .active_contradictions
                            .iter()
                            .map(|(a, b, d)| (a.to_string(), b.to_string(), d.clone()))
                            .collect(),
                        retries: output.topology_retry_events.len() as u32,
                        timestamp: chrono::Utc::now(),
                    });
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &coh_ev)
                        .await
                    {
                        tracing::warn!("failed to publish CoherenceIncompleteEvent: {e}");
                    }
                }
                // Publish leader election events.
                for ev in &output.leader_elected_events {
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(&output.task_id, &H2AIEvent::LeaderElected(ev.clone()))
                        .await
                    {
                        tracing::warn!(
                            task_id = %output.task_id,
                            "failed to publish LeaderElectedEvent: {e}"
                        );
                    }
                }
                for ev in &output.socratic_diagnosis_events {
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .publish_event_seq(
                            &output.task_id,
                            &H2AIEvent::SocraticDiagnosis(ev.clone()),
                        )
                        .await
                    {
                        tracing::warn!(
                            task_id = %output.task_id,
                            "failed to publish SocraticDiagnosisEvent: {e}"
                        );
                    }
                }
                // Publish MergeResolved so SSE clients receive the terminal event and close.
                let j_eff = compute_j_eff(
                    output.selection_resolved.valid_proposals.len(),
                    manifest_clone.explorers.count,
                    &calibration_for_merge,
                );
                let merge_ev = H2AIEvent::MergeResolved(MergeResolvedEvent {
                    task_id: output.task_id.clone(),
                    resolved_output: output.resolved_output.clone(),
                    j_eff,
                    timestamp: chrono::Utc::now(),
                    oracle_gate_passed: None,
                    zone3_hints: None,
                });
                if let Err(e) = state_clone
                    .nats
                    .as_ref()
                    .expect("NATS required")
                    .publish_event(&output.task_id, &merge_ev)
                    .await
                {
                    tracing::warn!("failed to publish MergeResolvedEvent: {e}");
                }
                // Spawn background OPRO trigger — non-blocking, errors logged internally
                if let Some(j_eff_value) = j_eff {
                    let opro_nats = state_clone.nats.clone().expect("NATS required for OPRO");
                    let opro_cfg = std::sync::Arc::clone(&state_clone.cfg);
                    let opro_adapter = std::sync::Arc::clone(&state_clone.adapter_pool[0]);
                    let opro_adapter_name = state_clone
                        .cfg
                        .adapter_profiles
                        .first()
                        .map_or_else(|| "default".to_string(), |p| p.name.clone());
                    tokio::spawn(async move {
                        if let Err(e) = crate::opro::run_opro_trigger(
                            opro_adapter_name,
                            "system_preamble".to_string(),
                            j_eff_value,
                            &opro_nats,
                            opro_adapter.as_ref(),
                            &opro_cfg,
                        )
                        .await
                        {
                            tracing::warn!("OPRO trigger failed: {}", e);
                        }
                    });
                }
                // Phase 6: async oracle dispatch (fire-and-forget, non-blocking)
                if let Some(ref oracle_spec) = oracle_spec_clone {
                    let nats_client = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required for oracle dispatch")
                        .client
                        .clone();
                    let task_id_oracle = output.task_id.clone();
                    let resolved = output.resolved_output.clone();
                    let q = output.attribution.q_confidence;
                    let n_used = output.selection_resolved.n_input_proposals as u32;
                    let spec = oracle_spec.clone();
                    tokio::spawn(async move {
                        h2ai_orchestrator::oracle::oracle_dispatch::fire(
                            &nats_client,
                            task_id_oracle,
                            h2ai_types::identity::TenantId::default(),
                            &resolved,
                            q,
                            n_used,
                            &spec,
                        )
                        .await;
                    });
                }
                state_clone.store.mark_resolved(&output.task_id);
                // GC: delete checkpoint now that task is permanently resolved.
                if let Err(e) = state_clone
                    .nats
                    .as_ref()
                    .expect("NATS required")
                    .delete_task_checkpoint(&output.task_id.to_string())
                    .await
                {
                    tracing::debug!(task_id = %output.task_id, "checkpoint GC on resolve (may not exist): {e}");
                }
            }
            Err(e) => {
                let msg = e.to_string();
                let is_network = msg.contains("network error")
                    || msg.contains("connection refused")
                    || msg.contains("timed out");
                if is_network {
                    tracing::warn!(target: "h2ai.tasks", "task engine stopped — LLM adapter unreachable: {msg}");
                } else {
                    tracing::error!(target: "h2ai.tasks", "task engine error: {msg}");
                }

                // Publish any VerificationScored events collected before failure so SSE
                // clients tracking Phase 3 can observe them even on TaskFailed.
                if let EngineError::MaxRetriesExhausted {
                    partial_verification_events,
                } = &e
                {
                    for event in partial_verification_events {
                        let h2ai_ev = H2AIEvent::VerificationScored(event.clone());
                        if let Err(pub_err) = state_clone
                            .nats
                            .as_ref()
                            .expect("NATS required")
                            .publish_event(&task_id_for_failure, &h2ai_ev)
                            .await
                        {
                            tracing::warn!("failed to publish partial VerificationScoredEvent on failure: {pub_err}");
                        }
                    }
                }

                let failed_ev = H2AIEvent::TaskFailed(TaskFailedEvent {
                    task_id: task_id_for_failure.clone(),
                    pruned_events: vec![],
                    topologies_tried: vec![],
                    tau_values_tried: vec![],
                    multiplication_condition_failure: None,
                    timestamp: chrono::Utc::now(),
                });
                if let Err(pub_err) = state_clone
                    .nats
                    .as_ref()
                    .expect("NATS required")
                    .publish_event(&task_id_for_failure, &failed_ev)
                    .await
                {
                    tracing::warn!("failed to publish TaskFailedEvent: {pub_err}");
                }
                state_clone.store.mark_failed(&task_id_for_failure);
                // GC: delete checkpoint on failure.
                if let Err(e) = state_clone
                    .nats
                    .as_ref()
                    .expect("NATS required")
                    .delete_task_checkpoint(&task_id_for_failure.to_string())
                    .await
                {
                    tracing::debug!("checkpoint GC on failure (may not exist): {e}");
                }
            }
        }

        // Persist estimator state to NATS — fire-and-forget.
        if let Some((ema, count)) = ts.tao_multiplier_estimator.read().await.persist_state() {
            if let Err(e) = state_clone
                .nats
                .as_ref()
                .expect("NATS required")
                .put_tao_estimator_state(&task_tenant_id, ema, count)
                .await
            {
                tracing::warn!("failed to persist tao_estimator: {e}");
            }
        }

        // Persist updated bandit state.
        {
            let bandit = ts.bandit_state.read().await;
            match serde_json::to_vec(&*bandit) {
                Ok(bytes) => {
                    if let Err(e) = state_clone
                        .nats
                        .as_ref()
                        .expect("NATS required")
                        .put_bandit_state(&task_tenant_id, bytes)
                        .await
                    {
                        tracing::warn!("failed to persist bandit state: {e}");
                    }
                }
                Err(e) => tracing::warn!("failed to serialize bandit state: {e}"),
            }
        }
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
        match nats.tail_task_events(&tid, from_seq).await {
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
