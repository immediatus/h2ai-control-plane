use crate::shadow_auditor::ShadowAuditorAccumulator;
use crate::tenant_registry::TenantState;
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_knowledge::provider::KnowledgeProvider;
use h2ai_knowledge::skill_provider::{CompositeProvider, SkillProvider};
use h2ai_orchestrator::engine::{EngineError, NatsDispatchConfig, ShadowAuditCtx};
use h2ai_orchestrator::session_journal::SessionJournal;
use h2ai_orchestrator::skill_extractor::skill_from_output;
use h2ai_orchestrator::task_runner::{
    DecompositionArgs, Decomposer, EngineRunner, OwnedEngineInput, ThinkingLoopArgs,
    ThinkingLoopRunner,
};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_state::backend::{NatsBackend, SkillStore};
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::events::CalibrationCompletedEvent;
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::manifest::TaskManifest;
use h2ai_types::sizing::OracleSpec;
use std::sync::Arc;

const THINKING_LOOP_SECTION: &str = "## Thinking Loop Analysis";

pub struct TaskPipelineInput {
    // Identity
    pub task_id: TaskId,
    pub tenant_id: TenantId,
    pub manifest: TaskManifest,
    pub calibration: CalibrationCompletedEvent,
    pub corpus: Vec<ConstraintDoc>,
    pub wiki_revision: u64,
    pub manifest_json: String,
    pub resolved_ids: Vec<String>,

    // Stage runners (mockable via Arc<dyn Trait>)
    pub thinking_loop_runner: Arc<dyn ThinkingLoopRunner>,
    pub decomposer: Arc<dyn Decomposer>,
    pub engine_runner: Arc<dyn EngineRunner>,

    // Infrastructure
    pub nats: Option<Arc<dyn NatsBackend>>,
    pub nats_raw_client: Option<async_nats::Client>,
    pub store: TaskStore,
    pub journal: Arc<SessionJournal>,
    pub cfg: Arc<H2AIConfig>,
    pub metrics: Arc<tokio::sync::RwLock<crate::metrics::MetricsState>>,
    pub drift_monitor: Arc<tokio::sync::Mutex<h2ai_autonomic::drift::DriftMonitor>>,

    // Adapters
    pub adapter_pool: Vec<Arc<dyn IComputeAdapter>>,
    pub verification_adapter: Arc<dyn IComputeAdapter>,
    pub auditor_adapter: Arc<dyn IComputeAdapter>,
    pub embedding_model: Option<Arc<dyn EmbeddingModel>>,
    pub researcher_adapter: Option<Arc<dyn IComputeAdapter>>,
    pub knowledge_provider: Arc<CompositeProvider>,

    // Tenant runtime state
    pub tenant_state: Arc<TenantState>,
    pub nats_dispatch: Option<NatsDispatchConfig>,
    pub srani_ema_cfi: f64,
    pub srani_count: usize,
    pub srani_grounding_chain: Option<Arc<h2ai_orchestrator::srani_grounding::SraniGroundingChain>>,
    pub gap_research_chain: Option<Arc<h2ai_orchestrator::srani_grounding::SraniGroundingChain>>,
    pub shadow_audit_ctx: Option<ShadowAuditCtx>,
    pub shadow_accumulator: Option<Arc<tokio::sync::Mutex<ShadowAuditorAccumulator>>>,
    pub registry: AdapterRegistry,
    pub oracle_spec: Option<OracleSpec>,
    pub debug_log_path: Option<String>,
    pub skill_provider: Arc<SkillProvider>,
}

pub async fn run_task_pipeline(mut input: TaskPipelineInput) {
    use h2ai_orchestrator::decomposition::compute_role_diversity;
    use h2ai_types::config::{AuditorConfig, TaoConfig, VerificationConfig};
    use h2ai_types::events::{H2AIEvent, TaskFailedEvent, ThinkingLoopCompletedEvent};
    use h2ai_types::prompts::ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT;

    let task_id = input.task_id.clone();
    let tenant_id = input.tenant_id.clone();
    let ts = Arc::clone(&input.tenant_state);

    // ── Stage 1: Thinking loop ────────────────────────────────────────────────
    let tao_multiplier = ts.tao_multiplier_estimator.read().await.multiplier();
    let tao_multiplier_estimator = Arc::clone(&ts.tao_multiplier_estimator);
    let bandit = Arc::clone(&ts.bandit_state);
    let srani_ema_cfi = input.srani_ema_cfi;
    let srani_count = input.srani_count;

    let thinking_constraint_tags: Vec<String> = input
        .corpus
        .iter()
        .flat_map(|d| d.domains.iter().cloned())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let thinking_report = input
        .thinking_loop_runner
        .run(ThinkingLoopArgs {
            task_description: input.manifest.description.clone(),
            constraint_ids: input.corpus.iter().map(|c| c.id.clone()).collect(),
            constraint_tags: thinking_constraint_tags,
            knowledge_provider: Some(Arc::clone(&input.knowledge_provider) as Arc<dyn KnowledgeProvider>),
            n_archetypes: input.cfg.thinking_loop.max_archetypes,
            cfg: input.cfg.thinking_loop.clone(),
            adapter: input.adapter_pool[0].clone(),
            embedding_model: input.embedding_model.clone(),
            nats_client: input.nats_raw_client.clone(),
            task_id: task_id.to_string(),
        })
        .await;

    // Publish ThinkingLoopCompletedEvent
    if let Some(ref nats) = input.nats {
        let ev = H2AIEvent::ThinkingLoopCompleted(ThinkingLoopCompletedEvent {
            task_id: task_id.clone(),
            enabled: input.cfg.thinking_loop.enabled,
            iterations_run: thinking_report.iteration,
            coverage_score: thinking_report.coverage_score,
            shared_understanding_len: thinking_report.shared_understanding.len(),
            archetypes: vec![],
            timestamp: chrono::Utc::now(),
        });
        if let Err(e) = nats.publish_event(&task_id, &ev).await {
            tracing::warn!(task_id = %task_id, "failed to publish ThinkingLoopCompletedEvent: {e}");
        }
    }

    let thinking_context = thinking_report.shared_understanding.clone();

    // ── Stage 2: Decomposition ────────────────────────────────────────────────
    let calibration_clone = input.calibration.clone();
    let n_max_usl = (calibration_clone.coefficients.n_max() as usize)
        .min(input.manifest.explorers.count)
        .max(1);
    let n_domains = input
        .corpus
        .iter()
        .flat_map(|d| d.domains.iter())
        .collect::<std::collections::HashSet<_>>()
        .len();
    let n_target = (n_domains + 1).max(2).min(n_max_usl.max(1));

    let slot_configs = match input
        .decomposer
        .decompose(DecompositionArgs {
            description: input.manifest.description.clone(),
            corpus: input.corpus.clone(),
            pareto_weights: input.manifest.pareto_weights.clone(),
            n_target,
            n_max: n_max_usl,
            adapter: input.adapter_pool[0].clone(),
            embedding_model: input.embedding_model.clone(),
            step_max_tokens: input.cfg.decomposition_step_max_tokens,
            json_max_tokens: input.cfg.decomposition_json_max_tokens,
            thinking_context: thinking_context.clone(),
            extra_slots: input.manifest.explorers.slot_configs.clone(),
        })
        .await
    {
        Ok(slots) => {
            let role_diversity = compute_role_diversity(&slots, input.embedding_model.as_deref());
            tracing::info!(
                target: "h2ai.decomposition",
                n_slots = slots.len(),
                n_eff_cosine_roles = role_diversity,
                "decomposition produced slots"
            );
            slots
        }
        Err(e) => {
            tracing::error!(target: "h2ai.decomposition", error = %e, "decomposition failed");
            let failed_ev = H2AIEvent::TaskFailed(TaskFailedEvent {
                task_id: task_id.clone(),
                pruned_events: vec![],
                topologies_tried: vec![],
                tau_values_tried: vec![],
                multiplication_condition_failure: None,
                timestamp: chrono::Utc::now(),
            });
            if let Some(ref nats) = input.nats {
                if let Err(pe) = nats.publish_event(&task_id, &failed_ev).await {
                    tracing::warn!("failed to publish TaskFailedEvent: {pe}");
                }
            }
            input.store.mark_failed(&task_id);
            return;
        }
    };

    // ── Stage 3: Build OwnedEngineInput ──────────────────────────────────────
    let use_adversarial = slot_configs.iter().any(|s| !s.rejection_criteria.is_empty());
    let calibration_source_for_attr = calibration_clone.calibration_source;
    let conformal_margin = input.drift_monitor.lock().await.active_conformal_margin();

    let pool_len = input.adapter_pool.len().max(1);
    let diversity_ids: Vec<u32> = if input.manifest.explorers.diversity_ids.is_empty() {
        (0..input.manifest.explorers.count as u32).collect()
    } else {
        input.manifest.explorers.diversity_ids.clone()
    };
    let explorer_arcs: Vec<Arc<dyn IComputeAdapter>> = diversity_ids
        .iter()
        .map(|id| input.adapter_pool[*id as usize % pool_len].clone())
        .collect();

    let mut manifest_for_engine = input.manifest.clone();
    manifest_for_engine.explorers.slot_configs = slot_configs;
    if !thinking_context.is_empty() {
        manifest_for_engine.context = Some(match manifest_for_engine.context.as_deref() {
            Some(ctx) if !ctx.is_empty() => {
                format!("{ctx}\n\n{THINKING_LOOP_SECTION}\n{thinking_context}")
            }
            _ => format!("{THINKING_LOOP_SECTION}\n{thinking_context}"),
        });
    }

    // Take non-Clone fields out (leaving None in place so input remains valid for post_run borrow).
    let nats_dispatch = input.nats_dispatch.take();
    let shadow_audit_ctx = input.shadow_audit_ctx.take();

    let engine_input = OwnedEngineInput {
        task_id: task_id.clone(),
        manifest: manifest_for_engine,
        calibration: calibration_clone.clone(),
        explorer_adapters: explorer_arcs,
        verification_adapter: input.verification_adapter.clone(),
        auditor_adapter: input.auditor_adapter.clone(),
        auditor_config: AuditorConfig {
            adapter: input.auditor_adapter.kind().clone(),
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: if use_adversarial {
            VerificationConfig {
                threshold: input.cfg.verify_threshold,
                evaluator_system_prompt: ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT.into(),
                record_adversarial_comparison: input.manifest.measure_verifier_ab,
                ..VerificationConfig::default()
            }
        } else {
            VerificationConfig {
                threshold: input.cfg.verify_threshold,
                record_adversarial_comparison: input.manifest.measure_verifier_ab,
                ..VerificationConfig::default()
            }
        },
        constraint_corpus: input.corpus.clone(),
        cfg: Arc::clone(&input.cfg),
        store: input.store.clone(),
        nats_dispatch,
        registry: input.registry.clone(),
        embedding_model: input.embedding_model.clone(),
        tao_multiplier,
        tao_estimator: tao_multiplier_estimator,
        synthesis_adapter: None,
        bandit_state: Some(bandit),
        shadow_audit_ctx,
        researcher_adapter: input.researcher_adapter.clone(),
        srani_ema_cfi,
        srani_count,
        srani_grounding_chain: input.srani_grounding_chain.clone(),
        gap_research_chain: input.gap_research_chain.clone(),
        nats_raw: None,
        tenant_id: tenant_id.clone(),
        nats: input.nats.clone(),
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: Some(Arc::clone(&input.knowledge_provider) as Arc<dyn KnowledgeProvider + Send + Sync>),
        induction_store: None,
        conformal_margin,
    };

    // ── Stage 4: Run engine ───────────────────────────────────────────────────
    let manifest_json = input.manifest_json.clone();
    let resolved_ids = input.resolved_ids.clone();
    let wiki_revision = input.wiki_revision;
    let calibration_for_merge = calibration_clone;

    match input.engine_runner.run(engine_input).await {
        Ok(output) => {
            post_run(
                output,
                &thinking_report,
                &input,
                manifest_json,
                resolved_ids,
                wiki_revision,
                calibration_for_merge,
                calibration_source_for_attr,
                srani_ema_cfi,
                srani_count,
                tenant_id,
                Arc::clone(&ts),
            )
            .await;
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

            if let EngineError::MaxRetriesExhausted {
                partial_verification_events,
                ..
            } = &e
            {
                for event in partial_verification_events {
                    let h2ai_ev =
                        h2ai_types::events::H2AIEvent::VerificationScored(event.clone());
                    if let Some(ref nats) = input.nats {
                        if let Err(pe) = nats.publish_event(&task_id, &h2ai_ev).await {
                            tracing::warn!(
                                "failed to publish partial VerificationScoredEvent: {pe}"
                            );
                        }
                    }
                }
            }

            let failed_ev = h2ai_types::events::H2AIEvent::TaskFailed(
                h2ai_types::events::TaskFailedEvent {
                    task_id: task_id.clone(),
                    pruned_events: vec![],
                    topologies_tried: vec![],
                    tau_values_tried: vec![],
                    multiplication_condition_failure: None,
                    timestamp: chrono::Utc::now(),
                },
            );
            if let Some(ref nats) = input.nats {
                if let Err(pe) = nats.publish_event(&task_id, &failed_ev).await {
                    tracing::warn!("failed to publish TaskFailedEvent: {pe}");
                }
                if let Err(e) = nats.delete_task_checkpoint(&task_id.to_string()).await {
                    tracing::debug!("checkpoint GC on failure: {e}");
                }
            }
            input.store.mark_failed(&task_id);
            input.drift_monitor.lock().await.observe(0.0);
        }
    }

    // Persist estimator state to NATS — fire-and-forget.
    if let Some((ema, count)) = ts.tao_multiplier_estimator.read().await.persist_state() {
        if let Some(ref nats) = input.nats {
            if let Err(e) = nats.put_tao_estimator_state(&input.tenant_id, ema, count).await {
                tracing::warn!("failed to persist tao_estimator: {e}");
            }
        }
    }

    // Persist updated bandit state.
    {
        let bandit = ts.bandit_state.read().await;
        match serde_json::to_vec(&*bandit) {
            Ok(bytes) => {
                if let Some(ref nats) = input.nats {
                    if let Err(e) = nats.put_bandit_state(&input.tenant_id, bytes).await {
                        tracing::warn!("failed to persist bandit state: {e}");
                    }
                }
            }
            Err(e) => tracing::warn!("failed to serialize bandit state: {e}"),
        }
    }
}

pub(crate) async fn post_run(
    output: h2ai_orchestrator::engine::EngineOutput,
    thinking_report: &h2ai_types::thinking::ThinkingReport,
    ctx: &TaskPipelineInput,
    manifest_json: String,
    resolved_ids: Vec<String>,
    wiki_revision: u64,
    calibration_for_merge: CalibrationCompletedEvent,
    calibration_source_for_attr: h2ai_types::events::CalibrationSource,
    srani_ema_cfi: f64,
    srani_count: usize,
    tenant_id: TenantId,
    ts: Arc<TenantState>,
) {
    use crate::routes::tasks::compute_j_eff;
    use h2ai_types::events::{
        CoherenceIncompleteEvent, H2AIEvent, MergeResolvedEvent, TaskAttributionEvent,
    };
    use h2ai_types::sizing::TaskQuadrant;

    let task_id = &output.task_id;
    let nats = match ctx.nats.as_ref() {
        Some(n) => n,
        None => {
            ctx.store.mark_resolved(task_id);
            let skill_nodes = skill_from_output(&output, &ctx.corpus, task_id);
            if !skill_nodes.is_empty() {
                ctx.skill_provider.push_all(skill_nodes);
            }
            if !output.topology_retry_events.is_empty()
                && !thinking_report.retrieved_node_ids.is_empty()
            {
                ctx.knowledge_provider
                    .record_violations(&thinking_report.retrieved_node_ids, 0.1);
            }
            return;
        }
    };

    // Extract skill nodes early — before any partial moves of `output`.
    let skill_nodes_for_persist = skill_from_output(&output, &ctx.corpus, task_id);

    // Checkpoint write (best-effort).
    {
        use h2ai_types::checkpoint::{ConstraintSnapshot, TaskCheckpoint};
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
        let evaluated_ids: Vec<String> = ctx.corpus.iter().map(|d| d.id.clone()).collect();
        let checkpoint = TaskCheckpoint {
            task_id: task_id.to_string(),
            phase: "Merging".into(),
            node_id,
            lease_seq: 0,
            proposals: vec![],
            auditor_survivors: vec![],
            resolved_output: Some(output.resolved_output.clone()),
            manifest_json: manifest_json.clone(),
            object_store_ref: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            constraint_snapshot: Some(ConstraintSnapshot {
                wiki_revision,
                resolved_ids,
                evaluated_ids,
                violation_ids: vec![],
            }),
            j_eff: compute_j_eff(
                output.selection_resolved.valid_proposals.len(),
                ctx.manifest.explorers.count,
                &calibration_for_merge,
            ),
        };
        if let Err(e) = nats.put_task_checkpoint(&checkpoint, None).await {
            tracing::warn!(task_id = %task_id, "checkpoint write failed (best-effort): {e}");
        }
    }

    // Debug log (sync, best-effort).
    if let Some(ref log_path) = ctx.debug_log_path {
        let record = crate::debug_record::TaskDebugRecord::build(
            &ctx.manifest.description,
            srani_ema_cfi,
            srani_count,
            &output,
            &ctx.cfg,
        );
        crate::debug_record::append_debug_record(log_path, &record);
    }

    // Metrics.
    {
        let mut metrics = ctx.metrics.write().await;
        metrics.mapek_mode_collapse_count += output.mode_collapse_count as u64;
        let constrained = output
            .topology_retry_events
            .len()
            .saturating_sub(output.mode_collapse_count);
        metrics.mapek_constrained_exploration_count += constrained as u64;
        match output.complexity_event.task_quadrant {
            TaskQuadrant::Precision => metrics.phase15_quadrant_precision += 1,
            TaskQuadrant::Coverage => metrics.phase15_quadrant_coverage += 1,
            TaskQuadrant::Complex => metrics.phase15_quadrant_complex += 1,
            TaskQuadrant::Degenerate => metrics.phase15_quadrant_degenerate += 1,
        }
        metrics.oracle_tasks_total += 1;
        if ctx.oracle_spec.is_some() {
            metrics.oracle_tasks_with_spec += 1;
        }
        metrics.oracle_coverage_rate = if metrics.oracle_tasks_total > 0 {
            metrics.oracle_tasks_with_spec as f64 / metrics.oracle_tasks_total as f64
        } else {
            0.0
        };
    }

    let complexity_ev = H2AIEvent::TaskComplexityAssessed(output.complexity_event.clone());
    match nats.publish_event_seq(task_id, &complexity_ev).await {
        Ok(seq) => {
            if let Some(task_state) = ctx.store.get(task_id) {
                ctx.journal.note_event(task_id, seq, &task_state);
            }
        }
        Err(e) => tracing::warn!("failed to publish TaskComplexityAssessedEvent: {e}"),
    }

    if let Some(ref frontier_ev) = output.frontier_event {
        let h2ai_ev = H2AIEvent::ConstraintFrontier(frontier_ev.clone());
        match nats.publish_event_seq(task_id, &h2ai_ev).await {
            Ok(seq) => {
                if let Some(task_state) = ctx.store.get(task_id) {
                    ctx.journal.note_event(task_id, seq, &task_state);
                }
            }
            Err(e) => tracing::warn!("failed to publish ConstraintFrontierEvent: {e}"),
        }
    }

    // Online ρ EMA update.
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
                            ctx.cfg.calibration_max_ensemble_size,
                        ));
                    }
                }
            }
        }
    }

    for event in output.verification_events {
        let h2ai_ev = H2AIEvent::VerificationScored(event);
        match nats.publish_event_seq(task_id, &h2ai_ev).await {
            Ok(seq) => {
                if let Some(task_state) = ctx.store.get(task_id) {
                    ctx.journal.note_event(task_id, seq, &task_state);
                }
            }
            Err(e) => tracing::warn!("failed to publish VerificationScoredEvent: {e}"),
        }
    }

    for event in output.failed_proposals {
        let h2ai_ev = H2AIEvent::ProposalFailed(event);
        match nats.publish_event_seq(task_id, &h2ai_ev).await {
            Ok(seq) => {
                if let Some(task_state) = ctx.store.get(task_id) {
                    ctx.journal.note_event(task_id, seq, &task_state);
                }
            }
            Err(e) => tracing::warn!("failed to publish ProposalFailedEvent: {e}"),
        }
    }

    let selection_ev = H2AIEvent::SelectionResolved(output.selection_resolved.clone());
    match nats.publish_event_seq(task_id, &selection_ev).await {
        Ok(seq) => {
            if let Some(task_state) = ctx.store.get(task_id) {
                ctx.journal.note_event(task_id, seq, &task_state);
            }
        }
        Err(e) => tracing::warn!("failed to publish SelectionResolvedEvent: {e}"),
    }

    // τ-spread EMA update.
    if !output.applied_optimizations.is_empty() {
        use h2ai_types::events::OptimizationKind;
        for opt in &output.applied_optimizations {
            if opt.kind == OptimizationKind::TauSpreadAdjusted {
                if let (Ok(before), Ok(after)) =
                    (opt.before.parse::<f64>(), opt.after.parse::<f64>())
                {
                    let mut est = ts.tau_spread_estimator.write().await;
                    est.update(before.min(after), before.max(after));
                }
            }
        }
    }

    let attr_ev = H2AIEvent::TaskAttribution(TaskAttributionEvent {
        task_id: task_id.clone(),
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
        tokens_used: output.tokens_used,
        skill_nodes_injected: thinking_report.skill_nodes_used,
        timestamp: chrono::Utc::now(),
        approval_decision: None,
        calibration_source: calibration_source_for_attr,
    });
    match nats.publish_event_seq(task_id, &attr_ev).await {
        Ok(seq) => {
            if let Some(task_state) = ctx.store.get(task_id) {
                ctx.journal.note_event(task_id, seq, &task_state);
            }
        }
        Err(e) => tracing::warn!("failed to publish TaskAttributionEvent: {e}"),
    }

    for comp_ev in &output.comparison_events {
        let ev = H2AIEvent::VerifierComparison(comp_ev.clone());
        if let Err(e) = nats.publish_event_seq(task_id, &ev).await {
            tracing::warn!("failed to publish VerifierComparisonEvent: {e}");
        }
    }

    if !output.shadow_audit_events.is_empty() {
        for shadow_ev in &output.shadow_audit_events {
            let ev = H2AIEvent::ShadowAudit(shadow_ev.clone());
            if let Err(e) = nats.publish_event_seq(task_id, &ev).await {
                tracing::warn!("failed to publish ShadowAuditorResultEvent: {e}");
            }
        }
        if let Some(ref acc) = ctx.shadow_accumulator {
            acc.lock()
                .await
                .process(output.shadow_audit_events.clone())
                .await;
        }
    }

    for warning in &output.correlated_warnings {
        let ev = H2AIEvent::CorrelatedEnsemble(warning.clone());
        if let Err(e) = nats.publish_event_seq(task_id, &ev).await {
            tracing::warn!("failed to publish CorrelatedEnsembleWarning: {e}");
        }
    }

    for srani_ev in &output.srani_events {
        let ev = H2AIEvent::CorrelatedFabrication(srani_ev.clone());
        if let Err(e) = nats.publish_event_seq(task_id, &ev).await {
            tracing::warn!("failed to publish CorrelatedFabricationEvent: {e}");
        }
    }

    // Persist updated SRANI EMA state.
    if output.srani_count_updated != srani_count {
        if let Err(e) = nats
            .put_srani_state(
                &tenant_id,
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

    for grounding in &output.researcher_grounding_events {
        let ev = H2AIEvent::ResearcherGrounding(grounding.clone());
        if let Err(e) = nats.publish_event_seq(task_id, &ev).await {
            tracing::warn!("failed to publish ResearcherGroundingEvent: {e}");
        }
    }

    if let Some(ref degraded) = output.diversity_degraded_event {
        let ev = H2AIEvent::DiversityGuardDegraded(degraded.clone());
        if let Err(e) = nats.publish_event_seq(task_id, &ev).await {
            tracing::warn!("failed to publish DiversityGuardDegradedEvent: {e}");
        }
    }

    if !output.coherence_state.is_closed() {
        let coh_ev = H2AIEvent::CoherenceIncomplete(CoherenceIncompleteEvent {
            task_id: task_id.clone(),
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
        if let Err(e) = nats.publish_event_seq(task_id, &coh_ev).await {
            tracing::warn!("failed to publish CoherenceIncompleteEvent: {e}");
        }
    }

    for ev in &output.leader_elected_events {
        if let Err(e) = nats
            .publish_event_seq(task_id, &H2AIEvent::LeaderElected(ev.clone()))
            .await
        {
            tracing::warn!(task_id = %task_id, "failed to publish LeaderElectedEvent: {e}");
        }
    }

    for ev in &output.socratic_diagnosis_events {
        if let Err(e) = nats
            .publish_event_seq(task_id, &H2AIEvent::SocraticDiagnosis(ev.clone()))
            .await
        {
            tracing::warn!(task_id = %task_id, "failed to publish SocraticDiagnosisEvent: {e}");
        }
    }

    let j_eff = compute_j_eff(
        output.selection_resolved.valid_proposals.len(),
        ctx.manifest.explorers.count,
        &calibration_for_merge,
    );
    let merge_ev = H2AIEvent::MergeResolved(MergeResolvedEvent {
        task_id: task_id.clone(),
        resolved_output: output.resolved_output.clone(),
        j_eff,
        timestamp: chrono::Utc::now(),
        oracle_gate_passed: None,
        zone3_hints: None,
    });
    if let Err(e) = nats.publish_event(task_id, &merge_ev).await {
        tracing::warn!("failed to publish MergeResolvedEvent: {e}");
    }

    // Background OPRO trigger.
    if let Some(j_eff_value) = j_eff {
        let opro_nats = Arc::clone(nats);
        let opro_cfg = Arc::clone(&ctx.cfg);
        let opro_adapter = Arc::clone(&ctx.adapter_pool[0]);
        let opro_adapter_name = ctx
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

    // Oracle dispatch (fire-and-forget).
    if let Some(ref oracle_spec) = ctx.oracle_spec {
        if let Some(ref nats_raw) = ctx.nats_raw_client {
            let nats_client = nats_raw.clone();
            let task_id_oracle = task_id.clone();
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
    }

    ctx.store.mark_resolved(task_id);

    // Violation feedback: down-weight nodes co-occurring with retried failures (GAP-F5 Step 2).
    if !output.topology_retry_events.is_empty()
        && !thinking_report.retrieved_node_ids.is_empty()
    {
        ctx.knowledge_provider
            .record_violations(&thinking_report.retrieved_node_ids, 0.1);
    }

    // Persist skill nodes extracted earlier (before output was partially moved).
    if !skill_nodes_for_persist.is_empty() {
        match serde_json::to_vec(&skill_nodes_for_persist) {
            Ok(bytes) => {
                if let Err(e) = nats.put_skill_nodes(&tenant_id, bytes).await {
                    tracing::warn!(task_id = %task_id, "failed to persist skill nodes: {e}");
                }
            }
            Err(e) => tracing::warn!(task_id = %task_id, "failed to serialize skill nodes: {e}"),
        }
        ctx.skill_provider.push_all(skill_nodes_for_persist);
    }

    // Feed consensus_agreement_rate to drift monitor.
    if let Some(rate) = output.consensus_agreement_rate {
        let events = ctx.drift_monitor.lock().await.observe(rate);
        for event in events {
            match event {
                h2ai_autonomic::drift::DriftEvent::Warning(w) => {
                    tracing::warn!(
                        target: "h2ai.calibration.drift",
                        metric = %w.metric,
                        recent_mean = w.recent_mean,
                        reference_mean = w.reference_mean,
                        deviation_sigmas = w.deviation_sigmas,
                        "CalibrationDriftWarning"
                    );
                }
                h2ai_autonomic::drift::DriftEvent::Changepoint(cp) => {
                    tracing::warn!(
                        target: "h2ai.calibration.drift",
                        bocpd_mass = cp.bocpd_run_length_posterior_mass,
                        conformal_margin = cp.conformal_margin_applied,
                        "CalibrationChangepoint — ORCA margin active for next tasks"
                    );
                }
            }
        }
    }

    // GC: delete checkpoint.
    if let Err(e) = nats.delete_task_checkpoint(&task_id.to_string()).await {
        tracing::debug!(task_id = %task_id, "checkpoint GC on resolve: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_orchestrator::decomposition::DecompositionError;
    use h2ai_orchestrator::session_journal::SessionJournal;
    use h2ai_orchestrator::task_runner::{Decomposer, EngineRunner, ThinkingLoopRunner};
    use h2ai_orchestrator::task_store::{TaskState, TaskStore};
    use h2ai_test_utils::{
        mock_adapter, stub_engine_output, stub_thinking_report, MockDecomposer, MockEngineRunner,
        MockNatsBackend, MockThinkingLoopRunner,
    };
    use h2ai_types::adapter::AdapterRegistry;
    use h2ai_types::config::ParetoWeights;
    use h2ai_types::events::CalibrationCompletedEvent;
    use h2ai_types::identity::{TaskId, TenantId};
    use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
    use h2ai_types::sizing::CoherencyCoefficients;
    use std::sync::Arc;

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

    fn minimal_manifest() -> TaskManifest {
        TaskManifest {
            description: "pipeline test".into(),
            pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
            topology: TopologyRequest { kind: "ensemble".into(), branching_factor: None },
            explorers: ExplorerRequest {
                count: 2,
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

    fn build_input(
        task_id: TaskId,
        store: TaskStore,
        thinking: Arc<dyn ThinkingLoopRunner>,
        decomposer: Arc<dyn Decomposer>,
        engine: Arc<dyn EngineRunner>,
        skill_provider: Arc<SkillProvider>,
    ) -> TaskPipelineInput {
        use h2ai_config::H2AIConfig;
        use h2ai_knowledge::provider::{KnowledgeProvider, PassthroughProvider};
        use h2ai_knowledge::skill_provider::CompositeProvider;

        let cfg = Arc::new(H2AIConfig::default());
        let adapter = Arc::new(mock_adapter("stub")) as Arc<dyn h2ai_types::adapter::IComputeAdapter>;
        let tenant_id = TenantId::default_tenant();
        let tenant_state = crate::tenant_registry::TenantRegistry::new()
            .get_or_create(&tenant_id, &cfg);

        TaskPipelineInput {
            task_id,
            tenant_id,
            manifest: minimal_manifest(),
            calibration: minimal_calibration(),
            corpus: vec![],
            wiki_revision: 0,
            manifest_json: "{}".into(),
            resolved_ids: vec![],
            thinking_loop_runner: thinking,
            decomposer,
            engine_runner: engine,
            nats: None,
            nats_raw_client: None,
            store,
            journal: Arc::new(SessionJournal::new_noop()),
            cfg: Arc::clone(&cfg),
            metrics: Arc::new(tokio::sync::RwLock::new(crate::metrics::MetricsState::default())),
            drift_monitor: Arc::new(tokio::sync::Mutex::new(
                h2ai_autonomic::drift::DriftMonitor::from_config(&cfg),
            )),
            adapter_pool: vec![adapter.clone()],
            verification_adapter: adapter.clone(),
            auditor_adapter: adapter.clone(),
            embedding_model: None,
            researcher_adapter: None,
            knowledge_provider: CompositeProvider::new(vec![
                Arc::new(PassthroughProvider::new_from_path(std::path::Path::new(".")))
                    as Arc<dyn KnowledgeProvider>,
            ]),
            tenant_state,
            nats_dispatch: None,
            srani_ema_cfi: 0.0,
            srani_count: 0,
            srani_grounding_chain: None,
            gap_research_chain: None,
            shadow_audit_ctx: None,
            shadow_accumulator: None,
            registry: AdapterRegistry::new(adapter),
            oracle_spec: None,
            debug_log_path: None,
            skill_provider,
        }
    }

    #[tokio::test]
    async fn pipeline_marks_failed_when_decomposer_errors() {
        let mut thinking = MockThinkingLoopRunner::new();
        thinking.expect_run().once().returning(|_| stub_thinking_report());

        let mut decomposer = MockDecomposer::new();
        decomposer
            .expect_decompose()
            .once()
            .returning(|_| Err(DecompositionError::EmptyResult));

        let store = TaskStore::new();
        let task_id = TaskId::new();
        store.insert(task_id.clone(), TaskState::new(task_id.clone(), TenantId::default_tenant()));

        let engine = MockEngineRunner::new();

        run_task_pipeline(build_input(
            task_id.clone(),
            store.clone(),
            Arc::new(thinking),
            Arc::new(decomposer),
            Arc::new(engine),
            SkillProvider::new(),
        ))
        .await;

        assert_eq!(store.get(&task_id).unwrap().status, "failed");
    }

    #[tokio::test]
    async fn pipeline_marks_resolved_when_engine_succeeds_and_nats_none() {
        let mut thinking = MockThinkingLoopRunner::new();
        thinking.expect_run().once().returning(|_| stub_thinking_report());

        let mut decomposer = MockDecomposer::new();
        decomposer.expect_decompose().once().returning(|_| Ok(vec![]));

        let store = TaskStore::new();
        let task_id = TaskId::new();
        let task_id_out = task_id.clone();
        store.insert(task_id.clone(), TaskState::new(task_id.clone(), TenantId::default_tenant()));

        let mut engine = MockEngineRunner::new();
        engine.expect_run().once().returning(move |_| Ok(stub_engine_output(task_id_out.clone())));

        run_task_pipeline(build_input(
            task_id.clone(),
            store.clone(),
            Arc::new(thinking),
            Arc::new(decomposer),
            Arc::new(engine),
            SkillProvider::new(),
        ))
        .await;

        assert_eq!(store.get(&task_id).unwrap().status, "resolved");
    }

    #[tokio::test]
    async fn pipeline_post_run_publishes_events_when_nats_present() {
        let mut thinking = MockThinkingLoopRunner::new();
        thinking.expect_run().returning(|_| stub_thinking_report());

        let mut decomposer = MockDecomposer::new();
        decomposer.expect_decompose().returning(|_| Ok(vec![]));

        let store = TaskStore::new();
        let task_id = TaskId::new();
        let task_id_out = task_id.clone();
        store.insert(task_id.clone(), TaskState::new(task_id.clone(), TenantId::default_tenant()));

        let mut engine = MockEngineRunner::new();
        engine.expect_run().returning(move |_| Ok(stub_engine_output(task_id_out.clone())));

        let mut mock_nats = MockNatsBackend::new();
        mock_nats.expect_publish_event().returning(|_, _| Ok(()));
        mock_nats.expect_publish_event_seq().returning(|_, _| Ok(1u64));
        mock_nats.expect_put_task_checkpoint().returning(|_, _| Ok(0u64));
        mock_nats.expect_delete_task_checkpoint().returning(|_| Ok(()));
        mock_nats.expect_put_bandit_state().returning(|_, _| Ok(()));

        let mut input = build_input(
            task_id.clone(),
            store.clone(),
            Arc::new(thinking),
            Arc::new(decomposer),
            Arc::new(engine),
            SkillProvider::new(),
        );
        input.nats = Some(Arc::new(mock_nats));

        run_task_pipeline(input).await;

        assert_eq!(store.get(&task_id).unwrap().status, "resolved");
    }

    #[tokio::test]
    async fn pipeline_extracts_skills_when_engine_returns_output_with_retry() {
        use h2ai_constraints::types::{
            ConstraintPredicate, ConstraintSeverity,
        };
        use h2ai_test_utils::stub_topology_retry_event;
        use h2ai_types::identity::ExplorerId;

        let mut thinking = MockThinkingLoopRunner::new();
        thinking.expect_run().returning(|_| stub_thinking_report());

        let mut decomposer = MockDecomposer::new();
        decomposer.expect_decompose().returning(|_| Ok(vec![]));

        let store = TaskStore::new();
        let task_id = TaskId::new();
        let task_id_out = task_id.clone();
        store.insert(task_id.clone(), TaskState::new(task_id.clone(), TenantId::default_tenant()));

        let mut engine = MockEngineRunner::new();
        engine.expect_run().returning(move |_| {
            let mut out = stub_engine_output(task_id_out.clone());
            out.topology_retry_events =
                vec![stub_topology_retry_event(task_id_out.clone(), 1, Some("violated C-001".into()))];
            out.selection_resolved.valid_proposals = vec![ExplorerId::new()];
            Ok(out)
        });

        let skill_provider = SkillProvider::new();
        let mut input = build_input(
            task_id.clone(),
            store.clone(),
            Arc::new(thinking),
            Arc::new(decomposer),
            Arc::new(engine),
            Arc::clone(&skill_provider),
        );
        input.corpus = vec![h2ai_constraints::types::ConstraintDoc {
            id: "C-001".into(),
            source_file: "C-001.yaml".into(),
            description: "auth constraint".into(),
            severity: ConstraintSeverity::Advisory,
            predicate: ConstraintPredicate::LlmJudge { rubric: "test".into() },
            remediation_hint: None,
            domains: vec!["auth".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        }];

        run_task_pipeline(input).await;

        assert_eq!(store.get(&task_id).unwrap().status, "resolved");
        // tombstone "violated C-001" → 1 Topic node (auth domain) + 1 Constraint-keyed Leaf (C-001)
        assert_eq!((*skill_provider).len(), 2, "one Topic node per domain + one Leaf per constraint ID");
    }

    #[tokio::test]
    async fn pipeline_extracts_skills_and_persists_when_nats_present() {
        use h2ai_constraints::types::{ConstraintPredicate, ConstraintSeverity};
        use h2ai_test_utils::stub_topology_retry_event;
        use h2ai_types::identity::ExplorerId;

        let mut thinking = MockThinkingLoopRunner::new();
        thinking.expect_run().returning(|_| stub_thinking_report());

        let mut decomposer = MockDecomposer::new();
        decomposer.expect_decompose().returning(|_| Ok(vec![]));

        let store = TaskStore::new();
        let task_id = TaskId::new();
        let task_id_out = task_id.clone();
        store.insert(task_id.clone(), TaskState::new(task_id.clone(), TenantId::default_tenant()));

        let mut engine = MockEngineRunner::new();
        engine.expect_run().returning(move |_| {
            let mut out = stub_engine_output(task_id_out.clone());
            out.topology_retry_events =
                vec![stub_topology_retry_event(task_id_out.clone(), 1, Some("violated C-001".into()))];
            out.selection_resolved.valid_proposals = vec![ExplorerId::new()];
            Ok(out)
        });

        let mut mock_nats = MockNatsBackend::new();
        mock_nats.expect_publish_event().returning(|_, _| Ok(()));
        mock_nats.expect_publish_event_seq().returning(|_, _| Ok(1u64));
        mock_nats.expect_put_task_checkpoint().returning(|_, _| Ok(0u64));
        mock_nats.expect_delete_task_checkpoint().returning(|_| Ok(()));
        mock_nats.expect_put_bandit_state().returning(|_, _| Ok(()));
        mock_nats.expect_put_skill_nodes().once().returning(|_, _| Ok(()));

        let skill_provider = SkillProvider::new();
        let mut input = build_input(
            task_id.clone(),
            store.clone(),
            Arc::new(thinking),
            Arc::new(decomposer),
            Arc::new(engine),
            Arc::clone(&skill_provider),
        );
        input.corpus = vec![h2ai_constraints::types::ConstraintDoc {
            id: "C-001".into(),
            source_file: "C-001.yaml".into(),
            description: "auth constraint".into(),
            severity: ConstraintSeverity::Advisory,
            predicate: ConstraintPredicate::LlmJudge { rubric: "test".into() },
            remediation_hint: None,
            domains: vec!["auth".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        }];
        input.nats = Some(Arc::new(mock_nats));

        run_task_pipeline(input).await;

        assert_eq!(store.get(&task_id).unwrap().status, "resolved");
        // tombstone "violated C-001" → 1 Topic node (auth domain) + 1 Constraint-keyed Leaf (C-001)
        assert_eq!((*skill_provider).len(), 2, "one Topic node per domain + one Leaf per constraint ID");
    }

    #[tokio::test]
    async fn skill_injection_scenario_cross_task_learning() {
        use h2ai_constraints::types::{ConstraintPredicate, ConstraintSeverity};
        use h2ai_knowledge::provider::KnowledgeProvider;
        use h2ai_knowledge::skill_provider::CompositeProvider;
        use h2ai_knowledge::types::{KnowledgeQuery, NodeDepth, RetrievalMode, SearchScope};
        use h2ai_test_utils::stub_topology_retry_event;
        use h2ai_types::identity::ExplorerId;

        // Step 1: Build a corpus with two domains: "auth" and "billing".
        let make_doc = |id: &str, domain: &str| h2ai_constraints::types::ConstraintDoc {
            id: id.into(),
            source_file: format!("{domain}.yaml"),
            description: format!("{domain} constraint"),
            severity: ConstraintSeverity::Advisory,
            predicate: ConstraintPredicate::LlmJudge { rubric: "test".into() },
            remediation_hint: None,
            domains: vec![domain.into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        };
        let corpus = vec![make_doc("C-AUTH", "auth"), make_doc("C-BILLING", "billing")];

        // Step 2: Create a SkillProvider.
        let skill_provider = SkillProvider::new();

        // Step 3: Create a CompositeProvider wrapping just the skill provider.
        let composite_provider = CompositeProvider::new(vec![
            Arc::clone(&skill_provider) as Arc<dyn KnowledgeProvider>,
        ]);

        // Step 4: Build an engine output with 1 topology retry (tombstone = failure signal)
        //         and 1 valid proposal so that skill_from_output produces nodes.
        let task_id = TaskId::new();
        let task_id_clone = task_id.clone();
        let mut output = h2ai_test_utils::stub_engine_output(task_id.clone());
        output.topology_retry_events = vec![stub_topology_retry_event(
            task_id_clone,
            1,
            Some("violated auth quota constraint".into()),
        )];
        output.selection_resolved.valid_proposals = vec![ExplorerId::new()];

        // Step 5: Call skill_from_output directly to get nodes.
        let nodes = skill_from_output(&output, &corpus, &task_id);

        // Step 6: Push nodes into the skill provider.
        skill_provider.push_all(nodes);

        // Step 7: Assert 2 nodes — one per domain in corpus.
        assert_eq!(
            (*skill_provider).len(),
            2,
            "expected one skill node per corpus domain (auth + billing)"
        );

        // Step 8: Build a KnowledgeQuery that matches domain "auth".
        let auth_tags: Vec<String> = vec!["auth".into()];
        static DEPTHS: &[NodeDepth] = &[NodeDepth::Topic, NodeDepth::Leaf];
        let query = KnowledgeQuery {
            text: "authentication token service design",
            tags: &auth_tags,
            explicit_ids: &[],
            top_k: 5,
            depths: DEPTHS,
            mode: RetrievalMode::CollapsedTree,
            scope: SearchScope::Auto,
            expand_hops: 0,
        };

        // Step 9: Query the composite provider.
        let result = composite_provider.query(&query).await;

        // Step 10: Assert result is non-empty.
        assert!(
            !result.nodes.is_empty(),
            "composite provider must return at least one node for the auth query"
        );

        // Step 11: Assert that at least one returned node contains the failure signal content
        //          or the domain "auth".
        let has_auth_content = result.nodes.iter().any(|(node, _)| {
            node.synthesis.contains("violated auth quota constraint")
                || node.synthesis.contains("auth")
                || node.domains.contains(&"auth".to_string())
        });
        assert!(
            has_auth_content,
            "at least one returned node must reference the auth failure signal or domain"
        );

        // Step 12: Assert no billing-domain-only node is returned (domain filtering works).
        let has_billing_only = result.nodes.iter().any(|(node, _)| {
            node.domains == vec!["billing".to_string()]
        });
        assert!(
            !has_billing_only,
            "billing-only nodes must not appear in an auth-scoped query"
        );
    }

    #[tokio::test]
    async fn attribution_event_carries_skill_nodes_injected() {
        use h2ai_types::events::H2AIEvent;

        let mut thinking = MockThinkingLoopRunner::new();
        // Return a report that signals 2 skill nodes were injected.
        thinking.expect_run().once().returning(|_| {
            let mut r = stub_thinking_report();
            r.skill_nodes_used = 2;
            r.retrieved_node_ids = vec!["skill-n1".into(), "skill-n2".into()];
            r
        });

        let mut decomposer = MockDecomposer::new();
        decomposer.expect_decompose().once().returning(|_| Ok(vec![]));

        let store = TaskStore::new();
        let task_id = TaskId::new();
        let task_id_out = task_id.clone();
        store.insert(task_id.clone(), TaskState::new(task_id.clone(), TenantId::default_tenant()));

        let mut engine = MockEngineRunner::new();
        engine.expect_run().returning(move |_| Ok(stub_engine_output(task_id_out.clone())));

        // Capture the TaskAttributionEvent's skill_nodes_injected field.
        let captured: Arc<std::sync::Mutex<u32>> = Arc::new(std::sync::Mutex::new(0));
        let cap = Arc::clone(&captured);
        let mut mock_nats = MockNatsBackend::new();
        mock_nats.expect_publish_event().returning(|_, _| Ok(()));
        mock_nats.expect_publish_event_seq().returning(move |_, ev| {
            if let H2AIEvent::TaskAttribution(a) = ev {
                *cap.lock().unwrap() = a.skill_nodes_injected;
            }
            Ok(1u64)
        });
        mock_nats.expect_put_task_checkpoint().returning(|_, _| Ok(0u64));
        mock_nats.expect_delete_task_checkpoint().returning(|_| Ok(()));
        mock_nats.expect_put_bandit_state().returning(|_, _| Ok(()));

        let mut input = build_input(
            task_id.clone(),
            store.clone(),
            Arc::new(thinking),
            Arc::new(decomposer),
            Arc::new(engine),
            SkillProvider::new(),
        );
        input.nats = Some(Arc::new(mock_nats));

        run_task_pipeline(input).await;

        assert_eq!(
            *captured.lock().unwrap(),
            2,
            "TaskAttributionEvent.skill_nodes_injected must equal ThinkingReport.skill_nodes_used"
        );
    }

    #[tokio::test]
    async fn pipeline_extracts_no_skills_on_clean_run() {
        let mut thinking = MockThinkingLoopRunner::new();
        thinking.expect_run().returning(|_| stub_thinking_report());

        let mut decomposer = MockDecomposer::new();
        decomposer.expect_decompose().returning(|_| Ok(vec![]));

        let store = TaskStore::new();
        let task_id = TaskId::new();
        let task_id_out = task_id.clone();
        store.insert(task_id.clone(), TaskState::new(task_id.clone(), TenantId::default_tenant()));

        let mut engine = MockEngineRunner::new();
        engine.expect_run().returning(move |_| Ok(stub_engine_output(task_id_out.clone())));

        let skill_provider = SkillProvider::new();
        run_task_pipeline(build_input(
            task_id.clone(),
            store.clone(),
            Arc::new(thinking),
            Arc::new(decomposer),
            Arc::new(engine),
            Arc::clone(&skill_provider),
        ))
        .await;

        assert_eq!(store.get(&task_id).unwrap().status, "resolved");
        assert_eq!((*skill_provider).len(), 0, "clean run must produce no skills");
    }

    #[tokio::test]
    async fn post_run_records_violations_when_retries_occurred() {
        use h2ai_test_utils::stub_topology_retry_event;
        use h2ai_types::identity::ExplorerId;

        let mut thinking = MockThinkingLoopRunner::new();
        // Report that retrieved "wiki-node-1" (non-synthetic) during the thinking loop.
        thinking.expect_run().once().returning(|_| {
            let mut r = stub_thinking_report();
            r.retrieved_node_ids = vec!["wiki-node-1".into()];
            r
        });

        let mut decomposer = MockDecomposer::new();
        decomposer.expect_decompose().returning(|_| Ok(vec![]));

        let store = TaskStore::new();
        let task_id = TaskId::new();
        let task_id_out = task_id.clone();
        store.insert(task_id.clone(), TaskState::new(task_id.clone(), TenantId::default_tenant()));

        let mut engine = MockEngineRunner::new();
        engine.expect_run().returning(move |_| {
            let mut out = stub_engine_output(task_id_out.clone());
            // Task had retries → should trigger record_violations
            out.topology_retry_events = vec![
                stub_topology_retry_event(task_id_out.clone(), 1, Some("violated C-001".into())),
            ];
            out.selection_resolved.valid_proposals = vec![ExplorerId::new()];
            Ok(out)
        });

        let skill_provider = SkillProvider::new();
        let input = build_input(
            task_id.clone(),
            store.clone(),
            Arc::new(thinking),
            Arc::new(decomposer),
            Arc::new(engine),
            Arc::clone(&skill_provider),
        );
        let composite = Arc::clone(&input.knowledge_provider);

        run_task_pipeline(input).await;

        // The wiki-node-1 should have received a 0.1 violation penalty.
        assert!(
            composite.violation_penalty_for("wiki-node-1") > 0.0,
            "wiki-node-1 must have a non-zero violation penalty after a retried task"
        );
    }

    #[tokio::test]
    async fn post_run_skips_violation_recording_when_no_retries() {
        let mut thinking = MockThinkingLoopRunner::new();
        thinking.expect_run().once().returning(|_| {
            let mut r = stub_thinking_report();
            r.retrieved_node_ids = vec!["wiki-node-2".into()];
            r
        });

        let mut decomposer = MockDecomposer::new();
        decomposer.expect_decompose().returning(|_| Ok(vec![]));

        let store = TaskStore::new();
        let task_id = TaskId::new();
        let task_id_out = task_id.clone();
        store.insert(task_id.clone(), TaskState::new(task_id.clone(), TenantId::default_tenant()));

        let mut engine = MockEngineRunner::new();
        engine.expect_run().returning(move |_| Ok(stub_engine_output(task_id_out.clone())));
        // No topology_retry_events → no violations

        let skill_provider = SkillProvider::new();
        let input = build_input(
            task_id.clone(),
            store.clone(),
            Arc::new(thinking),
            Arc::new(decomposer),
            Arc::new(engine),
            skill_provider,
        );
        let composite = Arc::clone(&input.knowledge_provider);

        run_task_pipeline(input).await;

        assert_eq!(
            composite.violation_penalty_for("wiki-node-2"),
            0.0,
            "no penalty must be applied when the task had no topology retries"
        );
    }
}
