use crate::state::AppState;
use h2ai_orchestrator::engine::{EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::{TaskPhase, TaskState};
use h2ai_types::checkpoint::TaskCheckpoint;
use h2ai_types::config::{AuditorConfig, TaoConfig, VerificationConfig};
use h2ai_types::events::{H2AIEvent, MergeResolvedEvent};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::TaskManifest;
use std::sync::Arc;
use uuid::Uuid;

/// Returns a stable node identifier as `"hostname:PID"`.
///
/// Used to distinguish own-node checkpoints (resume immediately) from
/// foreign-node checkpoints (attempt optimistic CAS claim with jitter).
pub fn local_node_id() -> String {
    let host = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".into());
    format!("{host}:{}", std::process::id())
}

/// Scan the NATS checkpoint KV bucket and resume all in-flight tasks.
///
/// Intended to be called once at startup, after `AppState` is fully
/// constructed but **before** the Axum listener is bound, so no new
/// tasks can race against recovery.
///
/// Strategy:
/// - Own-node tasks: resume immediately (node restarted mid-task).
/// - Foreign-node tasks: sleep a random jitter (0–1500 ms), then attempt
///   an optimistic compare-and-swap claim via `put_task_checkpoint`.
///   If the CAS fails another node won the race; skip silently.
pub async fn recover_in_flight_tasks(state: Arc<AppState>) {
    let my_node_id = local_node_id();
    let entries = state.nats.list_task_checkpoints().await;
    tracing::info!("recovery: found {} checkpoints to inspect", entries.len());

    for checkpoint in entries {
        if checkpoint.node_id != my_node_id {
            // Foreign-node task: apply jitter before racing for ownership.
            let jitter_ms = rand::random::<u64>() % 1500;
            tokio::time::sleep(std::time::Duration::from_millis(jitter_ms)).await;

            let mut claimed = checkpoint.clone();
            claimed.node_id = my_node_id.clone();

            match state
                .nats
                .put_task_checkpoint(&claimed, Some(checkpoint.lease_seq))
                .await
            {
                Ok(new_seq) => {
                    tracing::info!(
                        task_id = %checkpoint.task_id,
                        "recovery: claimed orphaned task"
                    );
                    let mut to_resume = claimed;
                    to_resume.lease_seq = new_seq;
                    spawn_resume(state.clone(), to_resume);
                }
                Err(_) => {
                    tracing::debug!(
                        task_id = %checkpoint.task_id,
                        "recovery: lost claim race, skipping"
                    );
                }
            }
        } else {
            tracing::info!(
                task_id = %checkpoint.task_id,
                phase  = %checkpoint.phase,
                "recovery: resuming own task"
            );
            spawn_resume(state.clone(), checkpoint);
        }
    }
}

/// Deserialize the checkpoint manifest and re-run the task from where it left off.
///
/// Runs inside a detached `tokio::spawn` so recovery does not block the startup path.
/// On success: publishes `MergeResolved` and GCs the checkpoint.
/// On failure: marks the task failed, publishes nothing further, and GCs the checkpoint.
fn spawn_resume(state: Arc<AppState>, checkpoint: TaskCheckpoint) {
    tokio::spawn(async move {
        // --- Deserialize manifest ---
        let manifest: TaskManifest = match serde_json::from_str(&checkpoint.manifest_json) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(
                    task_id = %checkpoint.task_id,
                    "recovery: corrupt manifest: {e}"
                );
                state
                    .nats
                    .delete_task_checkpoint(&checkpoint.task_id)
                    .await
                    .ok();
                return;
            }
        };

        // --- Parse task_id ---
        let task_id: TaskId = match Uuid::parse_str(&checkpoint.task_id) {
            Ok(u) => TaskId::from_uuid(u),
            Err(_) => {
                tracing::error!(
                    task_id = %checkpoint.task_id,
                    "recovery: invalid task_id format"
                );
                return;
            }
        };

        // --- Re-register in store so status queries work immediately ---
        state.store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), manifest.tenant_id.clone()),
        );
        if let Some(phase) = TaskPhase::try_from_name_str(&checkpoint.phase) {
            state.store.set_phase(&task_id, phase, 0, 0);
        }

        // --- Require calibration ---
        let ts = state.tenant_state(&manifest.tenant_id);
        let calibration = match ts.calibration.read().await.clone() {
            Some(c) => c,
            None => {
                tracing::warn!(
                    task_id = %task_id,
                    "recovery: no calibration available, skipping task"
                );
                return;
            }
        };

        // --- Load constraint corpus ---
        let resolver = state.constraint_resolver();
        let task_tags = manifest.constraint_tags.clone();
        let explicit_ids = manifest.constraints.clone();
        let corpus = resolver
            .resolve(&explicit_ids, &task_tags, &manifest.description)
            .await;

        // --- Snapshot tao_multiplier before building input ---
        let tao_multiplier = ts.tao_multiplier_estimator.read().await.multiplier();
        let tao_estimator = Arc::clone(&ts.tao_multiplier_estimator);
        let bandit = Arc::clone(&ts.bandit_state);

        // --- Build owned adapter arcs so we can take short-lived references into EngineInput ---
        let pool_arcs: Vec<std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>> =
            state.adapter_pool.clone();
        let pool_len = pool_arcs.len().max(1);
        let count = manifest.explorers.count;
        let diversity_ids: Vec<u32> = if manifest.explorers.diversity_ids.is_empty() {
            (0..count as u32).collect()
        } else {
            manifest.explorers.diversity_ids.clone()
        };
        let explorer_arcs: Vec<std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>> =
            diversity_ids
                .iter()
                .map(|id| pool_arcs[*id as usize % pool_len].clone())
                .collect();
        let verifier = state.verification_adapter.clone();
        let auditor = state.auditor_adapter.clone();
        let registry = state.registry();
        let cfg = state.cfg.clone();
        let tenant_id = manifest.tenant_id.clone();

        let input = EngineInput {
            task_id: task_id.clone(),
            manifest,
            calibration,
            explorer_adapters: explorer_arcs.iter().map(|a| a.as_ref()).collect(),
            verification_adapter: verifier.as_ref(),
            auditor_adapter: auditor.as_ref(),
            auditor_config: AuditorConfig {
                adapter: auditor.kind().clone(),
                ..Default::default()
            },
            tao_config: TaoConfig::default(),
            verification_config: VerificationConfig::default(),
            constraint_corpus: corpus,
            cfg: &cfg,
            store: state.store.clone(),
            nats_dispatch: None,
            registry: &registry,
            embedding_model: state.embedding_model.as_deref(),
            tao_multiplier,
            tao_estimator,
            synthesis_adapter: None,
            bandit_state: Some(bandit),
            shadow_audit_ctx: None,
            researcher_adapter: None,
            srani_ema_cfi: 0.45,
            srani_count: 0,
            srani_grounding_chain: None,
            nats_raw: None,
            tenant_id,
            nats: Some(state.nats.clone()),
            prev_assembled_contexts: Vec::new(),
            compression_adapter: None,
            stable_cache: None,
            knowledge_provider: Some(state.knowledge_provider.clone()),
            induction_store: None,
        };

        match ExecutionEngine::run_from_checkpoint(input, checkpoint.clone()).await {
            Ok(output) => {
                let ev = H2AIEvent::MergeResolved(MergeResolvedEvent {
                    task_id: output.task_id.clone(),
                    resolved_output: output.resolved_output.clone(),
                    j_eff: None,
                    timestamp: chrono::Utc::now(),
                    oracle_gate_passed: None,
                });
                if let Err(e) = state.nats.publish_event(&output.task_id, &ev).await {
                    tracing::warn!(
                        task_id = %output.task_id,
                        "recovery: publish MergeResolved failed: {e}"
                    );
                }
                state.store.mark_resolved(&output.task_id);
                if let Err(e) = state
                    .nats
                    .delete_task_checkpoint(&output.task_id.to_string())
                    .await
                {
                    tracing::debug!(
                        task_id = %output.task_id,
                        "recovery: checkpoint GC (may already be gone): {e}"
                    );
                }
            }
            Err(e) => {
                tracing::error!(task_id = %task_id, "recovery: run_from_checkpoint failed: {e}");
                state.store.mark_failed(&task_id);
                if let Err(gc_err) = state
                    .nats
                    .delete_task_checkpoint(&task_id.to_string())
                    .await
                {
                    tracing::debug!(
                        task_id = %task_id,
                        "recovery: checkpoint GC on failure: {gc_err}"
                    );
                }
            }
        }
    });
}
