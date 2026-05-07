use crate::state::AppState;
use h2ai_types::approval::ApprovalRecord;
use h2ai_types::events::{ApprovalResolvedEvent, H2AIEvent, TaskFailedEvent};
use std::sync::Arc;

pub async fn run_approval_reaper(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;

        let entries = state.nats.list_approval_records_with_revision().await;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        for (record, revision) in entries {
            if now_ms > record.timeout_at_ms {
                match state
                    .nats
                    .delete_approval_record_if_revision(&record.task_id, revision)
                    .await
                {
                    Ok(()) => {
                        tracing::info!(task_id = %record.task_id, "auto-rejecting timed-out approval");
                        auto_reject(&state, &record).await;
                    }
                    Err(_) => {
                        // Another pod already claimed this — skip silently
                    }
                }
            }
        }
    }
}

async fn auto_reject(state: &AppState, record: &ApprovalRecord) {
    let task_id_str = &record.task_id;
    let tid = match uuid::Uuid::parse_str(task_id_str).map(h2ai_types::identity::TaskId::from_uuid)
    {
        Ok(id) => id,
        Err(_) => {
            tracing::warn!(task_id = %task_id_str, "reaper: invalid task_id, skipping");
            return;
        }
    };

    let decided_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let resolved_ev = H2AIEvent::ApprovalResolved(ApprovalResolvedEvent {
        task_id: tid.clone(),
        approved: false,
        operator_id: "system:timeout".into(),
        reviewer_note: Some("Auto-rejected: review timeout exceeded".into()),
        decided_at_ms,
    });
    if let Err(e) = state.nats.publish_event(&tid, &resolved_ev).await {
        tracing::warn!(task_id = %task_id_str, "reaper: failed to publish ApprovalResolved: {e}");
    }

    let failed_ev = H2AIEvent::TaskFailed(TaskFailedEvent {
        task_id: tid.clone(),
        pruned_events: vec![],
        topologies_tried: vec![],
        tau_values_tried: vec![],
        multiplication_condition_failure: None,
        timestamp: chrono::Utc::now(),
    });
    if let Err(e) = state.nats.publish_event(&tid, &failed_ev).await {
        tracing::warn!(task_id = %task_id_str, "reaper: failed to publish TaskFailed: {e}");
    }

    state.store.mark_failed(&tid);

    if let Err(e) = state.nats.delete_task_checkpoint(task_id_str).await {
        tracing::debug!(task_id = %task_id_str, "reaper: checkpoint GC (may not exist): {e}");
    }
}
