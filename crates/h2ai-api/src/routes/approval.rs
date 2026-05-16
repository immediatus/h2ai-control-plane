use crate::{error::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use h2ai_types::approval::ApprovalDecision;
use h2ai_types::events::{ApprovalResolvedEvent, H2AIEvent};
use h2ai_types::identity::{TaskId, TenantId};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ApproveRequest {
    pub approved: bool,
    pub reviewer_note: Option<String>,
    pub operator_id: String,
}

/// `POST /tenants/{tenant_id}/tasks/{task_id}/approve`
///
/// Approve or reject a task awaiting human review.
/// Returns 404 if no pending approval record exists.
/// Returns 410 Gone (via InvalidRequest) if the review window has expired.
/// Returns 409 Conflict (via InvalidRequest) on concurrent approval attempts.
pub async fn approve_task(
    Path((tenant_id, task_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(req): Json<ApproveRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let tenant = TenantId::from(tenant_id.as_str());
    // 1. Load approval record with revision for CAS delete
    let kv_key = format!("{}/{}", tenant.bucket_safe(), task_id);
    let (record, revision) = state
        .nats
        .get_approval_record_with_revision(&tenant, &task_id)
        .await
        .map_err(|e| ApiError::NatsUnavailable(e.to_string()))?
        .ok_or_else(|| ApiError::TaskNotFound(task_id.clone()))?;

    // 2. Check timeout
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    if now_ms > record.timeout_at_ms {
        return Err(ApiError::InvalidRequest(format!(
            "approval window expired for task {task_id}"
        )));
    }

    // 3. Load checkpoint (must exist for approved path)
    let tid = parse_task_id(&task_id)?;
    let checkpoint = state
        .nats
        .get_task_checkpoint(&task_id)
        .await
        .map_err(|e| ApiError::NatsUnavailable(e.to_string()))?
        .ok_or_else(|| {
            ApiError::InvalidRequest(format!("checkpoint missing for task {task_id}"))
        })?;

    // 4. Atomic delete of approval record — only the first caller wins
    state
        .nats
        .delete_approval_record_if_revision(&kv_key, revision)
        .await
        .map_err(|_| {
            ApiError::InvalidRequest(format!(
                "concurrent approval attempt for task {task_id}; try again"
            ))
        })?;

    // 5. Publish ApprovalResolved event
    let decided_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let decision = ApprovalDecision {
        approved: req.approved,
        reviewer_note: req.reviewer_note.clone(),
        operator_id: req.operator_id.clone(),
        decided_at_ms,
    };
    let resolved_ev = H2AIEvent::ApprovalResolved(ApprovalResolvedEvent {
        task_id: tid.clone(),
        approved: req.approved,
        operator_id: req.operator_id.clone(),
        reviewer_note: req.reviewer_note.clone(),
        decided_at_ms,
    });
    if let Err(e) = state.nats.publish_event(&tid, &resolved_ev).await {
        tracing::warn!(task_id = %task_id, "failed to publish ApprovalResolved: {e}");
    }

    // 6. Branch on decision
    if req.approved {
        finalize_approved_task(&state, tid, checkpoint, decision).await?;
        Ok((
            StatusCode::ACCEPTED,
            Json(serde_json::json!({"status": "approved"})),
        ))
    } else {
        finalize_rejected_task(&state, &task_id, tid).await?;
        Ok((
            StatusCode::OK,
            Json(serde_json::json!({"status": "rejected"})),
        ))
    }
}

async fn finalize_approved_task(
    state: &AppState,
    task_id: TaskId,
    checkpoint: h2ai_types::checkpoint::TaskCheckpoint,
    _decision: ApprovalDecision,
) -> Result<(), ApiError> {
    let resolved_output = checkpoint.resolved_output.clone().unwrap_or_default();

    // Publish MergeResolved so SSE clients receive the terminal event
    let merge_ev = H2AIEvent::MergeResolved(h2ai_types::events::MergeResolvedEvent {
        task_id: task_id.clone(),
        resolved_output: resolved_output.clone(),
        j_eff: checkpoint.j_eff,
        timestamp: chrono::Utc::now(),
        oracle_gate_passed: None,
    });
    if let Err(e) = state.nats.publish_event(&task_id, &merge_ev).await {
        tracing::warn!(task_id = %task_id, "failed to publish MergeResolved on approve: {e}");
    }

    state.store.mark_resolved(&task_id);
    if let Err(e) = state
        .nats
        .delete_task_checkpoint(&task_id.to_string())
        .await
    {
        tracing::warn!(task_id = %task_id, "GC checkpoint on approve failed: {e}");
    }
    Ok(())
}

async fn finalize_rejected_task(
    state: &AppState,
    task_id_str: &str,
    task_id: TaskId,
) -> Result<(), ApiError> {
    let failed_ev = H2AIEvent::TaskFailed(h2ai_types::events::TaskFailedEvent {
        task_id: task_id.clone(),
        pruned_events: vec![],
        topologies_tried: vec![],
        tau_values_tried: vec![],
        multiplication_condition_failure: None,
        timestamp: chrono::Utc::now(),
    });
    if let Err(e) = state.nats.publish_event(&task_id, &failed_ev).await {
        tracing::warn!(task_id = %task_id, "failed to publish TaskFailed on reject: {e}");
    }

    state.store.mark_failed(&task_id);
    if let Err(e) = state.nats.delete_task_checkpoint(task_id_str).await {
        tracing::warn!(task_id = %task_id, "GC checkpoint on reject failed: {e}");
    }
    Ok(())
}

/// `GET /tenants/{tenant_id}/tasks/{task_id}/approval`
///
/// Returns the current `ApprovalRecord` if the task is awaiting approval.
/// Returns 404 if no record exists.
pub async fn get_approval(
    Path((tenant_id, task_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let tenant = TenantId::from(tenant_id.as_str());
    let record = state
        .nats
        .get_approval_record_with_revision(&tenant, &task_id)
        .await
        .map_err(|e| ApiError::NatsUnavailable(e.to_string()))?
        .map(|(r, _)| r)
        .ok_or_else(|| ApiError::TaskNotFound(task_id.clone()))?;

    Ok(Json(record))
}

fn parse_task_id(s: &str) -> Result<TaskId, ApiError> {
    uuid::Uuid::parse_str(s)
        .map(TaskId::from_uuid)
        .map_err(|_| ApiError::InvalidRequest(format!("invalid task_id: {s}")))
}
