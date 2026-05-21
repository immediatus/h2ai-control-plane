use crate::{error::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    response::{IntoResponse, Json},
};
use h2ai_types::identity::{TaskId, TenantId};
use serde_json::json;

/// `GET /tenants/{tenant_id}/tasks/{task_id}/recover`
///
/// Replays the `JetStream` event log for `task_id` and upserts the reconstructed
/// `TaskState` into the live `TaskStore`. After this call `GET /tasks/{id}/status`
/// returns accurate state even if the server restarted mid-execution.
///
/// Returns 404 if no events exist for the task in `JetStream`.
pub async fn recover_task(
    Path((tenant_id, task_id)): Path<(String, String)>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let tid_uuid = uuid::Uuid::parse_str(&task_id)
        .map_err(|_| ApiError::InvalidRequest(format!("invalid task_id: {task_id}")))?;
    let tid = TaskId::from_uuid(tid_uuid);
    let tenant = TenantId::from(tenant_id.as_str());

    let recovered = state
        .journal
        .replay(&tid)
        .await
        .map_err(|e| ApiError::NatsUnavailable(format!("replay failed: {e}")))?
        .ok_or_else(|| ApiError::TaskNotFound(task_id.clone()))?;

    // Validate ownership — recovered task must belong to the path tenant.
    if recovered.tenant_id != tenant {
        return Err(ApiError::TaskNotFound(task_id.clone()));
    }

    // Only upsert when no live entry exists — avoid overwriting in-progress state.
    if state.store.get_for_tenant(&tid, &tenant).is_none() {
        state.store.insert(tid.clone(), recovered.clone());
    }

    Ok(Json(json!({
        "task_id":             recovered.task_id.to_string(),
        "status":              recovered.status,
        "phase":               recovered.phase,
        "phase_name":          recovered.phase_name,
        "explorers_completed": recovered.explorers_completed,
        "explorers_total":     recovered.explorers_total,
        "proposals_valid":     recovered.proposals_valid,
        "proposals_pruned":    recovered.proposals_pruned,
        "autonomic_retries":   recovered.autonomic_retries,
    })))
}
