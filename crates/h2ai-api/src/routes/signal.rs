use crate::{error::ApiError, state::AppState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::signal::{ApproveSignal, ResumeSignal, SignalPayload};
use serde::Deserialize;

/// Wire-format DTO for `SignalPayload`, using an adjacently-tagged shape:
/// `{"kind": "Approve", "data": {...}}`.
///
/// This mirrors the custom serde used in `ResumeSignal` but exposes a
/// concrete `Deserialize` impl that Axum's `Json` extractor can use directly.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", content = "data")]
pub enum SignalPayloadDto {
    WaveContinue(h2ai_types::signal::WaveContinueSignal),
    Approve(ApproveSignal),
    #[serde(other)]
    Unknown,
}

impl From<SignalPayloadDto> for SignalPayload {
    fn from(dto: SignalPayloadDto) -> Self {
        match dto {
            SignalPayloadDto::WaveContinue(w) => Self::WaveContinue(w),
            SignalPayloadDto::Approve(a) => Self::Approve(a),
            SignalPayloadDto::Unknown => Self::Unknown,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SignalRequest {
    pub(crate) payload: SignalPayloadDto,
    pub timeout_ms: Option<u64>,
}

/// `POST /tenants/{tenant_id}/tasks/{task_id}/signal`
///
/// Inject an external signal into a running task. Returns 202 immediately after
/// `JetStream` publish — does not wait for engine acknowledgement.
pub async fn submit_signal(
    Path((tenant_id, task_id)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(req): Json<SignalRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Validate Approve requires non-empty operator_id
    match &req.payload {
        SignalPayloadDto::Approve(a) if a.operator_id.trim().is_empty() => {
            return Err(ApiError::InvalidRequest(
                "operator_id is required for Approve signals".into(),
            ));
        }
        _ => {}
    }

    let tid = uuid::Uuid::parse_str(&task_id)
        .map(TaskId::from_uuid)
        .map_err(|_| ApiError::InvalidRequest(format!("invalid task_id: {task_id}")))?;
    let tenant = TenantId::from(tenant_id.as_str());

    // Verify task is known.  If it's already resolved/failed, return 202
    // immediately — the signal arrived after the engine finished (race on timeout
    // or late delivery), which is not an error worth surfacing to the caller.
    match state.store.get(&tid) {
        None => return Err(ApiError::TaskNotFound(task_id.clone())),
        Some(_) if !state.store.is_active(&tid) => {
            return Ok((
                StatusCode::ACCEPTED,
                Json(serde_json::json!({"status": "already_resolved"})),
            ));
        }
        Some(_) => {}
    }

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let timeout_ms = req.timeout_ms.unwrap_or(state.cfg.hitl.timeout_ms).clamp(
        state.cfg.signal_min_timeout_ms,
        state.cfg.signal_max_timeout_ms,
    );

    let signal = ResumeSignal {
        task_id: tid,
        tenant_id: tenant,
        payload: req.payload.into(),
        timeout_at_ms: now_ms + timeout_ms,
        issued_at_ms: now_ms,
    };

    if let Some(nats) = &state.nats {
        nats.publish_signal(&signal)
            .await
            .map_err(|e| ApiError::NatsUnavailable(e.to_string()))?;
    }

    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({"status": "signal_queued"})),
    ))
}
