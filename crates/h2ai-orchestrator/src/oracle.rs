use h2ai_types::events::OraclePendingEvent;
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::sizing::OracleSpec;

/// Abstracts the NATS publish operation so `oracle_dispatch::fire` can be tested
/// without a live NATS server.
#[async_trait::async_trait]
pub trait OraclePublisher: Send + Sync {
    async fn publish_oracle(&self, subject: String, payload: bytes::Bytes);
}

#[async_trait::async_trait]
impl OraclePublisher for async_nats::Client {
    async fn publish_oracle(&self, subject: String, payload: bytes::Bytes) {
        let _ = self.publish(subject, payload).await;
    }
}

pub mod oracle_dispatch {
    use super::{OraclePendingEvent, OraclePublisher, OracleSpec, TaskId, TenantId};

    /// Fire-and-forget Phase 6 oracle evaluation.
    ///
    /// Publishes [`OraclePendingEvent`] to NATS per-tenant subject
    /// `h2ai.oracle.<tenant_id>.pending`.
    /// Does NOT await the oracle result — returns immediately after publish.
    pub async fn fire(
        nats: &impl OraclePublisher,
        task_id: TaskId,
        tenant_id: TenantId,
        output: &str,
        q_confidence: f64,
        n_used: u32,
        spec: &OracleSpec,
    ) {
        let event = OraclePendingEvent {
            task_id,
            tenant_id: tenant_id.clone(),
            winning_output: output.to_owned(),
            q_confidence,
            n_used,
            oracle_spec: spec.clone(),
            domain: spec.domain.clone(),
        };
        if let Ok(payload) = serde_json::to_vec(&event) {
            let subject = format!("h2ai.oracle.{}.pending", tenant_id.as_ref());
            nats.publish_oracle(subject, payload.into()).await;
        }
    }
}
