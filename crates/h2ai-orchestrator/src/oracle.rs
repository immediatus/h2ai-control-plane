use async_nats::Client as NatsClient;
use h2ai_types::events::OraclePendingEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::OracleSpec;

pub mod oracle_dispatch {
    use super::*;

    /// Fire-and-forget Phase 6 oracle evaluation.
    ///
    /// Publishes [`OraclePendingEvent`] to NATS core subject `h2ai.oracle.pending`.
    /// Does NOT await the oracle result — returns immediately after publish.
    pub async fn fire(
        nats: &NatsClient,
        task_id: TaskId,
        output: &str,
        q_confidence: f64,
        n_used: u32,
        spec: &OracleSpec,
    ) {
        let event = OraclePendingEvent {
            task_id,
            winning_output: output.to_owned(),
            q_confidence,
            n_used,
            oracle_spec: spec.clone(),
            domain: spec.domain.clone(),
        };
        if let Ok(payload) = serde_json::to_vec(&event) {
            let _ = nats.publish("h2ai.oracle.pending", payload.into()).await;
        }
    }
}
