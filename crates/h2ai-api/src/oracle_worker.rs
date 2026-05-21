use crate::oracle::client::OracleClient;
use futures::StreamExt;
use h2ai_types::events::{OraclePendingEvent, OracleResultEvent};

pub struct OracleWorker {
    pub nats_raw: async_nats::Client,
    pub oracle_client: OracleClient,
}

impl OracleWorker {
    #[must_use]
    pub fn new(nats_raw: async_nats::Client) -> Self {
        Self {
            nats_raw,
            oracle_client: OracleClient::new(),
        }
    }

    pub async fn run(self) {
        let mut sub = match self.nats_raw.subscribe("h2ai.oracle.*.pending").await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "OracleWorker: failed to subscribe");
                return;
            }
        };
        tracing::info!("OracleWorker: subscribed to h2ai.oracle.*.pending");

        while let Some(msg) = sub.next().await {
            let reply_subject = msg.reply.clone();
            let Ok(ev) = serde_json::from_slice::<OraclePendingEvent>(&msg.payload) else {
                tracing::warn!("OracleWorker: failed to parse OraclePendingEvent");
                continue;
            };

            let start = std::time::Instant::now();
            let timestamp_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let resp = self
                .oracle_client
                .evaluate(&ev.oracle_spec, &ev.task_id, &ev.winning_output)
                .await;

            let passed_f64 = if resp.passed { 1.0_f64 } else { 0.0_f64 };
            let result = OracleResultEvent {
                task_id: ev.task_id.clone(),
                q_confidence: ev.q_confidence,
                n_used: ev.n_used,
                passed: resp.passed,
                score: resp.score,
                residual: (ev.q_confidence - passed_f64).abs(),
                domain: ev.oracle_spec.domain.clone(),
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp_ms,
                tenant_id: ev.tenant_id.clone(),
                verdict: Some(h2ai_types::sizing::OracleVerdict {
                    details: resp.details,
                }),
            };

            match serde_json::to_vec(&result) {
                Ok(payload) => {
                    let _ = self
                        .nats_raw
                        .publish("h2ai.oracle.results", payload.clone().into())
                        .await;
                    if let Some(reply) = reply_subject {
                        let _ = self.nats_raw.publish(reply, payload.into()).await;
                    }
                    tracing::debug!(
                        task_id = %ev.task_id,
                        passed = resp.passed,
                        score = resp.score,
                        "oracle result published"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        task_id = %ev.task_id,
                        "OracleWorker: failed to serialize result"
                    );
                }
            }
        }
    }
}
