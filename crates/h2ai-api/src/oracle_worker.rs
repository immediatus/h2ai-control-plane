use crate::oracle::client::OracleClient;
use futures::StreamExt;
use h2ai_types::events::{OraclePendingEvent, OracleResultEvent};
use h2ai_types::sizing::OracleFamily;

/// FUSE worst-of-family reduction for multi-oracle verdicts.
///
/// Groups verdicts by `OracleFamily`, takes the minimum score within each family
/// (correlated failure modes count as one vote), then averages across families.
/// An empty input returns `(false, 0.0)`.
/// The result passes when the aggregated score ≥ 0.5.
#[must_use]
pub fn fuse_reduce_by_family(verdicts: &[(OracleFamily, bool, f64)]) -> (bool, f64) {
    if verdicts.is_empty() {
        return (false, 0.0);
    }
    // Use discriminant as HashMap key: Syntactic=0, Semantic=1, Human=2
    let mut family_min: std::collections::HashMap<u8, f64> = std::collections::HashMap::new();
    for (family, _, score) in verdicts {
        let key = match family {
            OracleFamily::Syntactic => 0,
            OracleFamily::Semantic => 1,
            OracleFamily::Human => 2,
        };
        let entry = family_min.entry(key).or_insert(1.0_f64);
        *entry = entry.min(*score);
    }
    let sum: f64 = family_min.values().sum();
    let final_score = sum / family_min.len() as f64;
    (final_score >= 0.5, final_score)
}

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
        self.run_with_ready(None).await;
    }

    pub async fn run_with_ready(self, ready: impl Into<Option<tokio::sync::oneshot::Sender<()>>>) {
        let ready = ready.into();
        let mut sub = match self.nats_raw.subscribe("h2ai.oracle.*.pending").await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "OracleWorker: failed to subscribe");
                return;
            }
        };
        tracing::info!("OracleWorker: subscribed to h2ai.oracle.*.pending");
        if let Some(tx) = ready {
            let _ = tx.send(());
        }

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

            let primary_resp = self
                .oracle_client
                .evaluate(&ev.oracle_spec, &ev.task_id, &ev.winning_output)
                .await;

            let (passed, score) = if ev.oracle_specs.is_empty() {
                // Single-oracle path (backward-compatible)
                (primary_resp.passed, primary_resp.score)
            } else {
                // Multi-oracle FUSE path: run primary + additional specs, reduce by family
                let mut verdicts: Vec<(OracleFamily, bool, f64)> = vec![(
                    ev.oracle_spec.domain.family(),
                    primary_resp.passed,
                    primary_resp.score,
                )];
                for spec in &ev.oracle_specs {
                    let r = self
                        .oracle_client
                        .evaluate(spec, &ev.task_id, &ev.winning_output)
                        .await;
                    verdicts.push((spec.domain.family(), r.passed, r.score));
                }
                fuse_reduce_by_family(&verdicts)
            };

            let passed_f64 = if passed { 1.0_f64 } else { 0.0_f64 };
            let result = OracleResultEvent {
                task_id: ev.task_id.clone(),
                q_confidence: ev.q_confidence,
                n_used: ev.n_used,
                passed,
                score,
                residual: (ev.q_confidence - passed_f64).abs(),
                domain: ev.oracle_spec.domain.clone(),
                duration_ms: start.elapsed().as_millis() as u64,
                timestamp_ms,
                tenant_id: ev.tenant_id.clone(),
                verdict: Some(h2ai_types::sizing::OracleVerdict {
                    details: primary_resp.details,
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
                        passed = result.passed,
                        score = result.score,
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
