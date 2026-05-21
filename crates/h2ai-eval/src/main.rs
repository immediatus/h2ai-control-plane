//! H2AI Oracle Sidecar
//!
//! Subscribes to `h2ai.oracle.pending`, POSTs the winning output to the external
//! oracle service specified in `oracle_spec.runner_uri`, and publishes
//! `OracleResultEvent` to `h2ai.oracle.results`.
//!
//! Environment variables:
//!   `NATS_URL`     — NATS server URL (default: <nats://localhost:4222>)

use futures::StreamExt;
use h2ai_types::events::{OraclePendingEvent, OracleResultEvent};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

const PENDING_SUBJECT: &str = "h2ai.oracle.pending";
const RESULTS_SUBJECT: &str = "h2ai.oracle.results";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("h2ai_eval=info".parse().unwrap()),
        )
        .init();

    let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_owned());

    info!("connecting to NATS at {nats_url}");
    let nc = async_nats::connect(&nats_url)
        .await
        .expect("NATS connect failed");

    let mut sub = nc
        .subscribe(PENDING_SUBJECT)
        .await
        .expect("subscribe failed");

    info!("oracle sidecar ready — listening on {PENDING_SUBJECT}");

    let http_client = reqwest::Client::new();

    while let Some(msg) = sub.next().await {
        let nc = nc.clone();
        let http_client = http_client.clone();
        tokio::spawn(async move {
            let pending: OraclePendingEvent = match serde_json::from_slice(&msg.payload) {
                Ok(p) => p,
                Err(e) => {
                    warn!(error = %e, "failed to parse OraclePendingEvent — dropping");
                    return;
                }
            };

            let task_id = pending.task_id.clone();
            info!(task_id = %task_id, "running oracle for task");

            let result = evaluate(&pending, &http_client).await;

            match serde_json::to_vec(&result) {
                Ok(payload) => {
                    if let Err(e) = nc.publish(RESULTS_SUBJECT, payload.into()).await {
                        warn!(task_id = %task_id, error = %e, "failed to publish OracleResultEvent");
                    } else {
                        info!(
                            task_id = %task_id,
                            passed = result.passed,
                            score = result.score,
                            residual = result.residual,
                            "oracle result published"
                        );
                    }
                }
                Err(e) => {
                    warn!(task_id = %task_id, error = %e, "failed to serialize OracleResultEvent");
                }
            }
        });
    }
}

/// POST the winning output to the external oracle service and return the result.
async fn evaluate(
    pending: &OraclePendingEvent,
    http_client: &reqwest::Client,
) -> OracleResultEvent {
    let spec = &pending.oracle_spec;
    let timeout = std::time::Duration::from_millis(spec.timeout_ms);

    let start_ms = now_ms();

    let body = serde_json::json!({
        "task_id": pending.task_id,
        "output": pending.winning_output,
        "domain": pending.domain,
    });

    let result = http_client
        .post(&spec.runner_uri)
        .timeout(timeout)
        .json(&body)
        .send()
        .await;

    let duration_ms = now_ms() - start_ms;
    let timestamp_ms = now_ms();

    match result {
        Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
            Ok(json) => {
                let passed = json
                    .get("passed")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let score = json
                    .get("score")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(if passed { 1.0 } else { 0.0 });
                let residual = (pending.q_confidence - f64::from(u8::from(passed))).abs();
                let verdict = Some(h2ai_types::sizing::OracleVerdict { details: json });

                OracleResultEvent {
                    task_id: pending.task_id.clone(),
                    q_confidence: pending.q_confidence,
                    n_used: pending.n_used,
                    passed,
                    score,
                    residual,
                    domain: pending.domain.clone(),
                    duration_ms,
                    timestamp_ms,
                    tenant_id: pending.tenant_id.clone(),
                    verdict,
                }
            }
            Err(e) => {
                warn!(task_id = %pending.task_id, error = %e, "oracle response parse failed");
                failure_result(pending, start_ms)
            }
        },
        Ok(resp) => {
            warn!(task_id = %pending.task_id, status = %resp.status(), "oracle HTTP error");
            failure_result(pending, start_ms)
        }
        Err(e) => {
            warn!(task_id = %pending.task_id, error = %e, "oracle HTTP call failed");
            failure_result(pending, start_ms)
        }
    }
}

fn failure_result(pending: &OraclePendingEvent, start_ms: u64) -> OracleResultEvent {
    OracleResultEvent {
        task_id: pending.task_id.clone(),
        q_confidence: pending.q_confidence,
        n_used: pending.n_used,
        passed: false,
        score: 0.0,
        residual: pending.q_confidence, // |q - 0.0|
        domain: pending.domain.clone(),
        duration_ms: now_ms() - start_ms,
        timestamp_ms: now_ms(),
        tenant_id: pending.tenant_id.clone(),
        verdict: None,
    }
}

fn now_ms() -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    ms
}
