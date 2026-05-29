#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Requires NATS — skipped when unavailable.

use futures::StreamExt;
use h2ai_api::oracle_worker::OracleWorker;
use h2ai_config::H2AIConfig;
use h2ai_types::events::{OraclePendingEvent, OracleResultEvent};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::sizing::{OracleDomain, OracleSpec};
use std::time::Duration;

async fn maybe_connect() -> Option<async_nats::Client> {
    let url = H2AIConfig::default().nats_url;
    async_nats::connect(&url).await.ok()
}

fn make_event(task_id: TaskId) -> OraclePendingEvent {
    OraclePendingEvent {
        task_id,
        winning_output: "test output".into(),
        q_confidence: 0.8,
        n_used: 2,
        oracle_spec: OracleSpec {
            runner_uri: String::new(), // empty → OracleClient returns immediately
            timeout_ms: 1000,
            domain: OracleDomain::Factual,
        },
        domain: OracleDomain::Factual,
        oracle_specs: vec![],
        tenant_id: TenantId::default(),
    }
}

// ── OracleWorker::new ─────────────────────────────────────────────────────────

#[tokio::test]
async fn oracle_worker_new_constructs_without_panic() {
    let Some(nats) = maybe_connect().await else {
        return;
    };
    let _worker = OracleWorker::new(nats);
}

// ── run: processes OraclePendingEvent and publishes OracleResultEvent ─────────

#[tokio::test]
async fn oracle_worker_processes_pending_event_and_publishes_result() {
    let Some(nats) = maybe_connect().await else {
        return;
    };

    // Subscribe to results BEFORE starting the worker
    let mut results_sub = nats.subscribe("h2ai.oracle.results").await.unwrap();

    let worker = OracleWorker::new(nats.clone());
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(worker.run_with_ready(ready_tx));
    ready_rx.await.unwrap();

    let task_id = TaskId::new();
    let event = make_event(task_id.clone());
    nats.publish(
        "h2ai.oracle.testspec.pending",
        serde_json::to_vec(&event).unwrap().into(),
    )
    .await
    .unwrap();

    // Filter by task_id since parallel tests may publish to the same subject
    let result = loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), results_sub.next())
            .await
            .expect("timeout waiting for oracle result")
            .expect("subscription closed");
        let r: OracleResultEvent = serde_json::from_slice(&msg.payload).unwrap();
        if r.task_id == task_id {
            break r;
        }
    };
    assert!(!result.passed, "empty runner_uri must produce passed=false");
}

// ── run: publishes to reply subject when present ──────────────────────────────

#[tokio::test]
async fn oracle_worker_publishes_to_reply_subject() {
    let Some(nats) = maybe_connect().await else {
        return;
    };

    let reply_subject = format!("h2ai.oracle.reply.{}", uuid::Uuid::new_v4());
    let mut reply_sub = nats.subscribe(reply_subject.clone()).await.unwrap();

    let worker = OracleWorker::new(nats.clone());
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(worker.run_with_ready(ready_tx));
    ready_rx.await.unwrap();

    let task_id = TaskId::new();
    let event = make_event(task_id.clone());
    nats.publish_with_reply(
        "h2ai.oracle.testreply.pending",
        reply_subject.clone(),
        serde_json::to_vec(&event).unwrap().into(),
    )
    .await
    .unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(5), reply_sub.next())
        .await
        .expect("timeout waiting for reply")
        .expect("subscription closed");

    let result: OracleResultEvent = serde_json::from_slice(&msg.payload).unwrap();
    assert_eq!(result.task_id, task_id);
}

// ── run: malformed payload is skipped, next valid event is processed ──────────

#[tokio::test]
async fn oracle_worker_skips_malformed_payload_and_continues() {
    let Some(nats) = maybe_connect().await else {
        return;
    };

    let mut results_sub = nats.subscribe("h2ai.oracle.results").await.unwrap();

    let worker = OracleWorker::new(nats.clone());
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(worker.run_with_ready(ready_tx));
    ready_rx.await.unwrap();

    // Send garbage — worker must skip
    nats.publish(
        "h2ai.oracle.badpayload.pending",
        b"not valid json!!!".as_ref().into(),
    )
    .await
    .unwrap();

    // Send valid event — must still be processed
    let task_id = TaskId::new();
    let event = make_event(task_id.clone());
    nats.publish(
        "h2ai.oracle.afterbad.pending",
        serde_json::to_vec(&event).unwrap().into(),
    )
    .await
    .unwrap();

    // Filter by task_id since parallel tests may publish to the same subject
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), results_sub.next())
            .await
            .expect("timeout waiting for oracle result after bad payload")
            .expect("subscription closed");
        let result: OracleResultEvent = serde_json::from_slice(&msg.payload).unwrap();
        if result.task_id == task_id {
            break;
        }
    }
}
