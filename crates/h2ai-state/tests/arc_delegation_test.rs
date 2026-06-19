#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::redundant_clone
)]
//! Tests that exercise the `Arc<T>` delegation impls in `backend.rs`.
//!
//! Each test wraps `InMemoryStateBackend` in an `Arc` and calls the relevant
//! trait method through the `Arc` wrapper so that the forwarding impls
//! (lines 196–431 of `backend.rs`) are covered.

use std::sync::Arc;

use h2ai_state::backend::{
    ConflictStore, ShadowDomainStore, SignalSubscriber, SkillStore, TaskCheckpointStore,
};
use h2ai_state::in_memory::InMemoryStateBackend;
use h2ai_types::checkpoint::TaskCheckpoint;
use h2ai_types::conflict::ConflictRateAccumulator;
use h2ai_types::identity::{TaskId, TenantId};

// ── SkillStore Arc<T> delegation (lines 197–208) ─────────────────────────────

#[tokio::test]
async fn arc_skill_store_put_and_get() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tenant = TenantId::from("arc-tenant-skill");
    let data = b"[\"skill-node\"]".to_vec();

    backend
        .put_skill_nodes(&tenant, data.clone())
        .await
        .unwrap();

    let loaded = backend.get_skill_nodes(&tenant).await.unwrap();
    assert_eq!(loaded, data);
}

#[tokio::test]
async fn arc_skill_store_get_missing_returns_empty() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tenant = TenantId::from("arc-tenant-skill-missing");

    let loaded = backend.get_skill_nodes(&tenant).await.unwrap();
    assert!(loaded.is_empty());
}

// ── ConflictStore Arc<T> delegation (lines 299–325) ──────────────────────────

#[tokio::test]
async fn arc_conflict_store_ensure_bucket_is_noop() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tenant = TenantId::from("arc-tenant-conflict");

    // ensure_conflict_bucket is a no-op on the in-memory backend.
    backend
        .ensure_conflict_bucket(&tenant, "h2ai-conflict")
        .await
        .unwrap();
}

#[tokio::test]
async fn arc_conflict_store_put_and_get_accumulator() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tenant = TenantId::from("arc-tenant-conflict-acc");
    let prefix = "h2ai-conflict";

    // Before any write, returns None.
    let initial = backend
        .get_conflict_accumulator(&tenant, prefix)
        .await
        .unwrap();
    assert!(initial.is_none());

    let acc = ConflictRateAccumulator::new(tenant.clone(), 0.5);
    backend
        .put_conflict_accumulator(&acc, prefix)
        .await
        .unwrap();

    let loaded = backend
        .get_conflict_accumulator(&tenant, prefix)
        .await
        .unwrap();
    assert!(loaded.is_some());
}

// ── SignalSubscriber Arc<T> delegation (lines 347–359) ───────────────────────

#[tokio::test]
async fn arc_signal_subscriber_subscribe_returns_empty_stream() {
    use futures::StreamExt;

    let backend = Arc::new(InMemoryStateBackend::new());
    let task_id = TaskId::new();
    let tenant_id = TenantId::from("arc-tenant-signal");

    let mut stream = backend
        .subscribe_signals(&task_id, &tenant_id)
        .await
        .unwrap();

    // The in-memory stream is empty; it should resolve immediately to None.
    let item = tokio::time::timeout(std::time::Duration::from_millis(10), stream.next()).await;
    assert!(
        matches!(item, Ok(None) | Err(_)),
        "arc in-memory signal stream must yield no items"
    );
}

#[tokio::test]
async fn arc_signal_subscriber_delete_consumer_is_noop() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let task_id = TaskId::new();

    backend.delete_signal_consumer(&task_id).await.unwrap();
}

// ── ShadowDomainStore Arc<T> delegation (lines 373–384) ──────────────────────

#[tokio::test]
async fn arc_shadow_domain_store_put_and_get() {
    use std::collections::HashSet;

    let backend = Arc::new(InMemoryStateBackend::new());

    let initial = backend.get_shadow_promoted_domains().await.unwrap();
    assert!(initial.is_empty());

    let domains: HashSet<String> = ["auth".into(), "billing".into()].into();
    backend.put_shadow_promoted_domains(&domains).await.unwrap();

    let loaded = backend.get_shadow_promoted_domains().await.unwrap();
    assert_eq!(loaded, domains);
}

// ── TaskCheckpointStore Arc<T> delegation (lines 411–432) ────────────────────

fn make_checkpoint(task_id: &str) -> TaskCheckpoint {
    TaskCheckpoint {
        task_id: task_id.to_owned(),
        phase: "ParallelGeneration".into(),
        node_id: "test-node".into(),
        lease_seq: 0,
        proposals: vec![],
        auditor_survivors: vec![],
        resolved_output: None,
        manifest_json: "{}".into(),
        object_store_ref: None,
        created_at_ms: 0,
        updated_at_ms: 0,
        constraint_snapshot: None,
        j_eff: None,
    }
}

#[tokio::test]
async fn arc_task_checkpoint_store_list_empty() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let list = backend.list_task_checkpoints().await;
    assert!(list.is_empty());
}

#[tokio::test]
async fn arc_task_checkpoint_store_put_and_get() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let cp = make_checkpoint("arc-task-1");

    let rev = backend.put_task_checkpoint(&cp, None).await.unwrap();
    assert_eq!(rev, 1);

    let loaded = backend.get_task_checkpoint("arc-task-1").await.unwrap();
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap().task_id, "arc-task-1");
}

#[tokio::test]
async fn arc_task_checkpoint_store_get_missing_returns_none() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let result = backend.get_task_checkpoint("no-such-task").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn arc_task_checkpoint_store_delete() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let cp = make_checkpoint("arc-task-del");

    backend.put_task_checkpoint(&cp, None).await.unwrap();
    assert!(backend
        .get_task_checkpoint("arc-task-del")
        .await
        .unwrap()
        .is_some());

    backend
        .delete_task_checkpoint("arc-task-del")
        .await
        .unwrap();
    assert!(backend
        .get_task_checkpoint("arc-task-del")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn arc_task_checkpoint_store_list_after_put() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let cp = make_checkpoint("arc-task-list");

    backend.put_task_checkpoint(&cp, None).await.unwrap();

    let list = backend.list_task_checkpoints().await;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].task_id, "arc-task-list");
}
