//! Unit tests for [`ReasoningStore`] on [`InMemoryStateBackend`].
//!
//! No NATS server required — all storage is in-memory.

use h2ai_state::backend::ReasoningStore;
use h2ai_state::InMemoryStateBackend;
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::reasoning_checkpoint::{
    ReasoningCheckpointPhase, TaskMetaState, TaskReasoningCheckpoint,
};

const CP_PREFIX: &str = "H2AI_CHECKPOINT";
const MS_PREFIX: &str = "H2AI_META";

fn make_checkpoint(task_id: TaskId, tenant_id: TenantId) -> TaskReasoningCheckpoint {
    TaskReasoningCheckpoint {
        task_id,
        tenant_id,
        created_at: 1_700_000_000,
        last_updated: 1_700_000_000,
        phase: ReasoningCheckpointPhase::Created,
        constraint_tags: vec!["accuracy".to_string()],
        domain: Some("code".to_string()),
        task_quadrant: None,
        system_context_with_rubric_hash: 0,
        constraint_corpus_fingerprint: 0,
        shared_understanding: None,
        tensions: None,
        archetype_selection: None,
        thinking_iterations: None,
        completed_waves: vec![],
        retry_count: 0,
        retry_context_that_resolved: None,
        tried_topologies: vec![],
        tau_values_that_converged: None,
        resolved_attribution_json: None,
        resolved_waste_ratio: None,
        hitl_timeouts_fired: 0,
    }
}

fn make_meta_state(task_id: TaskId, tenant_id: TenantId) -> TaskMetaState {
    TaskMetaState {
        task_id,
        tenant_id,
        resolved_at: 1_700_000_000,
        constraint_tags: vec!["accuracy".to_string()],
        domain: Some("factual".to_string()),
        task_quadrant: None,
        shared_understanding: "Test understanding.".to_string(),
        tensions: vec!["tension A".to_string()],
        archetype_results: vec![],
        thinking_iterations: 2,
        retry_count: 0,
        retry_context_that_resolved: None,
        tried_topologies: vec![],
        tau_values_that_converged: None,
        system_context_with_rubric_hash: 12345,
        constraint_corpus_fingerprint: 67890,
    }
}

// ── ensure_reasoning_buckets ──────────────────────────────────────────────────

#[tokio::test]
async fn ensure_reasoning_buckets_is_noop() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    backend
        .ensure_reasoning_buckets(&tenant, CP_PREFIX, MS_PREFIX)
        .await
        .expect("should succeed");
    // calling twice is idempotent
    backend
        .ensure_reasoning_buckets(&tenant, CP_PREFIX, MS_PREFIX)
        .await
        .expect("should succeed on second call");
}

// ── reasoning checkpoint ──────────────────────────────────────────────────────

#[tokio::test]
async fn checkpoint_put_get_roundtrip() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    let task_id = TaskId::new();

    let cp = make_checkpoint(task_id.clone(), tenant.clone());
    backend
        .put_reasoning_checkpoint(&cp, CP_PREFIX)
        .await
        .expect("put_reasoning_checkpoint");

    let loaded = backend
        .get_reasoning_checkpoint(&task_id, &tenant, CP_PREFIX)
        .await
        .expect("get_reasoning_checkpoint")
        .expect("should be Some");

    assert_eq!(loaded.task_id, task_id);
    assert_eq!(loaded.phase, ReasoningCheckpointPhase::Created);
    assert_eq!(loaded.domain, Some("code".to_string()));
}

#[tokio::test]
async fn checkpoint_get_returns_none_when_absent() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    let result = backend
        .get_reasoning_checkpoint(&TaskId::new(), &tenant, CP_PREFIX)
        .await
        .expect("get_reasoning_checkpoint");
    assert!(result.is_none());
}

#[tokio::test]
async fn checkpoint_overwrite_replaces_value() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    let task_id = TaskId::new();

    let mut cp = make_checkpoint(task_id.clone(), tenant.clone());
    backend
        .put_reasoning_checkpoint(&cp, CP_PREFIX)
        .await
        .expect("first put");

    cp.phase = ReasoningCheckpointPhase::ThinkingDone;
    backend
        .put_reasoning_checkpoint(&cp, CP_PREFIX)
        .await
        .expect("second put");

    let loaded = backend
        .get_reasoning_checkpoint(&task_id, &tenant, CP_PREFIX)
        .await
        .expect("get")
        .expect("Some");
    assert_eq!(loaded.phase, ReasoningCheckpointPhase::ThinkingDone);
}

// ── task meta state ───────────────────────────────────────────────────────────

#[tokio::test]
async fn meta_state_put_get_roundtrip() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    let task_id = TaskId::new();

    let meta = make_meta_state(task_id.clone(), tenant.clone());
    backend
        .put_task_meta_state(&meta, MS_PREFIX)
        .await
        .expect("put_task_meta_state");

    let loaded = backend
        .get_task_meta_state(&task_id, &tenant, MS_PREFIX)
        .await
        .expect("get_task_meta_state")
        .expect("should be Some");

    assert_eq!(loaded.task_id, task_id);
    assert_eq!(loaded.shared_understanding, "Test understanding.");
    assert_eq!(loaded.thinking_iterations, 2);
}

#[tokio::test]
async fn meta_state_get_returns_none_when_absent() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    let result = backend
        .get_task_meta_state(&TaskId::new(), &tenant, MS_PREFIX)
        .await
        .expect("get_task_meta_state");
    assert!(result.is_none());
}

#[tokio::test]
async fn list_meta_states_includes_written_entries() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();

    let task_id = TaskId::new();
    let meta = make_meta_state(task_id.clone(), tenant.clone());
    backend
        .put_task_meta_state(&meta, MS_PREFIX)
        .await
        .expect("put");

    let list = backend.list_task_meta_states(&tenant, MS_PREFIX, 100).await;
    let ids: Vec<String> = list.iter().map(|m| m.task_id.to_string()).collect();
    assert!(ids.contains(&task_id.to_string()));
}

#[tokio::test]
async fn list_meta_states_respects_limit() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();

    for _ in 0..5 {
        let meta = make_meta_state(TaskId::new(), tenant.clone());
        backend
            .put_task_meta_state(&meta, MS_PREFIX)
            .await
            .expect("put");
    }

    let list = backend.list_task_meta_states(&tenant, MS_PREFIX, 3).await;
    assert_eq!(list.len(), 3);
}

#[tokio::test]
async fn list_meta_states_isolates_by_tenant() {
    let backend = InMemoryStateBackend::new();
    let tenant_a = TenantId::default_tenant();
    let tenant_b = TenantId::from("other-tenant");

    let id_a = TaskId::new();
    let id_b = TaskId::new();
    backend
        .put_task_meta_state(&make_meta_state(id_a.clone(), tenant_a.clone()), MS_PREFIX)
        .await
        .expect("put a");
    backend
        .put_task_meta_state(&make_meta_state(id_b.clone(), tenant_b.clone()), MS_PREFIX)
        .await
        .expect("put b");

    let list_a = backend
        .list_task_meta_states(&tenant_a, MS_PREFIX, 100)
        .await;
    let ids_a: Vec<String> = list_a.iter().map(|m| m.task_id.to_string()).collect();
    assert!(ids_a.contains(&id_a.to_string()));
    assert!(!ids_a.contains(&id_b.to_string()));
}

#[tokio::test]
async fn list_meta_states_empty_returns_empty_vec() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    let list = backend.list_task_meta_states(&tenant, MS_PREFIX, 100).await;
    assert!(list.is_empty());
}
