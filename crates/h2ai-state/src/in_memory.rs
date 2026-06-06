//! In-memory implementation of [`SnapshotStore`], [`CalibrationStore`], and
//! [`EventPublisher`] for unit tests.
//!
//! Stored data lives entirely in `Arc<RwLock<HashMap>>` — no NATS, no I/O.
//! Routes (and any other consumer that depends on the trait abstractions in
//! [`crate::backend`]) can be tested without a live NATS server.
//!
//! Events published via [`EventPublisher`] are appended to an in-memory log so
//! tests can assert on what was emitted.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures::stream::BoxStream;

use async_trait::async_trait;
use tokio::sync::RwLock;

use h2ai_types::calibration::CalibrationRecord;
use h2ai_types::checkpoint::TaskCheckpoint;
use h2ai_types::conflict::ConflictRateAccumulator;
use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent, TaskSnapshot};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::prompt_variant::{AdapterOproState, PromptVariant};
use h2ai_types::reasoning_checkpoint::{TaskMetaState, TaskReasoningCheckpoint};

use crate::backend::{
    CalibrationStore, ConflictStore, EstimatorStore, EventPublisher, OproStore, ReasoningStore,
    ShadowDomainStore, SignalPublisher, SignalSubscriber, SkillStore, SnapshotStore, TailEvents,
    TaskCheckpointStore,
};
use crate::nats::NatsError;

/// A single event captured by the in-memory publisher.
///
/// `subject` is the explicit subject for `publish_to`; for the task-scoped
/// publishers we store `h2ai.tasks.<task_id>` so tests have the full picture.
#[derive(Debug, Clone)]
pub struct CapturedEvent {
    pub subject: String,
    pub event: H2AIEvent,
    pub seq: u64,
}

/// Zero-I/O backend that satisfies the full [`crate::backend::StateBackend`]
/// surface. Designed for unit tests of route handlers and engine code paths
/// that previously required a running NATS server.
#[derive(Default, Clone)]
pub struct InMemoryStateBackend {
    snapshots: Arc<RwLock<HashMap<String, TaskSnapshot>>>,
    calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>>,
    calibration_records: Arc<RwLock<HashMap<String, CalibrationRecord>>>,
    events: Arc<RwLock<Vec<CapturedEvent>>>,
    next_seq: Arc<AtomicU64>,
    // OproStore fields
    opro_states: Arc<RwLock<HashMap<String, AdapterOproState>>>,
    prompt_variants: Arc<RwLock<HashMap<String, PromptVariant>>>,
    active_variant_ptrs: Arc<RwLock<HashMap<String, String>>>,
    // EstimatorStore fields
    tao_states: Arc<RwLock<HashMap<String, (f64, usize)>>>,
    srani_states: Arc<RwLock<HashMap<String, (f64, usize)>>>,
    bandit_states: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    // ReasoningStore fields
    reasoning_checkpoints: Arc<RwLock<HashMap<String, TaskReasoningCheckpoint>>>,
    task_meta_states: Arc<RwLock<HashMap<String, TaskMetaState>>>,
    // ConflictStore fields
    conflict_accumulators: Arc<RwLock<HashMap<String, ConflictRateAccumulator>>>,
    // ShadowDomainStore fields
    shadow_domains: Arc<RwLock<HashSet<String>>>,
    // TaskCheckpointStore fields
    task_checkpoints: Arc<RwLock<HashMap<String, TaskCheckpoint>>>,
    // SkillStore fields
    skill_nodes: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl InMemoryStateBackend {
    /// Construct an empty backend with no snapshots, no calibration, and no
    /// captured events.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of every event captured so far. Useful for test assertions.
    pub async fn events(&self) -> Vec<CapturedEvent> {
        self.events.read().await.clone()
    }

    /// Convenience helper: count events for a given `task_id` prefix
    /// (subject `h2ai.tasks.<task_id>`).
    pub async fn event_count_for_task(&self, task_id: &TaskId) -> usize {
        let prefix = format!("h2ai.tasks.{task_id}");
        self.events
            .read()
            .await
            .iter()
            .filter(|e| e.subject == prefix)
            .count()
    }
}

#[async_trait]
impl SnapshotStore for InMemoryStateBackend {
    async fn put_snapshot(&self, snap: &TaskSnapshot) -> Result<(), NatsError> {
        self.snapshots
            .write()
            .await
            .insert(snap.task_id.to_string(), snap.clone());
        Ok(())
    }

    async fn get_snapshot(&self, task_id: &TaskId) -> Result<Option<TaskSnapshot>, NatsError> {
        Ok(self
            .snapshots
            .read()
            .await
            .get(&task_id.to_string())
            .cloned())
    }
}

#[async_trait]
impl CalibrationStore for InMemoryStateBackend {
    async fn put_calibration(&self, cal: &CalibrationCompletedEvent) -> Result<(), NatsError> {
        *self.calibration.write().await = Some(cal.clone());
        Ok(())
    }

    async fn get_calibration(&self) -> Result<Option<CalibrationCompletedEvent>, NatsError> {
        Ok(self.calibration.read().await.clone())
    }

    async fn get_calibration_record(
        &self,
        adapter_profile: &str,
    ) -> Result<Option<CalibrationRecord>, NatsError> {
        Ok(self
            .calibration_records
            .read()
            .await
            .get(adapter_profile)
            .cloned())
    }

    async fn put_calibration_record(&self, record: &CalibrationRecord) -> Result<(), NatsError> {
        self.calibration_records
            .write()
            .await
            .insert(record.adapter_profile.clone(), record.clone());
        Ok(())
    }
}

#[async_trait]
impl EventPublisher for InMemoryStateBackend {
    async fn publish_event(&self, task_id: &TaskId, event: &H2AIEvent) -> Result<(), NatsError> {
        let subject = format!("h2ai.tasks.{task_id}");
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst) + 1;
        self.events.write().await.push(CapturedEvent {
            subject,
            event: event.clone(),
            seq,
        });
        Ok(())
    }

    async fn publish_to(&self, subject: &str, event: &H2AIEvent) -> Result<(), NatsError> {
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst) + 1;
        self.events.write().await.push(CapturedEvent {
            subject: subject.to_owned(),
            event: event.clone(),
            seq,
        });
        Ok(())
    }

    async fn publish_event_seq(
        &self,
        task_id: &TaskId,
        event: &H2AIEvent,
    ) -> Result<u64, NatsError> {
        let subject = format!("h2ai.tasks.{task_id}");
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst) + 1;
        self.events.write().await.push(CapturedEvent {
            subject,
            event: event.clone(),
            seq,
        });
        Ok(seq)
    }
}

#[async_trait]
impl SignalPublisher for InMemoryStateBackend {
    async fn publish_signal(
        &self,
        _signal: &h2ai_types::signal::ResumeSignal,
    ) -> Result<(), crate::nats::NatsError> {
        Ok(())
    }
}

#[async_trait]
impl TailEvents for InMemoryStateBackend {
    async fn tail_task_events_boxed(
        &self,
        task_id: &TaskId,
        from_seq: u64,
    ) -> Result<BoxStream<'static, Result<(u64, H2AIEvent), NatsError>>, NatsError> {
        use futures::StreamExt;
        let prefix = format!("h2ai.tasks.{task_id}");
        let items: Vec<Result<(u64, H2AIEvent), NatsError>> = self
            .events
            .read()
            .await
            .iter()
            .filter(|e| e.subject == prefix && e.seq > from_seq)
            .map(|e| Ok((e.seq, e.event.clone())))
            .collect();
        Ok(futures::stream::iter(items).boxed())
    }
}

#[async_trait]
impl OproStore for InMemoryStateBackend {
    async fn put_prompt_variant(&self, variant: &PromptVariant) -> Result<(), NatsError> {
        let key = format!(
            "{}/{}/{}",
            variant.adapter_name, variant.prompt_key, variant.variant_id
        );
        self.prompt_variants
            .write()
            .await
            .insert(key, variant.clone());
        Ok(())
    }

    async fn get_prompt_variant(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<Option<PromptVariant>, NatsError> {
        let key = format!("{adapter_name}/{prompt_key}/{variant_id}");
        Ok(self.prompt_variants.read().await.get(&key).cloned())
    }

    async fn get_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
    ) -> Result<Option<String>, NatsError> {
        let key = format!("{adapter_name}/{prompt_key}");
        Ok(self.active_variant_ptrs.read().await.get(&key).cloned())
    }

    async fn set_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<(), NatsError> {
        let key = format!("{adapter_name}/{prompt_key}");
        self.active_variant_ptrs
            .write()
            .await
            .insert(key, variant_id.to_owned());
        Ok(())
    }

    async fn get_adapter_opro_state(
        &self,
        adapter_name: &str,
    ) -> Result<Option<AdapterOproState>, NatsError> {
        Ok(self.opro_states.read().await.get(adapter_name).cloned())
    }

    async fn put_adapter_opro_state(&self, state: &AdapterOproState) -> Result<(), NatsError> {
        self.opro_states
            .write()
            .await
            .insert(state.adapter_name.clone(), state.clone());
        Ok(())
    }
}

#[async_trait]
impl EstimatorStore for InMemoryStateBackend {
    async fn get_tao_estimator_state(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Option<(f64, usize)>, NatsError> {
        let key = format!("{}/tao", tenant_id.bucket_safe());
        Ok(self.tao_states.read().await.get(&key).copied())
    }

    async fn put_tao_estimator_state(
        &self,
        tenant_id: &TenantId,
        ema: f64,
        count: usize,
    ) -> Result<(), NatsError> {
        let key = format!("{}/tao", tenant_id.bucket_safe());
        self.tao_states.write().await.insert(key, (ema, count));
        Ok(())
    }

    async fn get_srani_state(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Option<(f64, usize)>, NatsError> {
        let key = format!("{}/srani", tenant_id.bucket_safe());
        Ok(self.srani_states.read().await.get(&key).copied())
    }

    async fn put_srani_state(
        &self,
        tenant_id: &TenantId,
        ema_cfi: f64,
        count: usize,
    ) -> Result<(), NatsError> {
        let key = format!("{}/srani", tenant_id.bucket_safe());
        self.srani_states
            .write()
            .await
            .insert(key, (ema_cfi, count));
        Ok(())
    }

    async fn get_bandit_state(&self, tenant_id: &TenantId) -> Result<Option<Vec<u8>>, NatsError> {
        let key = format!("{}/bandit", tenant_id.bucket_safe());
        Ok(self.bandit_states.read().await.get(&key).cloned())
    }

    async fn put_bandit_state(
        &self,
        tenant_id: &TenantId,
        json_bytes: Vec<u8>,
    ) -> Result<(), NatsError> {
        let key = format!("{}/bandit", tenant_id.bucket_safe());
        self.bandit_states.write().await.insert(key, json_bytes);
        Ok(())
    }
}

#[async_trait]
impl SkillStore for InMemoryStateBackend {
    async fn put_skill_nodes(
        &self,
        tenant_id: &TenantId,
        json_bytes: Vec<u8>,
    ) -> Result<(), NatsError> {
        self.skill_nodes
            .write()
            .await
            .insert(tenant_id.bucket_safe(), json_bytes);
        Ok(())
    }

    async fn get_skill_nodes(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Vec<u8>, NatsError> {
        Ok(self
            .skill_nodes
            .read()
            .await
            .get(&tenant_id.bucket_safe())
            .cloned()
            .unwrap_or_default())
    }
}

#[async_trait]
impl ReasoningStore for InMemoryStateBackend {
    async fn ensure_reasoning_buckets(
        &self,
        _tenant_id: &TenantId,
        _checkpoint_prefix: &str,
        _meta_state_prefix: &str,
    ) -> Result<(), NatsError> {
        Ok(())
    }

    async fn put_reasoning_checkpoint(
        &self,
        checkpoint: &TaskReasoningCheckpoint,
        _checkpoint_prefix: &str,
    ) -> Result<(), NatsError> {
        let key = format!("{}/{}", checkpoint.tenant_id, checkpoint.task_id);
        self.reasoning_checkpoints
            .write()
            .await
            .insert(key, checkpoint.clone());
        Ok(())
    }

    async fn get_reasoning_checkpoint(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
        _checkpoint_prefix: &str,
    ) -> Result<Option<TaskReasoningCheckpoint>, NatsError> {
        let key = format!("{tenant_id}/{task_id}");
        Ok(self.reasoning_checkpoints.read().await.get(&key).cloned())
    }

    async fn put_task_meta_state(
        &self,
        meta: &TaskMetaState,
        _meta_state_prefix: &str,
    ) -> Result<(), NatsError> {
        let key = format!("{}/{}", meta.tenant_id, meta.task_id);
        self.task_meta_states
            .write()
            .await
            .insert(key, meta.clone());
        Ok(())
    }

    async fn get_task_meta_state(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
        _meta_state_prefix: &str,
    ) -> Result<Option<TaskMetaState>, NatsError> {
        let key = format!("{tenant_id}/{task_id}");
        Ok(self.task_meta_states.read().await.get(&key).cloned())
    }

    async fn list_task_meta_states(
        &self,
        tenant_id: &TenantId,
        _meta_state_prefix: &str,
        limit: usize,
    ) -> Vec<TaskMetaState> {
        let prefix = format!("{tenant_id}/");
        self.task_meta_states
            .read()
            .await
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .take(limit)
            .map(|(_, v)| v.clone())
            .collect()
    }
}

#[async_trait]
impl ConflictStore for InMemoryStateBackend {
    async fn ensure_conflict_bucket(
        &self,
        _tenant_id: &TenantId,
        _bucket_prefix: &str,
    ) -> Result<(), crate::nats::NatsError> {
        Ok(())
    }

    async fn get_conflict_accumulator(
        &self,
        tenant_id: &TenantId,
        _bucket_prefix: &str,
    ) -> Result<Option<ConflictRateAccumulator>, crate::nats::NatsError> {
        Ok(self
            .conflict_accumulators
            .read()
            .await
            .get(&tenant_id.to_string())
            .cloned())
    }

    async fn put_conflict_accumulator(
        &self,
        acc: &ConflictRateAccumulator,
        _bucket_prefix: &str,
    ) -> Result<(), crate::nats::NatsError> {
        self.conflict_accumulators
            .write()
            .await
            .insert(acc.tenant_id.to_string(), acc.clone());
        Ok(())
    }
}

#[async_trait]
impl SignalSubscriber for InMemoryStateBackend {
    async fn subscribe_signals(
        &self,
        _task_id: &TaskId,
        _tenant_id: &TenantId,
    ) -> Result<
        futures::stream::BoxStream<'static, Result<h2ai_types::signal::ResumeSignal, crate::nats::NatsError>>,
        crate::nats::NatsError,
    > {
        use futures::stream;
        Ok(Box::pin(stream::empty()))
    }

    async fn delete_signal_consumer(&self, _task_id: &TaskId) -> Result<(), crate::nats::NatsError> {
        Ok(())
    }
}

#[async_trait]
impl ShadowDomainStore for InMemoryStateBackend {
    async fn put_shadow_promoted_domains(
        &self,
        domains: &HashSet<String>,
    ) -> Result<(), crate::nats::NatsError> {
        *self.shadow_domains.write().await = domains.clone();
        Ok(())
    }

    async fn get_shadow_promoted_domains(&self) -> Result<HashSet<String>, crate::nats::NatsError> {
        Ok(self.shadow_domains.read().await.clone())
    }
}

#[async_trait]
impl TaskCheckpointStore for InMemoryStateBackend {
    async fn list_task_checkpoints(&self) -> Vec<TaskCheckpoint> {
        self.task_checkpoints.read().await.values().cloned().collect()
    }

    async fn put_task_checkpoint(
        &self,
        cp: &TaskCheckpoint,
        _expected_revision: Option<u64>,
    ) -> Result<u64, crate::nats::NatsError> {
        self.task_checkpoints
            .write()
            .await
            .insert(cp.task_id.clone(), cp.clone());
        Ok(1)
    }

    async fn get_task_checkpoint(
        &self,
        task_id: &str,
    ) -> Result<Option<TaskCheckpoint>, crate::nats::NatsError> {
        Ok(self.task_checkpoints.read().await.get(task_id).cloned())
    }

    async fn delete_task_checkpoint(&self, task_id: &str) -> Result<(), crate::nats::NatsError> {
        self.task_checkpoints.write().await.remove(task_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SkillStore;
    use h2ai_types::identity::TenantId;

    #[tokio::test]
    async fn skill_store_roundtrip() {
        let backend = InMemoryStateBackend::new();
        let tenant = TenantId::default_tenant();
        let bytes = b"[\"skill-node-json\"]".to_vec();

        backend.put_skill_nodes(&tenant, bytes.clone()).await.unwrap();
        let loaded = backend.get_skill_nodes(&tenant).await.unwrap();
        assert_eq!(loaded, bytes);
    }

    #[tokio::test]
    async fn skill_store_empty_returns_empty_vec() {
        let backend = InMemoryStateBackend::new();
        let tenant = TenantId::default_tenant();
        let loaded = backend.get_skill_nodes(&tenant).await.unwrap();
        assert!(loaded.is_empty());
    }
}
