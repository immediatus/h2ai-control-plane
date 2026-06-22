//! State backend abstractions.
//!
//! These traits decouple route handlers (and other consumers) from the concrete
//! `NatsClient` so production code can depend on abstractions while tests use an
//! in-memory backend. The traits are intentionally narrow (Interface Segregation)
//! and object-safe (no generic methods, only `&self`) so they can be stored as
//! `Arc<dyn ...>` in `AppState`.
//!
//! `NatsClient` implements all of them (see `nats.rs`), and `InMemoryStateBackend`
//! implements them with `HashMap`-backed storage (see `in_memory.rs`).
//!
//! Composite trait [`StateBackend`] is automatically implemented for any type that
//! implements all the component traits via the blanket impl below.

use async_trait::async_trait;
use futures::stream::BoxStream;
use std::sync::Arc;

use h2ai_types::calibration::CalibrationRecord;
use h2ai_types::checkpoint::TaskCheckpoint;
use h2ai_types::conflict::ConflictRateAccumulator;
use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent, TaskSnapshot};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::prompt_variant::{AdapterOproState, PromptVariant};
use h2ai_types::reasoning_checkpoint::{TaskMetaState, TaskReasoningCheckpoint};
use h2ai_types::signal::ResumeSignal;
use std::collections::HashSet;

use crate::nats::NatsError;

/// Publishes `H2AIEvent`s to the orchestration stream (or an in-memory sink).
#[async_trait]
pub trait EventPublisher: Send + Sync {
    /// Publish an event to the default `h2ai.tasks.<task_id>` subject.
    async fn publish_event(&self, task_id: &TaskId, event: &H2AIEvent) -> Result<(), NatsError>;

    /// Publish an event to an arbitrary subject (used for calibration SSE, etc.).
    async fn publish_to(&self, subject: &str, event: &H2AIEvent) -> Result<(), NatsError>;

    /// Publish an event and return the assigned `JetStream` sequence number.
    async fn publish_event_seq(
        &self,
        task_id: &TaskId,
        event: &H2AIEvent,
    ) -> Result<u64, NatsError>;
}

/// Persists / loads `TaskSnapshot`s for crash-recovery.
#[async_trait]
pub trait SnapshotStore: Send + Sync {
    async fn put_snapshot(&self, snap: &TaskSnapshot) -> Result<(), NatsError>;
    async fn get_snapshot(&self, task_id: &TaskId) -> Result<Option<TaskSnapshot>, NatsError>;
}

/// Persists / loads calibration artefacts (latest pool-level calibration plus
/// per-adapter `CalibrationRecord` telemetry).
#[async_trait]
pub trait CalibrationStore: Send + Sync {
    /// Persist the latest pool-level calibration result.
    async fn put_calibration(&self, cal: &CalibrationCompletedEvent) -> Result<(), NatsError>;

    /// Retrieve the latest pool-level calibration result, or `None` if absent.
    async fn get_calibration(&self) -> Result<Option<CalibrationCompletedEvent>, NatsError>;

    /// Retrieve a per-adapter `CalibrationRecord`, or `None` if absent.
    async fn get_calibration_record(
        &self,
        adapter_profile: &str,
    ) -> Result<Option<CalibrationRecord>, NatsError>;

    /// Persist a per-adapter `CalibrationRecord`. The record's `adapter_profile`
    /// field is used as the storage key.
    async fn put_calibration_record(&self, record: &CalibrationRecord) -> Result<(), NatsError>;
}

/// Streams historical events for a task from sequence `from_seq` onward.
///
/// Implemented by `NatsClient` (reads from `JetStream`) and `InMemoryStateBackend`
/// (replays from the in-memory event log) so `SessionJournal` can be tested
/// without a live NATS connection.
#[async_trait]
pub trait TailEvents: Send + Sync {
    /// Returns a boxed stream of `(sequence, event)` pairs for `task_id`.
    ///
    /// `from_seq = 0` starts from the beginning; any non-zero value starts
    /// from the first event **after** that sequence number, matching the
    /// `JetStream` `ByStartSequence` semantics.
    async fn tail_task_events_boxed(
        &self,
        task_id: &TaskId,
        from_seq: u64,
    ) -> Result<BoxStream<'static, Result<(u64, H2AIEvent), NatsError>>, NatsError>;
}

/// Composite trait used by [`SessionJournal`]: snapshot persistence + event tailing.
///
/// Implemented automatically for any type that implements both `SnapshotStore`
/// and `TailEvents` via the blanket impl below.
pub trait SessionJournalBackend: SnapshotStore + TailEvents + Send + Sync {}

impl<T> SessionJournalBackend for T where T: SnapshotStore + TailEvents + Send + Sync {}

/// Composite trait: production code that needs the full state surface depends on this.
///
/// Any type implementing all three component traits gets `StateBackend` automatically
/// via the blanket impl below.
pub trait StateBackend: EventPublisher + SnapshotStore + CalibrationStore + Send + Sync {}

impl<T> StateBackend for T where T: EventPublisher + SnapshotStore + CalibrationStore + Send + Sync {}

/// Publishes `ResumeSignal`s to the signals `JetStream` stream (or a no-op sink in tests).
#[async_trait]
pub trait SignalPublisher: Send + Sync {
    async fn publish_signal(
        &self,
        signal: &h2ai_types::signal::ResumeSignal,
    ) -> Result<(), NatsError>;
}

/// Persists and retrieves OPRO prompt variants and adapter OPRO state.
#[async_trait]
pub trait OproStore: Send + Sync {
    async fn put_prompt_variant(&self, variant: &PromptVariant) -> Result<(), NatsError>;
    async fn get_prompt_variant(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<Option<PromptVariant>, NatsError>;
    async fn get_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
    ) -> Result<Option<String>, NatsError>;
    async fn set_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<(), NatsError>;
    async fn get_adapter_opro_state(
        &self,
        adapter_name: &str,
    ) -> Result<Option<AdapterOproState>, NatsError>;
    async fn put_adapter_opro_state(&self, state: &AdapterOproState) -> Result<(), NatsError>;
}

/// Persists and retrieves per-tenant estimator state (TAO EMA, bandit).
#[async_trait]
pub trait EstimatorStore: Send + Sync {
    async fn get_tao_estimator_state(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Option<(f64, usize)>, NatsError>;
    async fn put_tao_estimator_state(
        &self,
        tenant_id: &TenantId,
        ema: f64,
        count: usize,
    ) -> Result<(), NatsError>;
    async fn get_bandit_state(&self, tenant_id: &TenantId) -> Result<Option<Vec<u8>>, NatsError>;
    async fn put_bandit_state(
        &self,
        tenant_id: &TenantId,
        json_bytes: Vec<u8>,
    ) -> Result<(), NatsError>;
}

/// Persists and retrieves per-tenant skill nodes extracted from task resolution traces.
/// Values are raw JSON bytes (`serde_json::to_vec(&Vec<KnowledgeNode>)`) so this trait
/// does not depend on `h2ai-knowledge`. Callers in `h2ai-api` handle (de)serialization.
#[async_trait]
pub trait SkillStore: Send + Sync {
    /// Overwrite the tenant's skill node list with `json_bytes`.
    async fn put_skill_nodes(
        &self,
        tenant_id: &TenantId,
        json_bytes: Vec<u8>,
    ) -> Result<(), NatsError>;

    /// Load the tenant's skill node list as raw JSON bytes.
    /// Returns an empty `Vec` (not an error) when no entry exists.
    async fn get_skill_nodes(&self, tenant_id: &TenantId) -> Result<Vec<u8>, NatsError>;
}

#[async_trait]
impl<T: SkillStore + Send + Sync + ?Sized> SkillStore for Arc<T> {
    async fn put_skill_nodes(
        &self,
        tenant_id: &TenantId,
        json_bytes: Vec<u8>,
    ) -> Result<(), NatsError> {
        (**self).put_skill_nodes(tenant_id, json_bytes).await
    }
    async fn get_skill_nodes(&self, tenant_id: &TenantId) -> Result<Vec<u8>, NatsError> {
        (**self).get_skill_nodes(tenant_id).await
    }
}

/// Persists and retrieves reasoning checkpoints and task meta-state records.
///
/// Implemented by `NatsClient` (NATS KV buckets) and `InMemoryStateBackend`
/// (in-memory `HashMap`s) so that engine code can be tested without a live
/// NATS server.
#[async_trait]
pub trait ReasoningStore: Send + Sync {
    /// Create per-tenant KV buckets if they do not already exist.
    /// On `InMemoryStateBackend` this is a no-op (always succeeds).
    async fn ensure_reasoning_buckets(
        &self,
        tenant_id: &TenantId,
        checkpoint_prefix: &str,
        meta_state_prefix: &str,
    ) -> Result<(), NatsError>;

    /// Write (or overwrite) a reasoning checkpoint. Key: `task_id` string.
    async fn put_reasoning_checkpoint(
        &self,
        checkpoint: &TaskReasoningCheckpoint,
        checkpoint_prefix: &str,
    ) -> Result<(), NatsError>;

    /// Load a reasoning checkpoint by `task_id`. Returns `None` if not found.
    async fn get_reasoning_checkpoint(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
        checkpoint_prefix: &str,
    ) -> Result<Option<TaskReasoningCheckpoint>, NatsError>;

    /// Write (or overwrite) a task meta-state record. Key: `task_id` string.
    async fn put_task_meta_state(
        &self,
        meta: &TaskMetaState,
        meta_state_prefix: &str,
    ) -> Result<(), NatsError>;

    /// Load a task meta-state by `task_id`. Returns `None` if not found.
    async fn get_task_meta_state(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
        meta_state_prefix: &str,
    ) -> Result<Option<TaskMetaState>, NatsError>;

    /// List up to `limit` meta-state records for a tenant.
    /// Entries that fail to deserialize are silently skipped.
    async fn list_task_meta_states(
        &self,
        tenant_id: &TenantId,
        meta_state_prefix: &str,
        limit: usize,
    ) -> Vec<TaskMetaState>;
}

/// Persists and retrieves the per-tenant conflict-rate accumulator used by
/// the beta-quality adjustment in the engine.
///
/// The `bucket_prefix` argument is a caller-supplied configuration string
/// (e.g. `"h2ai-conflict"`) that scopes the storage key.  On the in-memory
/// backend the prefix is accepted but not used for routing — a single HashMap
/// entry per tenant suffices for unit tests.
#[async_trait]
pub trait ConflictStore: Send + Sync {
    /// Ensure the backing bucket exists.  On the in-memory backend this is a no-op.
    async fn ensure_conflict_bucket(
        &self,
        tenant_id: &TenantId,
        bucket_prefix: &str,
    ) -> Result<(), NatsError>;

    /// Load the conflict-rate accumulator for a tenant.  Returns `None` if no
    /// record has been written yet.
    async fn get_conflict_accumulator(
        &self,
        tenant_id: &TenantId,
        bucket_prefix: &str,
    ) -> Result<Option<ConflictRateAccumulator>, NatsError>;

    /// Write (or overwrite) the conflict-rate accumulator.
    async fn put_conflict_accumulator(
        &self,
        acc: &ConflictRateAccumulator,
        bucket_prefix: &str,
    ) -> Result<(), NatsError>;
}

#[async_trait]
impl<T: ConflictStore + Send + Sync + ?Sized> ConflictStore for Arc<T> {
    async fn ensure_conflict_bucket(
        &self,
        tenant_id: &TenantId,
        bucket_prefix: &str,
    ) -> Result<(), NatsError> {
        (**self)
            .ensure_conflict_bucket(tenant_id, bucket_prefix)
            .await
    }
    async fn get_conflict_accumulator(
        &self,
        tenant_id: &TenantId,
        bucket_prefix: &str,
    ) -> Result<Option<ConflictRateAccumulator>, NatsError> {
        (**self)
            .get_conflict_accumulator(tenant_id, bucket_prefix)
            .await
    }
    async fn put_conflict_accumulator(
        &self,
        acc: &ConflictRateAccumulator,
        bucket_prefix: &str,
    ) -> Result<(), NatsError> {
        (**self).put_conflict_accumulator(acc, bucket_prefix).await
    }
}

/// Subscribes to and cleans up the per-task signal stream used by the engine's
/// wave-boundary window and HITL gate.
///
/// `subscribe_signals` returns a `BoxStream` so the trait is object-safe and
/// consumers never depend on NATS-specific stream types.  On the in-memory
/// backend the stream yields nothing — unit tests that need signal delivery
/// use a `MockNatsBackend` instead.
#[async_trait]
pub trait SignalSubscriber: Send + Sync {
    async fn subscribe_signals(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
    ) -> Result<BoxStream<'static, Result<ResumeSignal, NatsError>>, NatsError>;

    /// Delete the durable pull consumer created by `subscribe_signals`.
    /// On the in-memory backend this is a no-op.
    async fn delete_signal_consumer(&self, task_id: &TaskId) -> Result<(), NatsError>;
}

#[async_trait]
impl<T: SignalSubscriber + Send + Sync + ?Sized> SignalSubscriber for Arc<T> {
    async fn subscribe_signals(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
    ) -> Result<BoxStream<'static, Result<ResumeSignal, NatsError>>, NatsError> {
        (**self).subscribe_signals(task_id, tenant_id).await
    }
    async fn delete_signal_consumer(&self, task_id: &TaskId) -> Result<(), NatsError> {
        (**self).delete_signal_consumer(task_id).await
    }
}

/// Persists the set of shadow-promoted domains for the two-auditor AND-vote mode.
///
/// On the in-memory backend the store is a single `Arc<RwLock<HashSet<String>>>`.
/// `get_shadow_promoted_domains` returns an empty set if nothing has been written.
#[async_trait]
pub trait ShadowDomainStore: Send + Sync {
    async fn put_shadow_promoted_domains(&self, domains: &HashSet<String>)
        -> Result<(), NatsError>;

    async fn get_shadow_promoted_domains(&self) -> Result<HashSet<String>, NatsError>;
}

#[async_trait]
impl<T: ShadowDomainStore + Send + Sync + ?Sized> ShadowDomainStore for Arc<T> {
    async fn put_shadow_promoted_domains(
        &self,
        domains: &HashSet<String>,
    ) -> Result<(), NatsError> {
        (**self).put_shadow_promoted_domains(domains).await
    }
    async fn get_shadow_promoted_domains(&self) -> Result<HashSet<String>, NatsError> {
        (**self).get_shadow_promoted_domains().await
    }
}

/// Persists and retrieves crash-recovery `TaskCheckpoint`s.
///
/// Distinct from `ReasoningStore` checkpoints (`TaskReasoningCheckpoint`) —
/// these live in a separate KV bucket and track in-flight task lease ownership
/// for node-level crash recovery.
///
/// `put_task_checkpoint` returns the storage revision (`u64`) so callers can
/// use compare-and-swap semantics (`expected_revision: Some(old_rev)`) for
/// atomic lease acquisition.  The in-memory impl always returns `1`.
#[async_trait]
pub trait TaskCheckpointStore: Send + Sync {
    async fn list_task_checkpoints(&self) -> Vec<TaskCheckpoint>;

    async fn put_task_checkpoint(
        &self,
        cp: &TaskCheckpoint,
        expected_revision: Option<u64>,
    ) -> Result<u64, NatsError>;

    async fn get_task_checkpoint(&self, task_id: &str)
        -> Result<Option<TaskCheckpoint>, NatsError>;

    async fn delete_task_checkpoint(&self, task_id: &str) -> Result<(), NatsError>;
}

#[async_trait]
impl<T: TaskCheckpointStore + Send + Sync + ?Sized> TaskCheckpointStore for Arc<T> {
    async fn list_task_checkpoints(&self) -> Vec<TaskCheckpoint> {
        (**self).list_task_checkpoints().await
    }
    async fn put_task_checkpoint(
        &self,
        cp: &TaskCheckpoint,
        expected_revision: Option<u64>,
    ) -> Result<u64, NatsError> {
        (**self).put_task_checkpoint(cp, expected_revision).await
    }
    async fn get_task_checkpoint(
        &self,
        task_id: &str,
    ) -> Result<Option<TaskCheckpoint>, NatsError> {
        (**self).get_task_checkpoint(task_id).await
    }
    async fn delete_task_checkpoint(&self, task_id: &str) -> Result<(), NatsError> {
        (**self).delete_task_checkpoint(task_id).await
    }
}

/// Composite supertrait: the single type that replaces `Arc<NatsClient>` in
/// `AppState` and `EngineInput`.
///
/// Any type that implements all twelve component traits receives `NatsBackend`
/// automatically via the blanket impl.  `NatsClient` satisfies the blanket once
/// Tasks 1–4 are complete.  `MockNatsBackend` in `h2ai-test-utils` satisfies it
/// via explicit `mockall::mock!` declarations.
pub trait NatsBackend:
    EventPublisher
    + SnapshotStore
    + CalibrationStore
    + TailEvents
    + SignalPublisher
    + SignalSubscriber
    + OproStore
    + EstimatorStore
    + SkillStore
    + ReasoningStore
    + ConflictStore
    + ShadowDomainStore
    + TaskCheckpointStore
    + Send
    + Sync
{
}

impl<T> NatsBackend for T where
    T: EventPublisher
        + SnapshotStore
        + CalibrationStore
        + TailEvents
        + SignalPublisher
        + SignalSubscriber
        + OproStore
        + EstimatorStore
        + SkillStore
        + ReasoningStore
        + ConflictStore
        + ShadowDomainStore
        + TaskCheckpointStore
        + Send
        + Sync
{
}

/// Narrow trait for `NatsDispatchAdapter`: dispatches work to edge agents via
/// NATS and waits for their result.
///
/// Kept separate from `NatsBackend` because `InMemoryStateBackend` has no concept
/// of edge-agent dispatch, and `AppState` / `EngineInput` never call these methods.
#[async_trait]
pub trait TaskDispatchBackend: Send + Sync {
    /// Publish a `TaskPayload` to the ephemeral task subject so an edge agent
    /// can pick it up.
    async fn publish_task_payload(
        &self,
        payload: &h2ai_types::agent::TaskPayload,
    ) -> Result<(), NatsError>;

    /// Subscribe to `H2AI_RESULTS` and return the first `TaskResult` for
    /// `task_id` within `timeout`.  Call this BEFORE `publish_task_payload`
    /// to avoid the race where the result arrives before the consumer exists.
    async fn await_task_result_once(
        &self,
        task_id: &TaskId,
        timeout: std::time::Duration,
    ) -> Result<h2ai_types::agent::TaskResult, NatsError>;
}

#[async_trait]
impl<T: TaskDispatchBackend + Send + Sync + ?Sized> TaskDispatchBackend for Arc<T> {
    async fn publish_task_payload(
        &self,
        payload: &h2ai_types::agent::TaskPayload,
    ) -> Result<(), NatsError> {
        (**self).publish_task_payload(payload).await
    }
    async fn await_task_result_once(
        &self,
        task_id: &TaskId,
        timeout: std::time::Duration,
    ) -> Result<h2ai_types::agent::TaskResult, NatsError> {
        (**self).await_task_result_once(task_id, timeout).await
    }
}

// ── Arc<T> forwarding impls ───────────────────────────────────────────────────
// Allow callers to pass `&Arc<T>` wherever `&T: OproStore` is expected.

#[async_trait]
impl<T: OproStore + ?Sized> OproStore for Arc<T> {
    async fn put_prompt_variant(&self, variant: &PromptVariant) -> Result<(), NatsError> {
        (**self).put_prompt_variant(variant).await
    }
    async fn get_prompt_variant(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<Option<PromptVariant>, NatsError> {
        (**self)
            .get_prompt_variant(adapter_name, prompt_key, variant_id)
            .await
    }
    async fn get_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
    ) -> Result<Option<String>, NatsError> {
        (**self)
            .get_active_variant_ptr(adapter_name, prompt_key)
            .await
    }
    async fn set_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<(), NatsError> {
        (**self)
            .set_active_variant_ptr(adapter_name, prompt_key, variant_id)
            .await
    }
    async fn get_adapter_opro_state(
        &self,
        adapter_name: &str,
    ) -> Result<Option<AdapterOproState>, NatsError> {
        (**self).get_adapter_opro_state(adapter_name).await
    }
    async fn put_adapter_opro_state(&self, state: &AdapterOproState) -> Result<(), NatsError> {
        (**self).put_adapter_opro_state(state).await
    }
}
