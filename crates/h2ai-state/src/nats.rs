#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
use async_nats::jetstream::{self, kv, stream};
use async_nats::Client;
use h2ai_config::StateConfig;
use h2ai_types::calibration::{AuditorHealth, CalibrationRecord};
use h2ai_types::checkpoint::TaskCheckpoint;
use h2ai_types::checkpoint_delta::{CheckpointKind, TaskCheckpointEntry};
use h2ai_types::conflict::ConflictRateAccumulator;
use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent, TaskSnapshot};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::prompt_variant::{AdapterOproState, PromptVariant};
use h2ai_types::reasoning_checkpoint::{TaskMetaState, TaskReasoningCheckpoint};
use h2ai_types::sizing::OracleObservation;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Debug, Error)]
pub enum NatsError {
    #[error("connection failed: {0}")]
    ConnectionFailed(#[from] async_nats::ConnectError),
    #[error("stream error: {0}")]
    StreamError(String),
    #[error("publish error: {0}")]
    PublishError(String),
    #[error("kv error: {0}")]
    KvError(String),
    #[error("subscribe error: {0}")]
    SubscribeError(String),
    #[error("serialization error: {0}")]
    Serialize(String),
}

/// In-memory cached reconstructed checkpoint for a task.
pub struct CachedCheckpoint {
    pub checkpoint: TaskCheckpoint,
    pub seq: u32,
    pub cached_at: std::time::Instant,
}

pub struct NatsClient {
    pub client: Client,
    jetstream: jetstream::Context,
    state_cfg: StateConfig,
    /// LRU cache of reconstructed checkpoints, keyed by `task_id` string.
    delta_cache: Arc<RwLock<LruCache<String, CachedCheckpoint>>>,
}

impl NatsClient {
    pub async fn connect(url: &str) -> Result<Self, NatsError> {
        Self::connect_with_cfg(url, StateConfig::default()).await
    }

    pub async fn connect_with_cfg(url: &str, state_cfg: StateConfig) -> Result<Self, NatsError> {
        let client = async_nats::connect(url).await?;
        let jetstream = jetstream::new(client.clone());
        let cache_size = NonZeroUsize::new(state_cfg.delta.cache_max_entries.max(1)).unwrap();
        Ok(Self {
            client,
            jetstream,
            delta_cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            state_cfg,
        })
    }

    /// Create all required `JetStream` streams and KV buckets.
    /// Safe to call multiple times — uses `get_or_create` semantics.
    #[allow(clippy::too_many_lines)]
    pub async fn ensure_infrastructure(&self) -> Result<(), NatsError> {
        // Stream 1: all task orchestration events (durable, file-backed)
        self.jetstream
            .get_or_create_stream(stream::Config {
                name: self.state_cfg.tasks_stream.clone(),
                subjects: vec!["h2ai.tasks.>".to_owned()],
                storage: stream::StorageType::File,
                retention: stream::RetentionPolicy::Limits,
                ..Default::default()
            })
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;

        // Stream 2: telemetry + audit (durable, file-backed, separate namespace)
        self.jetstream
            .get_or_create_stream(stream::Config {
                name: self.state_cfg.telemetry_stream.clone(),
                subjects: vec!["h2ai.telemetry.>".to_owned(), "audit.events.>".to_owned()],
                storage: stream::StorageType::File,
                retention: stream::RetentionPolicy::Limits,
                ..Default::default()
            })
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;

        // Stream 3: task result responses from edge agents
        self.jetstream
            .get_or_create_stream(stream::Config {
                name: self.state_cfg.results_stream.clone(),
                subjects: vec!["h2ai.results.>".to_owned()],
                storage: stream::StorageType::Memory,
                retention: stream::RetentionPolicy::WorkQueue,
                ..Default::default()
            })
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;

        // KV bucket: calibration cache
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.calibration_bucket.clone(),
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: durable session memory
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.sessions_bucket.clone(),
            description: "Durable session memory — pipeline conversation history".to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: task state snapshots for crash-recovery replay optimization
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.snapshots_bucket.clone(),
            description: "Task state snapshots — latest-only, accelerates replay after crash"
                .to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: TaoMultiplierEstimator EMA state for drift tracking
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.estimator_bucket.clone(),
            description: "TaoMultiplierEstimator EMA state — survives restarts".to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: per-tenant skill nodes extracted from task resolution traces
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.skills_bucket.clone(),
            description: "Per-tenant skill nodes extracted from resolved task traces".to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: rolling oracle calibration observations for conformal interval estimation
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.oracle_calibration_bucket.clone(),
            description: "Rolling oracle calibration window — max 200 OracleObservation entries"
                .to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: shadow auditor promoted domains (domains in two-auditor AND-vote mode)
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.audit_shadow_bucket.clone(),
            description: "Shadow auditor promoted domains — persisted across restarts".to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: task phase checkpoints for crash-recovery (zstd-compressed, latest-only)
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.task_checkpoints_bucket.clone(),
            description: "Task phase checkpoints — zstd-compressed, latest-only per task"
                .to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            max_age: std::time::Duration::from_hours(24), // 24h TTL
            ..Default::default()
        })
        .await?;

        // Object Store bucket: checkpoint payload overflow for entries > 800 KB
        self.ensure_object_store(async_nats::jetstream::object_store::Config {
            bucket: self.state_cfg.checkpoint_payloads_bucket.clone(),
            description: Some(
                "Checkpoint payload overflow — delete before KV entry on GC".to_owned(),
            ),
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: HITL approval records pending human decision
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.approvals_bucket.clone(),
            description: "HITL approval records awaiting human decision".to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            max_age: std::time::Duration::from_hours(1), // 1h TTL — longer than max review timeout
            ..Default::default()
        })
        .await?;

        // KV bucket: OPRO prompt variants and per-adapter bandit/OPRO state
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.prompt_variants_bucket.clone(),
            description: "OPRO prompt variants and per-adapter OPRO/bandit state".to_owned(),
            history: 5,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: per-adapter CalibrationRecord telemetry
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.calibration_records_bucket.clone(),
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: AuditorHealth circuit-breaker state
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.auditor_health_bucket.clone(),
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: probe lease CAS tokens (Memory: leases are ephemeral)
        self.ensure_kv_bucket(kv::Config {
            bucket: self.state_cfg.probe_lease_bucket.clone(),
            storage: stream::StorageType::Memory,
            ..Default::default()
        })
        .await?;

        // Stream 4: HITL resume signals (durable, per-task pull consumers)
        self.provision_signals_stream().await?;

        Ok(())
    }

    /// Publish a typed `H2AIEvent` to the task's `JetStream` subject.
    pub async fn publish_event(
        &self,
        task_id: &TaskId,
        event: &H2AIEvent,
    ) -> Result<(), NatsError> {
        let subject = format!("h2ai.tasks.{task_id}");
        self.publish_to(&subject, event).await
    }

    /// Publish a typed `H2AIEvent` to an arbitrary subject.
    pub async fn publish_to(&self, subject: &str, event: &H2AIEvent) -> Result<(), NatsError> {
        let payload = serde_json::to_vec(event).map_err(|e| NatsError::Serialize(e.to_string()))?;
        self.jetstream
            .publish(subject.to_owned(), payload.into())
            .await
            .map_err(|e| NatsError::PublishError(e.to_string()))?;
        Ok(())
    }

    /// Like `publish_event` but awaits the `PubAck` and returns the `JetStream` sequence number.
    /// Use when the caller needs the sequence for snapshot tracking.
    pub async fn publish_event_seq(
        &self,
        task_id: &TaskId,
        event: &H2AIEvent,
    ) -> Result<u64, NatsError> {
        let subject = format!("h2ai.tasks.{task_id}");
        let payload = serde_json::to_vec(event).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let ack_future = self
            .jetstream
            .publish(subject, payload.into())
            .await
            .map_err(|e| NatsError::PublishError(e.to_string()))?;
        let ack = ack_future
            .await
            .map_err(|e| NatsError::PublishError(e.to_string()))?;
        Ok(ack.sequence)
    }

    /// Write a task state snapshot to NATS KV. Key: `snapshots/{task_id}/latest`.
    pub async fn put_snapshot(&self, snapshot: &TaskSnapshot) -> Result<(), NatsError> {
        let key = format!("snapshots/{}/latest", snapshot.task_id);
        let payload =
            serde_json::to_vec(snapshot).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.snapshots_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put(&key, payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Load the latest task state snapshot, or `None` if no snapshot exists yet.
    pub async fn get_snapshot(&self, task_id: &TaskId) -> Result<Option<TaskSnapshot>, NatsError> {
        let key = format!("snapshots/{task_id}/latest");
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.snapshots_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get(&key).await {
            Ok(Some(entry)) => {
                let snapshot = serde_json::from_slice::<TaskSnapshot>(&entry)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(snapshot))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Persist the latest calibration result in the KV cache.
    pub async fn put_calibration(
        &self,
        event: &CalibrationCompletedEvent,
    ) -> Result<(), NatsError> {
        let payload = serde_json::to_vec(event).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.calibration_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put("latest", payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve the cached calibration result, or None if absent.
    pub async fn get_calibration(&self) -> Result<Option<CalibrationCompletedEvent>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.calibration_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get("latest").await {
            Ok(Some(entry)) => {
                let event = serde_json::from_slice::<CalibrationCompletedEvent>(&entry)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(event))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Persist a `CalibrationRecord` for an adapter profile.
    ///
    /// Key: `adapter_profile` field of the record.
    pub async fn put_calibration_record(
        &self,
        record: &CalibrationRecord,
    ) -> Result<(), NatsError> {
        let payload =
            serde_json::to_vec(record).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.calibration_records_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put(&record.adapter_profile, payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve a `CalibrationRecord` for an adapter profile, or None if absent.
    pub async fn get_calibration_record(
        &self,
        adapter_profile: &str,
    ) -> Result<Option<CalibrationRecord>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.calibration_records_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get(adapter_profile).await {
            Ok(Some(entry)) => {
                let record = serde_json::from_slice::<CalibrationRecord>(&entry)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(record))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Persist `AuditorHealth` for an adapter profile.
    pub async fn put_auditor_health(
        &self,
        adapter_profile: &str,
        health: &AuditorHealth,
    ) -> Result<(), NatsError> {
        let payload =
            serde_json::to_vec(health).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.auditor_health_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put(adapter_profile, payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve `AuditorHealth` for an adapter profile, or None if absent.
    pub async fn get_auditor_health(
        &self,
        adapter_profile: &str,
    ) -> Result<Option<AuditorHealth>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.auditor_health_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get(adapter_profile).await {
            Ok(Some(entry)) => {
                let health = serde_json::from_slice::<AuditorHealth>(&entry)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(health))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Attempt to acquire a probe lease for an adapter profile (`HalfOpen` CAS).
    ///
    /// Uses NATS KV `create` (atomic create-if-not-exists): only one caller wins per
    /// `ttl_secs` window. Returns `true` if this caller won the lease, `false` if another
    /// caller holds it.
    ///
    /// The lease value is the unix timestamp (seconds) at which it was acquired.
    pub async fn acquire_probe_lease(
        &self,
        adapter_profile: &str,
        ttl_secs: u64,
    ) -> Result<bool, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.probe_lease_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = adapter_profile;
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Check for stale lease before attempting create
        if let Ok(Some(existing)) = kv.get(key).await {
            if let Ok(s) = std::str::from_utf8(&existing) {
                if let Ok(acquired_at) = s.parse::<u64>() {
                    if now_secs.saturating_sub(acquired_at) < ttl_secs {
                        return Ok(false); // lease still held by another caller
                    }
                    // Stale lease — delete it so create can succeed
                    let _ = kv.delete(key).await;
                }
            }
        }
        let payload = now_secs.to_string().into_bytes();
        match kv.create(key, payload.into()).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false), // another caller won the race
        }
    }

    /// Release a probe lease for an adapter profile.
    ///
    /// A no-op if the lease does not exist.
    pub async fn release_probe_lease(&self, adapter_profile: &str) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.probe_lease_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let _ = kv.delete(adapter_profile).await;
        Ok(())
    }

    /// Persist the rolling oracle calibration observations window.
    ///
    /// Replaces the existing entry wholesale. Callers are responsible for
    /// enforcing the 200-observation FIFO cap before calling this.
    pub async fn put_oracle_observations(
        &self,
        observations: &[OracleObservation],
    ) -> Result<(), NatsError> {
        let payload =
            serde_json::to_vec(observations).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.oracle_calibration_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put("observations", payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve the stored oracle calibration observations, or empty vec if absent.
    pub async fn get_oracle_observations(&self) -> Result<Vec<OracleObservation>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.oracle_calibration_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get("observations").await {
            Ok(Some(entry)) => {
                let obs = serde_json::from_slice::<Vec<OracleObservation>>(&entry)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(obs)
            }
            Ok(None) => Ok(vec![]),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Persist the set of domains currently in two-auditor AND-vote mode.
    ///
    /// Stored as a JSON array of strings under key `"promoted"` in `H2AI_AUDIT_SHADOW`.
    pub async fn put_shadow_promoted_domains(
        &self,
        domains: &std::collections::HashSet<String>,
    ) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.audit_shadow_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let payload = serde_json::to_vec(domains).map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put("promoted", payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve the set of domains currently in two-auditor AND-vote mode.
    ///
    /// Returns an empty set if the key is absent (first startup).
    pub async fn get_shadow_promoted_domains(
        &self,
    ) -> Result<std::collections::HashSet<String>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.audit_shadow_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get("promoted").await {
            Ok(Some(entry)) => serde_json::from_slice::<std::collections::HashSet<String>>(&entry)
                .map_err(|e| NatsError::KvError(e.to_string())),
            Ok(None) => Ok(std::collections::HashSet::new()),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Persist the `TaoMultiplierEstimator` EMA state so it survives process restarts.
    pub async fn put_tao_estimator_state(
        &self,
        tenant_id: &TenantId,
        ema: f64,
        count: usize,
    ) -> Result<(), NatsError> {
        #[derive(serde::Serialize)]
        struct State {
            ema: f64,
            count: usize,
        }
        let payload = serde_json::to_vec(&State { ema, count })
            .map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.estimator_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}/tao", tenant_id.bucket_safe());
        kv.put(&key, payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve the persisted `TaoMultiplierEstimator` EMA state, or `None` if absent.
    pub async fn get_tao_estimator_state(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Option<(f64, usize)>, NatsError> {
        #[derive(serde::Deserialize)]
        struct State {
            ema: f64,
            count: usize,
        }
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.estimator_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}/tao", tenant_id.bucket_safe());
        match kv.get(&key).await {
            Ok(Some(entry)) => {
                let s: State = serde_json::from_slice(&entry)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some((s.ema, s.count)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Persist the SRANI adaptive EMA state so it survives process restarts.
    pub async fn put_srani_state(
        &self,
        tenant_id: &TenantId,
        ema_cfi: f64,
        count: usize,
    ) -> Result<(), NatsError> {
        #[derive(serde::Serialize)]
        struct State {
            ema_cfi: f64,
            count: usize,
        }
        let payload = serde_json::to_vec(&State { ema_cfi, count })
            .map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.estimator_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}/srani", tenant_id.bucket_safe());
        kv.put(&key, payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve the persisted SRANI adaptive EMA state, or `None` if absent.
    pub async fn get_srani_state(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Option<(f64, usize)>, NatsError> {
        #[derive(serde::Deserialize)]
        struct State {
            ema_cfi: f64,
            count: usize,
        }
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.estimator_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}/srani", tenant_id.bucket_safe());
        match kv.get(&key).await {
            Ok(Some(entry)) => {
                let s: State = serde_json::from_slice(&entry)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some((s.ema_cfi, s.count)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Persist raw JSON bytes to the `H2AI_ESTIMATOR` bucket under key `{tenant_safe}/bandit`.
    /// Callers are responsible for serialization (avoids a circular crate dependency).
    pub async fn put_bandit_state(
        &self,
        tenant_id: &TenantId,
        json_bytes: Vec<u8>,
    ) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.estimator_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}/bandit", tenant_id.bucket_safe());
        kv.put(&key, json_bytes.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve raw JSON bytes for `BanditState` from the `H2AI_ESTIMATOR` bucket.
    /// Returns `None` when no entry exists (first run). Callers deserialize the bytes.
    pub async fn get_bandit_state(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Option<Vec<u8>>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.estimator_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}/bandit", tenant_id.bucket_safe());
        match kv.get(&key).await {
            Ok(Some(entry)) => Ok(Some(entry.to_vec())),
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    pub async fn put_skill_nodes(
        &self,
        tenant_id: &TenantId,
        json_bytes: Vec<u8>,
    ) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.skills_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}/skills", tenant_id.bucket_safe());
        kv.put(&key, json_bytes.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    pub async fn get_skill_nodes(&self, tenant_id: &TenantId) -> Result<Vec<u8>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.skills_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}/skills", tenant_id.bucket_safe());
        match kv.get(&key).await {
            Ok(Some(entry)) => Ok(entry.to_vec()),
            Ok(None) => Ok(vec![]),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Write the active safety configuration as a JSON snapshot to NATS KV.
    ///
    /// Bucket: `H2AI_ESTIMATOR`. Key: `h2ai.config.safety_profile`.
    /// Overwrites any previous snapshot unconditionally.
    pub async fn put_safety_profile_snapshot(
        &self,
        cfg: &h2ai_config::SafetyConfig,
    ) -> Result<(), async_nats::Error> {
        #[derive(serde::Serialize)]
        struct SafetyProfileSnapshot {
            profile: String,
            krum_fault_tolerance: usize,
            diversity_threshold: f64,
            shadow_auditor_enabled: bool,
            require_bivariate_cg: bool,
            timestamp_ms: u64,
        }
        let snapshot = SafetyProfileSnapshot {
            profile: cfg.profile.as_str().to_string(),
            krum_fault_tolerance: cfg.krum_fault_tolerance,
            diversity_threshold: cfg.diversity_threshold,
            shadow_auditor_enabled: cfg.shadow_auditor.enabled,
            require_bivariate_cg: cfg.require_bivariate_cg,
            #[allow(clippy::cast_possible_truncation)]
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        let payload =
            serde_json::to_vec(&snapshot).map_err(|e| Box::new(e) as async_nats::Error)?;
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.estimator_bucket)
            .await?;
        kv.put("h2ai.config.safety_profile", payload.into()).await?;
        Ok(())
    }

    /// Open an ordered pull consumer on `h2ai.tasks.{task_id}` and return a stream of events.
    ///
    /// `from_seq = 0` delivers from the beginning; non-zero starts after that sequence.
    /// Uses an ordered pull consumer (no delivery subject required, ephemeral and replay-safe).
    pub async fn tail_task_events(
        &self,
        task_id: &TaskId,
        from_seq: u64,
    ) -> Result<
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<(u64, H2AIEvent), NatsError>> + Send>>,
        NatsError,
    > {
        use futures::StreamExt;
        use jetstream::consumer::DeliverPolicy;
        let subject = format!("h2ai.tasks.{task_id}");
        let deliver_policy = if from_seq == 0 {
            DeliverPolicy::All
        } else {
            DeliverPolicy::ByStartSequence {
                start_sequence: from_seq + 1,
            }
        };
        let consumer_cfg = jetstream::consumer::pull::OrderedConfig {
            filter_subject: subject,
            deliver_policy,
            ..Default::default()
        };
        let js_stream = self
            .jetstream
            .get_stream(&self.state_cfg.tasks_stream)
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;
        let consumer = js_stream
            .create_consumer(consumer_cfg)
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;
        let messages = consumer
            .messages()
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;
        let mapped = messages.map(|msg| {
            let msg = msg.map_err(|e| NatsError::StreamError(e.to_string()))?;
            let seq = msg.info().map_or(0, |i| i.stream_sequence);
            let event: H2AIEvent = serde_json::from_slice(&msg.payload)
                .map_err(|e| NatsError::Serialize(e.to_string()))?;
            Ok((seq, event))
        });
        Ok(Box::pin(mapped))
    }

    /// Publish a `TaskPayload` to the ephemeral task subject for an edge agent.
    /// Subject: `h2ai.tasks.ephemeral.{task_id`}  (core NATS, not `JetStream`)
    pub async fn publish_task_payload(
        &self,
        payload: &h2ai_types::agent::TaskPayload,
    ) -> Result<(), NatsError> {
        use h2ai_nats::subjects::ephemeral_task_subject;
        let subject = ephemeral_task_subject(&payload.task_id);
        let bytes = serde_json::to_vec(payload).map_err(|e| NatsError::Serialize(e.to_string()))?;
        self.client
            .publish(subject, bytes.into())
            .await
            .map_err(|e| NatsError::PublishError(e.to_string()))
    }

    /// Subscribe to `H2AI_RESULTS` `JetStream` and return the first `TaskResult`
    /// for the given `task_id` within the given timeout.
    ///
    /// IMPORTANT: Call this BEFORE `publish_task_payload` to avoid the race
    /// where the result message arrives before the consumer is created.
    pub async fn await_task_result_once(
        &self,
        task_id: &h2ai_types::identity::TaskId,
        timeout: std::time::Duration,
    ) -> Result<h2ai_types::agent::TaskResult, NatsError> {
        use futures::StreamExt;
        use h2ai_nats::subjects::task_result_subject;
        use jetstream::consumer::{AckPolicy, DeliverPolicy};

        let subject = task_result_subject(task_id);
        // WorkQueue retention requires AckPolicy::Explicit — OrderedConfig defaults to None.
        let consumer_cfg = jetstream::consumer::pull::Config {
            filter_subject: subject,
            deliver_policy: DeliverPolicy::All,
            ack_policy: AckPolicy::Explicit,
            ..Default::default()
        };
        let js_stream = self
            .jetstream
            .get_stream(&self.state_cfg.results_stream)
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;
        let consumer = js_stream
            .create_consumer(consumer_cfg)
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;
        let mut messages = consumer
            .messages()
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;

        let msg = tokio::time::timeout(timeout, messages.next())
            .await
            .map_err(|_| {
                NatsError::StreamError(format!(
                    "timeout waiting for task result: task_id={task_id}"
                ))
            })?
            .ok_or_else(|| NatsError::StreamError("result stream closed".into()))?
            .map_err(|e| NatsError::StreamError(e.to_string()))?;

        let result: h2ai_types::agent::TaskResult = serde_json::from_slice(&msg.payload)
            .map_err(|e| NatsError::Serialize(e.to_string()))?;

        // Ack so the work-queue message is deleted from the stream
        msg.ack()
            .await
            .map_err(|e| NatsError::StreamError(format!("ack failed: {e}")))?;

        Ok(result)
    }

    // ── prompt variants / OPRO state ────────────────────────────────────────

    /// Store a `PromptVariant` at key `{adapter_name}/{prompt_key}/{variant_id}`.
    pub async fn put_prompt_variant(&self, variant: &PromptVariant) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.prompt_variants_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!(
            "{}/{}/{}",
            variant.adapter_name, variant.prompt_key, variant.variant_id
        );
        let bytes = serde_json::to_vec(variant).map_err(|e| NatsError::Serialize(e.to_string()))?;
        kv.put(&key, bytes.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Fetch a `PromptVariant` by `adapter_name`, `prompt_key`, `variant_id`.
    /// Returns `None` if the key does not exist.
    pub async fn get_prompt_variant(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<Option<PromptVariant>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.prompt_variants_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{adapter_name}/{prompt_key}/{variant_id}");
        match kv
            .get(&key)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?
        {
            Some(bytes) => {
                let variant = serde_json::from_slice(&bytes)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(variant))
            }
            None => Ok(None),
        }
    }

    /// Get the active variant ID pointer for an `adapter+prompt_key`.
    /// Key: `{adapter_name}/{prompt_key}/_active`.
    pub async fn get_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
    ) -> Result<Option<String>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.prompt_variants_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{adapter_name}/{prompt_key}/_active");
        Ok(kv
            .get(&key)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string()))
    }

    /// Set the active variant ID pointer for an `adapter+prompt_key`.
    /// Key: `{adapter_name}/{prompt_key}/_active`.
    pub async fn set_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.prompt_variants_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{adapter_name}/{prompt_key}/_active");
        kv.put(&key, variant_id.as_bytes().to_vec().into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Get the per-adapter OPRO state.
    /// Key: `{adapter_name}/_opro_state`.
    pub async fn get_adapter_opro_state(
        &self,
        adapter_name: &str,
    ) -> Result<Option<AdapterOproState>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.prompt_variants_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{adapter_name}/_opro_state");
        match kv
            .get(&key)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?
        {
            Some(bytes) => {
                let state = serde_json::from_slice(&bytes)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(state))
            }
            None => Ok(None),
        }
    }

    /// Persist the per-adapter OPRO state.
    /// Key: `{adapter_name}/_opro_state`.
    pub async fn put_adapter_opro_state(&self, state: &AdapterOproState) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.prompt_variants_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}/_opro_state", state.adapter_name);
        let bytes = serde_json::to_vec(state).map_err(|e| NatsError::Serialize(e.to_string()))?;
        kv.put(&key, bytes.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    // ── internal ────────────────────────────────────────────────────────────

    /// Delete a KV bucket. Returns `Ok(())` if already absent.
    pub async fn delete_kv_bucket(&self, bucket: &str) -> Result<(), NatsError> {
        match self.jetstream.delete_key_value(bucket).await {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("not found") || msg.contains("stream not found") {
                    Ok(())
                } else {
                    Err(NatsError::KvError(msg))
                }
            }
        }
    }

    async fn ensure_kv_bucket(&self, config: kv::Config) -> Result<(), NatsError> {
        match self.jetstream.get_key_value(&config.bucket).await {
            Ok(_) => Ok(()),
            Err(get_err) => {
                tracing::debug!(
                    bucket = %config.bucket,
                    error = %get_err,
                    "KV bucket not found; attempting to create"
                );
                self.jetstream
                    .create_key_value(config)
                    .await
                    .map(|_| ())
                    .map_err(|e| NatsError::KvError(e.to_string()))
            }
        }
    }

    async fn ensure_object_store(
        &self,
        config: async_nats::jetstream::object_store::Config,
    ) -> Result<(), NatsError> {
        match self.jetstream.get_object_store(&config.bucket).await {
            Ok(_) => Ok(()),
            Err(get_err) => {
                tracing::debug!(
                    bucket = %config.bucket,
                    error = %get_err,
                    "Object Store bucket not found; attempting to create"
                );
                self.jetstream
                    .create_object_store(config)
                    .await
                    .map(|_| ())
                    .map_err(|e| NatsError::StreamError(e.to_string()))
            }
        }
    }

    /// Write a task checkpoint to NATS KV with zstd compression.
    ///
    /// Pass `expected_revision = None` for first write (unconditional).
    /// Pass `Some(rev)` to use optimistic concurrency — fails if another node
    /// has written since `rev`. Returns the new revision on success.
    pub async fn put_task_checkpoint(
        &self,
        checkpoint: &TaskCheckpoint,
        expected_revision: Option<u64>,
    ) -> Result<u64, NatsError> {
        let json =
            serde_json::to_vec(checkpoint).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let compressed = zstd::encode_all(json.as_slice(), 3)
            .map_err(|e| NatsError::Serialize(format!("zstd encode: {e}")))?;
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.task_checkpoints_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let revision = match expected_revision {
            None => kv
                .put(&checkpoint.task_id, compressed.into())
                .await
                .map_err(|e| NatsError::KvError(e.to_string()))?,
            Some(rev) => kv
                .update(&checkpoint.task_id, compressed.into(), rev)
                .await
                .map_err(|e| NatsError::KvError(e.to_string()))?,
        };
        Ok(revision)
    }

    /// Load and decompress a checkpoint by `task_id`. Returns `None` if not found.
    pub async fn get_task_checkpoint(
        &self,
        task_id: &str,
    ) -> Result<Option<TaskCheckpoint>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.task_checkpoints_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get(task_id).await {
            Ok(Some(bytes)) => {
                let decompressed = zstd::decode_all(bytes.as_ref())
                    .map_err(|e| NatsError::Serialize(format!("zstd decode: {e}")))?;
                let checkpoint = serde_json::from_slice::<TaskCheckpoint>(&decompressed)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(checkpoint))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Load all in-flight checkpoints from the bucket.
    /// Entries that fail to decompress or deserialize are skipped with a warning.
    pub async fn list_task_checkpoints(&self) -> Vec<TaskCheckpoint> {
        use futures::StreamExt;

        let kv = match self
            .jetstream
            .get_key_value(&self.state_cfg.task_checkpoints_bucket)
            .await
        {
            Ok(kv) => kv,
            Err(e) => {
                tracing::warn!("list_task_checkpoints: KV open failed: {e}");
                return vec![];
            }
        };
        let keys: Vec<String> = match kv.keys().await {
            Ok(stream) => stream.filter_map(|r| async move { r.ok() }).collect().await,
            Err(e) => {
                tracing::warn!("list_task_checkpoints: keys() failed: {e}");
                return vec![];
            }
        };
        let mut result = Vec::with_capacity(keys.len());
        for key in &keys {
            match self.get_task_checkpoint(key).await {
                Ok(Some(c)) => result.push(c),
                Ok(None) => {}
                Err(e) => tracing::warn!(task_id = %key, "corrupt checkpoint skipped: {e}"),
            }
        }
        result
    }

    /// Delete a checkpoint, first cleaning up any Object Store overflow object.
    ///
    /// Always call this method instead of deleting the KV entry directly —
    /// it prevents orphaned blobs in `H2AI_CHECKPOINT_PAYLOADS`.
    pub async fn delete_task_checkpoint(&self, task_id: &str) -> Result<(), NatsError> {
        // 1. Load to check for an Object Store reference
        if let Some(checkpoint) = self.get_task_checkpoint(task_id).await? {
            if let Some(obj_ref) = &checkpoint.object_store_ref {
                match self
                    .jetstream
                    .get_object_store(&self.state_cfg.checkpoint_payloads_bucket)
                    .await
                {
                    Ok(store) => {
                        if let Err(e) = store.delete(obj_ref).await {
                            tracing::warn!(
                                obj_ref = %obj_ref,
                                "failed to delete checkpoint object — storage may leak: {e}"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(bucket = %self.state_cfg.checkpoint_payloads_bucket, "failed to open checkpoint payloads object store: {e}");
                    }
                }
            }
        }
        // 2. Delete the KV entry
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.task_checkpoints_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.delete(task_id)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    // ── JetStream signal delivery ────────────────────────────────────────────────

    /// Provision the `H2AI_SIGNALS` `JetStream` stream (idempotent).
    pub async fn provision_signals_stream(&self) -> Result<(), NatsError> {
        let cfg = stream::Config {
            name: self.state_cfg.signals_stream.clone(),
            subjects: vec![format!("{}.>", self.state_cfg.signals_subject_prefix)],
            retention: stream::RetentionPolicy::Limits,
            max_age: std::time::Duration::from_hours(24),
            storage: stream::StorageType::File,
            num_replicas: 1,
            ..Default::default()
        };
        self.jetstream
            .get_or_create_stream(cfg)
            .await
            .map(|_| ())
            .map_err(|e| NatsError::KvError(e.to_string()))
    }

    /// Publish a `ResumeSignal` to the signals stream.
    ///
    /// Subject: `h2ai.signals.{tenant_bucket_safe}.{task_id}`
    pub async fn publish_signal(
        &self,
        signal: &h2ai_types::signal::ResumeSignal,
    ) -> Result<(), NatsError> {
        let subject = format!(
            "{}.{}.{}",
            self.state_cfg.signals_subject_prefix,
            signal.tenant_id.bucket_safe(),
            signal.task_id,
        );
        let payload =
            serde_json::to_vec(signal).map_err(|e| NatsError::Serialize(e.to_string()))?;
        self.jetstream
            .publish(subject, payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Create a pull consumer for the given task and return a stream of `ResumeSignal` items.
    ///
    /// The consumer is durable (`SIGNAL-{task_id_no_dashes}`) and filters to the exact task subject.
    /// Call `delete_signal_consumer` when done to clean up.
    pub async fn subscribe_signals(
        &self,
        task_id: &h2ai_types::identity::TaskId,
        tenant_id: &h2ai_types::identity::TenantId,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures::Stream<Item = Result<h2ai_types::signal::ResumeSignal, NatsError>>
                    + Send,
            >,
        >,
        NatsError,
    > {
        use async_nats::jetstream::consumer::pull;
        use async_nats::jetstream::consumer::{AckPolicy, DeliverPolicy};
        use futures::StreamExt;

        let consumer_name = format!("SIGNAL-{}", task_id.to_string().replace('-', ""));
        let filter = format!(
            "{}.{}.{}",
            self.state_cfg.signals_subject_prefix,
            tenant_id.bucket_safe(),
            task_id
        );

        let stream = self
            .jetstream
            .get_stream(&self.state_cfg.signals_stream)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;

        // Use a pull consumer: the client fetches messages on demand rather than
        // relying on server-side push delivery to a _INBOX subject.  Durable push
        // consumers with _INBOX deliver_subjects become stale across server restarts
        // and reconnects, causing the engine's select! to never receive signals.
        let pull_cfg = pull::Config {
            name: Some(consumer_name.clone()),
            durable_name: Some(consumer_name.clone()),
            filter_subject: filter,
            deliver_policy: DeliverPolicy::All,
            ack_policy: AckPolicy::Explicit,
            ..Default::default()
        };
        let consumer = if let Ok(c) = stream.create_consumer(pull_cfg.clone()).await {
            c
        } else {
            // A stale push consumer with the same durable name may exist from a
            // previous run.  Delete it and retry once.
            let _ = stream.delete_consumer(&consumer_name).await;
            stream
                .create_consumer(pull_cfg)
                .await
                .map_err(|e| NatsError::KvError(e.to_string()))?
        };

        let messages = consumer
            .messages()
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;

        let mapped = messages.then(|msg_res| async move {
            let msg = msg_res.map_err(|e| NatsError::KvError(e.to_string()))?;
            let signal = serde_json::from_slice::<h2ai_types::signal::ResumeSignal>(&msg.payload)
                .map_err(|e| NatsError::Serialize(e.to_string()))?;
            msg.ack()
                .await
                .map_err(|e| NatsError::KvError(e.to_string()))?;
            Ok(signal)
        });

        Ok(Box::pin(mapped))
    }

    /// Delete the pull consumer created by `subscribe_signals` for a given task.
    pub async fn delete_signal_consumer(
        &self,
        task_id: &h2ai_types::identity::TaskId,
    ) -> Result<(), NatsError> {
        let consumer_name = format!("SIGNAL-{}", task_id.to_string().replace('-', ""));
        let stream = self
            .jetstream
            .get_stream(&self.state_cfg.signals_stream)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        stream
            .delete_consumer(&consumer_name)
            .await
            .map(|_| ())
            .map_err(|e| NatsError::KvError(e.to_string()))
    }

    // ── delta checkpoint write/read path ─────────────────────────────────────

    /// Update the `{task_id}/seq/latest` pointer in the `task_checkpoints` bucket using
    /// optimistic CAS (up to 3 attempts). Value is the seq number as little-endian u32 bytes.
    async fn update_latest_seq(&self, task_id: &str, seq: u32) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.task_checkpoints_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{task_id}/seq/latest");
        let seq_bytes: Vec<u8> = seq.to_le_bytes().to_vec();

        for attempt in 0..3u32 {
            if let Some(entry) = kv
                .entry(&key)
                .await
                .map_err(|e| NatsError::KvError(e.to_string()))?
            {
                let revision = entry.revision;
                match kv.update(&key, seq_bytes.clone().into(), revision).await {
                    Ok(_) => return Ok(()),
                    Err(_) if attempt < 2 => continue,
                    Err(e) => {
                        return Err(NatsError::KvError(format!(
                            "update_latest_seq CAS failed after 3 attempts: {e}"
                        )))
                    }
                }
            }
            kv.put(&key, seq_bytes.clone().into())
                .await
                .map_err(|e| NatsError::KvError(e.to_string()))?;
            return Ok(());
        }
        Err(NatsError::KvError(
            "update_latest_seq: max CAS retries exceeded".into(),
        ))
    }

    /// Persist a checkpoint using delta encoding.
    ///
    /// When `delta.enabled = false` or `seq` falls on a base interval, stores a full
    /// `CheckpointKind::Base`. Otherwise computes an RFC-6902 patch against the base
    /// checkpoint and stores a `CheckpointKind::Delta`.
    ///
    /// After the write, updates the `{task_id}/seq/latest` CAS pointer and invalidates
    /// the in-memory LRU cache for the task.
    pub async fn put_checkpoint_delta(
        &self,
        task_id: &str,
        checkpoint: &TaskCheckpoint,
        seq: u32,
    ) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.task_checkpoints_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;

        // Parse task_id as UUID; fall back to new UUID for legacy string task ids.
        let task_uuid = uuid::Uuid::parse_str(task_id).unwrap_or_else(|_| uuid::Uuid::new_v4());
        let typed_task_id = TaskId::from_uuid(task_uuid);

        let entry = if !self.state_cfg.delta.enabled
            || should_store_base(seq, self.state_cfg.delta.base_interval)
        {
            // Full base checkpoint
            TaskCheckpointEntry {
                task_id: typed_task_id,
                seq,
                base_seq: seq,
                kind: CheckpointKind::Base(Box::new(checkpoint.clone())),
                timestamp: chrono::Utc::now(),
            }
        } else {
            // Delta against the nearest base
            let base_seq =
                (seq / self.state_cfg.delta.base_interval) * self.state_cfg.delta.base_interval;
            let base_key = format!("{task_id}/seq/{base_seq:08}");
            let base_bytes = kv
                .get(&base_key)
                .await
                .map_err(|e| NatsError::KvError(e.to_string()))?
                .ok_or_else(|| {
                    NatsError::KvError(format!(
                        "base checkpoint not found for task={task_id} base_seq={base_seq}"
                    ))
                })?;
            let base_entry: TaskCheckpointEntry = serde_json::from_slice(&base_bytes)
                .map_err(|e| NatsError::Serialize(e.to_string()))?;
            let base_cp = match &base_entry.kind {
                CheckpointKind::Base(cp) => (*cp).clone(),
                CheckpointKind::Delta(_) => {
                    return Err(NatsError::KvError(format!(
                        "base_seq={base_seq} entry is a Delta, expected Base"
                    )))
                }
            };
            let patch = generate_delta(&base_cp, checkpoint)?;
            TaskCheckpointEntry {
                task_id: typed_task_id,
                seq,
                base_seq,
                kind: CheckpointKind::Delta(patch.0),
                timestamp: chrono::Utc::now(),
            }
        };

        let key = format!("{task_id}/seq/{seq:08}");
        let bytes = serde_json::to_vec(&entry).map_err(|e| NatsError::Serialize(e.to_string()))?;
        kv.put(&key, bytes.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;

        // Update the latest seq pointer
        self.update_latest_seq(task_id, seq).await?;

        // Invalidate cache
        self.delta_cache.write().await.pop(task_id);

        Ok(())
    }

    /// Return the most recent checkpoint for `task_id`, using the LRU cache when warm.
    ///
    /// Read path:
    /// 1. Check LRU cache (TTL-gated).
    /// 2. Read `{task_id}/seq/latest` to get the highest written seq.
    /// 3. If no delta entry exists, fall back to `get_task_checkpoint` (legacy flat key).
    /// 4. Reconstruct the checkpoint at that seq (apply patch if Delta).
    /// 5. Populate the cache.
    pub async fn get_latest_checkpoint(
        &self,
        task_id: &str,
    ) -> Result<Option<TaskCheckpoint>, NatsError> {
        // Cache lookup (write lock so LRU order is updated on hit)
        {
            let mut cache = self.delta_cache.write().await;
            if let Some(cached) = cache.get(task_id) {
                let ttl = std::time::Duration::from_secs(self.state_cfg.delta.cache_ttl_secs);
                if cached.cached_at.elapsed() < ttl {
                    return Ok(Some(cached.checkpoint.clone()));
                }
                // TTL expired
                cache.pop(task_id);
            }
        }

        let kv = self
            .jetstream
            .get_key_value(&self.state_cfg.task_checkpoints_bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;

        let latest_key = format!("{task_id}/seq/latest");
        let seq = match kv
            .get(&latest_key)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?
        {
            Some(bytes) => {
                let arr: [u8; 4] = bytes.as_ref().try_into().map_err(|_| {
                    NatsError::Serialize(format!(
                        "latest seq key has unexpected byte length for task={task_id}"
                    ))
                })?;
                u32::from_le_bytes(arr)
            }
            None => {
                // No delta entries — try legacy flat key
                return self.get_legacy_checkpoint(task_id).await;
            }
        };

        self.reconstruct_at_seq(task_id, seq, &kv).await
    }

    /// Reconstruct (and cache) the checkpoint at a specific seq.
    async fn reconstruct_at_seq(
        &self,
        task_id: &str,
        seq: u32,
        kv: &async_nats::jetstream::kv::Store,
    ) -> Result<Option<TaskCheckpoint>, NatsError> {
        let key = format!("{task_id}/seq/{seq:08}");
        let Some(bytes) = kv
            .get(&key)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?
        else {
            return Ok(None);
        };
        let entry: TaskCheckpointEntry =
            serde_json::from_slice(&bytes).map_err(|e| NatsError::Serialize(e.to_string()))?;

        let checkpoint = match entry.kind {
            CheckpointKind::Base(cp) => *cp,
            CheckpointKind::Delta(ops) => {
                // Fetch the base
                let base_key = format!("{task_id}/seq/{:08}", entry.base_seq);
                let base_bytes = kv
                    .get(&base_key)
                    .await
                    .map_err(|e| NatsError::KvError(e.to_string()))?
                    .ok_or_else(|| {
                        NatsError::KvError(format!(
                            "base checkpoint missing for task={task_id} base_seq={}",
                            entry.base_seq
                        ))
                    })?;
                let base_entry: TaskCheckpointEntry = serde_json::from_slice(&base_bytes)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                let base_cp = match base_entry.kind {
                    CheckpointKind::Base(cp) => *cp,
                    CheckpointKind::Delta(_) => {
                        return Err(NatsError::KvError(format!(
                        "base_seq={} entry is itself a Delta — corrupt chain for task={task_id}",
                        entry.base_seq
                    )))
                    }
                };
                let patch = json_patch::Patch(ops);
                apply_patches(&base_cp, &[patch])?
            }
        };

        // Populate cache
        self.delta_cache.write().await.put(
            task_id.to_string(),
            CachedCheckpoint {
                checkpoint: checkpoint.clone(),
                seq,
                cached_at: std::time::Instant::now(),
            },
        );

        Ok(Some(checkpoint))
    }

    /// Backward-compatibility fallback: fetch via the old flat-key format (`task_id` directly),
    /// which is what `get_task_checkpoint` uses (zstd-compressed JSON).
    async fn get_legacy_checkpoint(
        &self,
        task_id: &str,
    ) -> Result<Option<TaskCheckpoint>, NatsError> {
        self.get_task_checkpoint(task_id).await
    }

    // ── per-tenant reasoning memory ─────────────────────────────────────────

    /// Create per-tenant reasoning checkpoint and meta-state KV buckets if they
    /// do not already exist. Safe to call multiple times (`get_or_create` semantics).
    pub async fn ensure_tenant_reasoning_buckets(
        &self,
        tenant_id: &TenantId,
        checkpoint_prefix: &str,
        meta_state_prefix: &str,
    ) -> Result<(), NatsError> {
        let checkpoint_bucket = tenant_bucket_name(checkpoint_prefix, tenant_id);
        let meta_bucket = tenant_bucket_name(meta_state_prefix, tenant_id);

        self.ensure_kv_bucket(kv::Config {
            bucket: checkpoint_bucket,
            description: format!("Reasoning checkpoints for tenant {tenant_id}"),
            history: 1,
            storage: stream::StorageType::File,
            max_age: std::time::Duration::from_hours(168),
            ..Default::default()
        })
        .await?;

        self.ensure_kv_bucket(kv::Config {
            bucket: meta_bucket,
            description: format!("Task meta-state records for tenant {tenant_id}"),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        Ok(())
    }

    /// Write (or overwrite) a `TaskReasoningCheckpoint` to the tenant-scoped bucket.
    /// Key: `task_id` string. Compressed with zstd level 3.
    pub async fn put_reasoning_checkpoint(
        &self,
        checkpoint: &TaskReasoningCheckpoint,
        checkpoint_prefix: &str,
    ) -> Result<(), NatsError> {
        let bucket = tenant_bucket_name(checkpoint_prefix, &checkpoint.tenant_id);
        let json =
            serde_json::to_vec(checkpoint).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let compressed = zstd::encode_all(json.as_slice(), 3)
            .map_err(|e| NatsError::Serialize(format!("zstd encode: {e}")))?;
        let kv = self
            .jetstream
            .get_key_value(&bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put(&checkpoint.task_id.to_string(), compressed.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Load a `TaskReasoningCheckpoint` by `task_id`. Returns `None` if not found.
    pub async fn get_reasoning_checkpoint(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
        checkpoint_prefix: &str,
    ) -> Result<Option<TaskReasoningCheckpoint>, NatsError> {
        let bucket = tenant_bucket_name(checkpoint_prefix, tenant_id);
        let kv = self
            .jetstream
            .get_key_value(&bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get(&task_id.to_string()).await {
            Ok(Some(bytes)) => {
                let decompressed = zstd::decode_all(bytes.as_ref())
                    .map_err(|e| NatsError::Serialize(format!("zstd decode: {e}")))?;
                let cp = serde_json::from_slice::<TaskReasoningCheckpoint>(&decompressed)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(cp))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Write an immutable `TaskMetaState` projection to the tenant-scoped meta-state bucket.
    /// Key: `task_id` string. Not compressed (small records, queried frequently).
    pub async fn put_task_meta_state(
        &self,
        meta: &TaskMetaState,
        meta_state_prefix: &str,
    ) -> Result<(), NatsError> {
        let bucket = tenant_bucket_name(meta_state_prefix, &meta.tenant_id);
        let json = serde_json::to_vec(meta).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value(&bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put(&meta.task_id.to_string(), json.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Load a `TaskMetaState` by `task_id`. Returns `None` if not found.
    pub async fn get_task_meta_state(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
        meta_state_prefix: &str,
    ) -> Result<Option<TaskMetaState>, NatsError> {
        let bucket = tenant_bucket_name(meta_state_prefix, tenant_id);
        let kv = self
            .jetstream
            .get_key_value(&bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get(&task_id.to_string()).await {
            Ok(Some(bytes)) => {
                let meta = serde_json::from_slice::<TaskMetaState>(&bytes)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(meta))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// List up to `limit` `TaskMetaState` records for a tenant.
    /// Entries that fail to deserialize are skipped with a warning.
    pub async fn list_task_meta_states(
        &self,
        tenant_id: &TenantId,
        meta_state_prefix: &str,
        limit: usize,
    ) -> Vec<TaskMetaState> {
        use futures::StreamExt;

        let bucket = tenant_bucket_name(meta_state_prefix, tenant_id);
        let kv = match self.jetstream.get_key_value(&bucket).await {
            Ok(kv) => kv,
            Err(e) => {
                tracing::warn!("list_task_meta_states: KV open failed: {e}");
                return vec![];
            }
        };
        let keys: Vec<String> = match kv.keys().await {
            Ok(stream) => stream.filter_map(|r| async move { r.ok() }).collect().await,
            Err(e) => {
                tracing::warn!("list_task_meta_states: keys() failed: {e}");
                return vec![];
            }
        };
        let mut result = Vec::with_capacity(keys.len().min(limit));
        for key in keys.iter().take(limit) {
            match kv.get(key).await {
                Ok(Some(bytes)) => match serde_json::from_slice::<TaskMetaState>(&bytes) {
                    Ok(meta) => result.push(meta),
                    Err(e) => {
                        tracing::warn!("list_task_meta_states: deserialize failed for {key}: {e}");
                    }
                },
                Ok(None) => {}
                Err(e) => tracing::warn!("list_task_meta_states: get failed for {key}: {e}"),
            }
        }
        result
    }

    // ── per-tenant conflict-rate accumulator ────────────────────────────────

    const CONFLICT_ACCUMULATOR_KEY: &str = "accumulator";

    /// Create the per-tenant conflict-rate bucket if it does not already exist.
    pub async fn ensure_tenant_conflict_bucket(
        &self,
        tenant_id: &TenantId,
        bucket_prefix: &str,
    ) -> Result<(), NatsError> {
        let bucket = tenant_bucket_name(bucket_prefix, tenant_id);
        self.ensure_kv_bucket(kv::Config {
            bucket,
            description: format!("Conflict-rate accumulator for tenant {tenant_id}"),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await
    }

    /// Load the `ConflictRateAccumulator` for a tenant. Returns `None` when no record exists yet.
    pub async fn get_conflict_accumulator(
        &self,
        tenant_id: &TenantId,
        bucket_prefix: &str,
    ) -> Result<Option<ConflictRateAccumulator>, NatsError> {
        let bucket = tenant_bucket_name(bucket_prefix, tenant_id);
        let kv = self
            .jetstream
            .get_key_value(&bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get(Self::CONFLICT_ACCUMULATOR_KEY).await {
            Ok(Some(bytes)) => {
                let acc = serde_json::from_slice::<ConflictRateAccumulator>(&bytes)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(acc))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Write (or overwrite) a `ConflictRateAccumulator` to the tenant-scoped bucket.
    pub async fn put_conflict_accumulator(
        &self,
        acc: &ConflictRateAccumulator,
        bucket_prefix: &str,
    ) -> Result<(), NatsError> {
        let bucket = tenant_bucket_name(bucket_prefix, &acc.tenant_id);
        let json = serde_json::to_vec(acc).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value(&bucket)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put(Self::CONFLICT_ACCUMULATOR_KEY, json.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }
}

// ── tenant-scoped bucket helpers ────────────────────────────────────────────

#[must_use]
pub fn tenant_bucket_name(prefix: &str, tenant_id: &TenantId) -> String {
    format!("{}_{}", prefix, tenant_id.bucket_safe())
}

// ── delta encoding helpers ──────────────────────────────────────────────────

/// Returns `true` when `seq` should be stored as a full Base checkpoint.
///
/// Sequence 0 is always a base. Thereafter, every `base_interval`-th checkpoint
/// is stored as a base so that patch chains never grow unbounded.
#[must_use]
pub const fn should_store_base(seq: u32, base_interval: u32) -> bool {
    seq == 0 || seq.is_multiple_of(base_interval)
}

/// Compute the RFC-6902 JSON Patch diff between `base` and `current`.
///
/// The returned `Patch` is empty (zero operations) when the two checkpoints
/// are identical. Callers store this alongside the current `seq` so a
/// reader can reconstruct `current` by applying the patch to the base.
pub fn generate_delta(
    base: &TaskCheckpoint,
    current: &TaskCheckpoint,
) -> Result<json_patch::Patch, NatsError> {
    let base_val = serde_json::to_value(base).map_err(|e| NatsError::Serialize(e.to_string()))?;
    let current_val =
        serde_json::to_value(current).map_err(|e| NatsError::Serialize(e.to_string()))?;
    Ok(json_patch::diff(&base_val, &current_val))
}

/// Reconstruct a `TaskCheckpoint` by applying a sequence of patches to `base`.
///
/// Patches are applied in order. Typically called with a single-element slice
/// (base → current diff), but the signature accepts multiple patches so a
/// reader can fast-forward across several delta checkpoints in one call.
pub fn apply_patches(
    base: &TaskCheckpoint,
    patches: &[json_patch::Patch],
) -> Result<TaskCheckpoint, NatsError> {
    let mut val = serde_json::to_value(base).map_err(|e| NatsError::Serialize(e.to_string()))?;
    for patch in patches {
        json_patch::patch(&mut val, &patch.0)
            .map_err(|e| NatsError::Serialize(format!("json-patch apply: {e}")))?;
    }
    serde_json::from_value(val).map_err(|e| NatsError::Serialize(e.to_string()))
}

// ── Trait impls delegating to existing NatsClient methods ───────────────────

#[async_trait::async_trait]
impl crate::backend::EventPublisher for NatsClient {
    async fn publish_event(&self, task_id: &TaskId, event: &H2AIEvent) -> Result<(), NatsError> {
        self.publish_event(task_id, event).await
    }
    async fn publish_to(&self, subject: &str, event: &H2AIEvent) -> Result<(), NatsError> {
        self.publish_to(subject, event).await
    }
    async fn publish_event_seq(
        &self,
        task_id: &TaskId,
        event: &H2AIEvent,
    ) -> Result<u64, NatsError> {
        self.publish_event_seq(task_id, event).await
    }
}

#[async_trait::async_trait]
impl crate::backend::SnapshotStore for NatsClient {
    async fn put_snapshot(&self, snap: &TaskSnapshot) -> Result<(), NatsError> {
        self.put_snapshot(snap).await
    }
    async fn get_snapshot(&self, task_id: &TaskId) -> Result<Option<TaskSnapshot>, NatsError> {
        self.get_snapshot(task_id).await
    }
}

#[async_trait::async_trait]
impl crate::backend::CalibrationStore for NatsClient {
    async fn put_calibration(&self, cal: &CalibrationCompletedEvent) -> Result<(), NatsError> {
        self.put_calibration(cal).await
    }
    async fn get_calibration(&self) -> Result<Option<CalibrationCompletedEvent>, NatsError> {
        self.get_calibration().await
    }
    async fn get_calibration_record(
        &self,
        adapter_profile: &str,
    ) -> Result<Option<CalibrationRecord>, NatsError> {
        self.get_calibration_record(adapter_profile).await
    }
    async fn put_calibration_record(&self, record: &CalibrationRecord) -> Result<(), NatsError> {
        self.put_calibration_record(record).await
    }
}

#[async_trait::async_trait]
impl crate::backend::SignalPublisher for NatsClient {
    async fn publish_signal(
        &self,
        signal: &h2ai_types::signal::ResumeSignal,
    ) -> Result<(), NatsError> {
        self.publish_signal(signal).await
    }
}

#[async_trait::async_trait]
impl crate::backend::TailEvents for NatsClient {
    async fn tail_task_events_boxed(
        &self,
        task_id: &TaskId,
        from_seq: u64,
    ) -> Result<futures::stream::BoxStream<'static, Result<(u64, H2AIEvent), NatsError>>, NatsError>
    {
        self.tail_task_events(task_id, from_seq)
            .await
            .map(futures::StreamExt::boxed)
    }
}

#[async_trait::async_trait]
impl crate::backend::OproStore for NatsClient {
    async fn put_prompt_variant(&self, variant: &PromptVariant) -> Result<(), NatsError> {
        self.put_prompt_variant(variant).await
    }
    async fn get_prompt_variant(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<Option<PromptVariant>, NatsError> {
        self.get_prompt_variant(adapter_name, prompt_key, variant_id)
            .await
    }
    async fn get_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
    ) -> Result<Option<String>, NatsError> {
        self.get_active_variant_ptr(adapter_name, prompt_key).await
    }
    async fn set_active_variant_ptr(
        &self,
        adapter_name: &str,
        prompt_key: &str,
        variant_id: &str,
    ) -> Result<(), NatsError> {
        self.set_active_variant_ptr(adapter_name, prompt_key, variant_id)
            .await
    }
    async fn get_adapter_opro_state(
        &self,
        adapter_name: &str,
    ) -> Result<Option<AdapterOproState>, NatsError> {
        self.get_adapter_opro_state(adapter_name).await
    }
    async fn put_adapter_opro_state(&self, state: &AdapterOproState) -> Result<(), NatsError> {
        self.put_adapter_opro_state(state).await
    }
}

#[async_trait::async_trait]
impl crate::backend::EstimatorStore for NatsClient {
    async fn get_tao_estimator_state(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Option<(f64, usize)>, NatsError> {
        self.get_tao_estimator_state(tenant_id).await
    }
    async fn put_tao_estimator_state(
        &self,
        tenant_id: &TenantId,
        ema: f64,
        count: usize,
    ) -> Result<(), NatsError> {
        self.put_tao_estimator_state(tenant_id, ema, count).await
    }
    async fn get_srani_state(
        &self,
        tenant_id: &TenantId,
    ) -> Result<Option<(f64, usize)>, NatsError> {
        self.get_srani_state(tenant_id).await
    }
    async fn put_srani_state(
        &self,
        tenant_id: &TenantId,
        ema_cfi: f64,
        count: usize,
    ) -> Result<(), NatsError> {
        self.put_srani_state(tenant_id, ema_cfi, count).await
    }
    async fn get_bandit_state(&self, tenant_id: &TenantId) -> Result<Option<Vec<u8>>, NatsError> {
        self.get_bandit_state(tenant_id).await
    }
    async fn put_bandit_state(
        &self,
        tenant_id: &TenantId,
        json_bytes: Vec<u8>,
    ) -> Result<(), NatsError> {
        self.put_bandit_state(tenant_id, json_bytes).await
    }
}

#[async_trait::async_trait]
impl crate::backend::ReasoningStore for NatsClient {
    async fn ensure_reasoning_buckets(
        &self,
        tenant_id: &TenantId,
        checkpoint_prefix: &str,
        meta_state_prefix: &str,
    ) -> Result<(), NatsError> {
        self.ensure_tenant_reasoning_buckets(tenant_id, checkpoint_prefix, meta_state_prefix)
            .await
    }

    async fn put_reasoning_checkpoint(
        &self,
        checkpoint: &TaskReasoningCheckpoint,
        checkpoint_prefix: &str,
    ) -> Result<(), NatsError> {
        self.put_reasoning_checkpoint(checkpoint, checkpoint_prefix)
            .await
    }

    async fn get_reasoning_checkpoint(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
        checkpoint_prefix: &str,
    ) -> Result<Option<TaskReasoningCheckpoint>, NatsError> {
        self.get_reasoning_checkpoint(task_id, tenant_id, checkpoint_prefix)
            .await
    }

    async fn put_task_meta_state(
        &self,
        meta: &TaskMetaState,
        meta_state_prefix: &str,
    ) -> Result<(), NatsError> {
        self.put_task_meta_state(meta, meta_state_prefix).await
    }

    async fn get_task_meta_state(
        &self,
        task_id: &TaskId,
        tenant_id: &TenantId,
        meta_state_prefix: &str,
    ) -> Result<Option<TaskMetaState>, NatsError> {
        self.get_task_meta_state(task_id, tenant_id, meta_state_prefix)
            .await
    }

    async fn list_task_meta_states(
        &self,
        tenant_id: &TenantId,
        meta_state_prefix: &str,
        limit: usize,
    ) -> Vec<TaskMetaState> {
        self.list_task_meta_states(tenant_id, meta_state_prefix, limit)
            .await
    }
}

#[async_trait::async_trait]
impl crate::backend::ConflictStore for NatsClient {
    async fn ensure_conflict_bucket(
        &self,
        tenant_id: &TenantId,
        bucket_prefix: &str,
    ) -> Result<(), NatsError> {
        self.ensure_tenant_conflict_bucket(tenant_id, bucket_prefix)
            .await
    }

    async fn get_conflict_accumulator(
        &self,
        tenant_id: &TenantId,
        bucket_prefix: &str,
    ) -> Result<Option<ConflictRateAccumulator>, NatsError> {
        NatsClient::get_conflict_accumulator(self, tenant_id, bucket_prefix).await
    }

    async fn put_conflict_accumulator(
        &self,
        acc: &ConflictRateAccumulator,
        bucket_prefix: &str,
    ) -> Result<(), NatsError> {
        NatsClient::put_conflict_accumulator(self, acc, bucket_prefix).await
    }
}

#[async_trait::async_trait]
impl crate::backend::SignalSubscriber for NatsClient {
    async fn subscribe_signals(
        &self,
        task_id: &h2ai_types::identity::TaskId,
        tenant_id: &h2ai_types::identity::TenantId,
    ) -> Result<
        futures::stream::BoxStream<'static, Result<h2ai_types::signal::ResumeSignal, NatsError>>,
        NatsError,
    > {
        NatsClient::subscribe_signals(self, task_id, tenant_id).await
    }

    async fn delete_signal_consumer(
        &self,
        task_id: &h2ai_types::identity::TaskId,
    ) -> Result<(), NatsError> {
        NatsClient::delete_signal_consumer(self, task_id).await
    }
}

#[async_trait::async_trait]
impl crate::backend::ShadowDomainStore for NatsClient {
    async fn put_shadow_promoted_domains(
        &self,
        domains: &std::collections::HashSet<String>,
    ) -> Result<(), NatsError> {
        NatsClient::put_shadow_promoted_domains(self, domains).await
    }

    async fn get_shadow_promoted_domains(
        &self,
    ) -> Result<std::collections::HashSet<String>, NatsError> {
        NatsClient::get_shadow_promoted_domains(self).await
    }
}

#[async_trait::async_trait]
impl crate::backend::TaskCheckpointStore for NatsClient {
    async fn list_task_checkpoints(&self) -> Vec<h2ai_types::checkpoint::TaskCheckpoint> {
        NatsClient::list_task_checkpoints(self).await
    }

    async fn put_task_checkpoint(
        &self,
        cp: &h2ai_types::checkpoint::TaskCheckpoint,
        expected_revision: Option<u64>,
    ) -> Result<u64, NatsError> {
        NatsClient::put_task_checkpoint(self, cp, expected_revision).await
    }

    async fn get_task_checkpoint(
        &self,
        task_id: &str,
    ) -> Result<Option<h2ai_types::checkpoint::TaskCheckpoint>, NatsError> {
        NatsClient::get_task_checkpoint(self, task_id).await
    }

    async fn delete_task_checkpoint(&self, task_id: &str) -> Result<(), NatsError> {
        NatsClient::delete_task_checkpoint(self, task_id).await
    }
}

#[async_trait::async_trait]
impl crate::backend::SkillStore for NatsClient {
    async fn put_skill_nodes(
        &self,
        tenant_id: &TenantId,
        json_bytes: Vec<u8>,
    ) -> Result<(), NatsError> {
        NatsClient::put_skill_nodes(self, tenant_id, json_bytes).await
    }

    async fn get_skill_nodes(&self, tenant_id: &TenantId) -> Result<Vec<u8>, NatsError> {
        NatsClient::get_skill_nodes(self, tenant_id).await
    }
}

#[async_trait::async_trait]
impl crate::backend::TaskDispatchBackend for NatsClient {
    async fn publish_task_payload(
        &self,
        payload: &h2ai_types::agent::TaskPayload,
    ) -> Result<(), NatsError> {
        NatsClient::publish_task_payload(self, payload).await
    }

    async fn await_task_result_once(
        &self,
        task_id: &h2ai_types::identity::TaskId,
        timeout: std::time::Duration,
    ) -> Result<h2ai_types::agent::TaskResult, NatsError> {
        NatsClient::await_task_result_once(self, task_id, timeout).await
    }
}
