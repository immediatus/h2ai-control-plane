use async_nats::jetstream::{self, kv, stream};
use async_nats::Client;
use h2ai_types::checkpoint::TaskCheckpoint;
use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent, TaskSnapshot};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::OracleObservation;
use thiserror::Error;

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

pub struct NatsClient {
    pub client: Client,
    jetstream: jetstream::Context,
}

impl NatsClient {
    pub async fn connect(url: &str) -> Result<Self, NatsError> {
        let client = async_nats::connect(url).await?;
        let jetstream = jetstream::new(client.clone());
        Ok(Self { client, jetstream })
    }

    /// Create all required JetStream streams and KV buckets.
    /// Safe to call multiple times — uses get_or_create semantics.
    pub async fn ensure_infrastructure(&self) -> Result<(), NatsError> {
        // Stream 1: all task orchestration events (durable, file-backed)
        self.jetstream
            .get_or_create_stream(stream::Config {
                name: "H2AI_TASKS".to_owned(),
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
                name: "H2AI_TELEMETRY".to_owned(),
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
                name: "H2AI_RESULTS".to_owned(),
                subjects: vec!["h2ai.results.>".to_owned()],
                storage: stream::StorageType::Memory,
                retention: stream::RetentionPolicy::WorkQueue,
                ..Default::default()
            })
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;

        // KV bucket: calibration cache
        self.ensure_kv_bucket(kv::Config {
            bucket: "H2AI_CALIBRATION".to_owned(),
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: durable session memory
        self.ensure_kv_bucket(kv::Config {
            bucket: "H2AI_SESSIONS".to_owned(),
            description: "Durable session memory — pipeline conversation history".to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: task state snapshots for crash-recovery replay optimization
        self.ensure_kv_bucket(kv::Config {
            bucket: "H2AI_SNAPSHOTS".to_owned(),
            description: "Task state snapshots — latest-only, accelerates replay after crash"
                .to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: TaoMultiplierEstimator EMA state for drift tracking
        self.ensure_kv_bucket(kv::Config {
            bucket: "H2AI_ESTIMATOR".to_owned(),
            description: "TaoMultiplierEstimator EMA state — survives restarts".to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: rolling oracle calibration observations for conformal interval estimation
        self.ensure_kv_bucket(kv::Config {
            bucket: "H2AI_ORACLE_CALIBRATION".to_owned(),
            description: "Rolling oracle calibration window — max 200 OracleObservation entries"
                .to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: task phase checkpoints for crash-recovery (zstd-compressed, latest-only)
        self.ensure_kv_bucket(kv::Config {
            bucket: "H2AI_TASK_CHECKPOINTS".to_owned(),
            description: "Task phase checkpoints — zstd-compressed, latest-only per task"
                .to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            max_age: std::time::Duration::from_secs(86400), // 24h TTL
            ..Default::default()
        })
        .await?;

        // Object Store bucket: checkpoint payload overflow for entries > 800 KB
        self.ensure_object_store(async_nats::jetstream::object_store::Config {
            bucket: "H2AI_CHECKPOINT_PAYLOADS".to_owned(),
            description: Some(
                "Checkpoint payload overflow — delete before KV entry on GC".to_owned(),
            ),
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: HITL approval records pending human decision
        self.ensure_kv_bucket(kv::Config {
            bucket: "H2AI_APPROVALS".to_owned(),
            description: "HITL approval records awaiting human decision".to_owned(),
            history: 1,
            storage: stream::StorageType::File,
            max_age: std::time::Duration::from_secs(3600), // 1h TTL — longer than max review timeout
            ..Default::default()
        })
        .await?;

        // KV bucket: compact tag→[constraint_id] index (small, cacheable with TTL)
        self.ensure_kv_bucket(kv::Config {
            bucket: "H2AI_CONSTRAINT_WIKI".to_owned(),
            description: "Compact constraint tag index — tag→[id] map, lazy-loaded per request"
                .to_owned(),
            history: 5,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // KV bucket: individual constraint metas — one entry per constraint, fetched on demand
        self.ensure_kv_bucket(kv::Config {
            bucket: "H2AI_CONSTRAINT_META".to_owned(),
            description: "Per-constraint metadata — fetched lazily by ID, never bulk-loaded"
                .to_owned(),
            history: 3,
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        // Object Store: full constraint predicate payloads (LlmJudge rubrics, Oracle configs)
        self.ensure_object_store(async_nats::jetstream::object_store::Config {
            bucket: "H2AI_CONSTRAINT_PAYLOADS".to_owned(),
            description: Some(
                "Constraint predicate payloads — lazy-fetched during Phase 4 evaluation".to_owned(),
            ),
            storage: stream::StorageType::File,
            ..Default::default()
        })
        .await?;

        Ok(())
    }

    /// Publish a typed H2AIEvent to the task's JetStream subject.
    pub async fn publish_event(
        &self,
        task_id: &TaskId,
        event: &H2AIEvent,
    ) -> Result<(), NatsError> {
        let subject = format!("h2ai.tasks.{task_id}");
        self.publish_to(&subject, event).await
    }

    /// Publish a typed H2AIEvent to an arbitrary subject.
    pub async fn publish_to(&self, subject: &str, event: &H2AIEvent) -> Result<(), NatsError> {
        let payload = serde_json::to_vec(event).map_err(|e| NatsError::Serialize(e.to_string()))?;
        self.jetstream
            .publish(subject.to_owned(), payload.into())
            .await
            .map_err(|e| NatsError::PublishError(e.to_string()))?;
        Ok(())
    }

    /// Like `publish_event` but awaits the `PubAck` and returns the JetStream sequence number.
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
            .get_key_value("H2AI_SNAPSHOTS")
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
            .get_key_value("H2AI_SNAPSHOTS")
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
            .get_key_value("H2AI_CALIBRATION")
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
            .get_key_value("H2AI_CALIBRATION")
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
            .get_key_value("H2AI_ORACLE_CALIBRATION")
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
            .get_key_value("H2AI_ORACLE_CALIBRATION")
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

    /// Persist the TaoMultiplierEstimator EMA state so it survives process restarts.
    pub async fn put_tao_estimator_state(&self, ema: f64, count: usize) -> Result<(), NatsError> {
        #[derive(serde::Serialize)]
        struct State {
            ema: f64,
            count: usize,
        }
        let payload = serde_json::to_vec(&State { ema, count })
            .map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value("H2AI_ESTIMATOR")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put("tao_multiplier/state", payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve the persisted TaoMultiplierEstimator EMA state, or `None` if absent.
    pub async fn get_tao_estimator_state(&self) -> Result<Option<(f64, usize)>, NatsError> {
        #[derive(serde::Deserialize)]
        struct State {
            ema: f64,
            count: usize,
        }
        let kv = self
            .jetstream
            .get_key_value("H2AI_ESTIMATOR")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get("tao_multiplier/state").await {
            Ok(Some(entry)) => {
                let s: State = serde_json::from_slice(&entry)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some((s.ema, s.count)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// Persist raw JSON bytes to the `H2AI_ESTIMATOR` bucket under key `bandit_state`.
    /// Callers are responsible for serialization (avoids a circular crate dependency).
    pub async fn put_bandit_state(&self, json_bytes: Vec<u8>) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_ESTIMATOR")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.put("bandit_state", json_bytes.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Retrieve raw JSON bytes for `BanditState` from the `H2AI_ESTIMATOR` bucket.
    /// Returns `None` when no entry exists (first run). Callers deserialize the bytes.
    pub async fn get_bandit_state(&self) -> Result<Option<Vec<u8>>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_ESTIMATOR")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.get("bandit_state").await {
            Ok(Some(entry)) => Ok(Some(entry.to_vec())),
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
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
            .get_stream("H2AI_TASKS")
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
            let seq = msg.info().map(|i| i.stream_sequence).unwrap_or(0);
            let event: H2AIEvent = serde_json::from_slice(&msg.payload)
                .map_err(|e| NatsError::Serialize(e.to_string()))?;
            Ok((seq, event))
        });
        Ok(Box::pin(mapped))
    }

    /// Publish a TaskPayload to the ephemeral task subject for an edge agent.
    /// Subject: h2ai.tasks.ephemeral.{task_id}  (core NATS, not JetStream)
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

    /// Subscribe to H2AI_RESULTS JetStream and return the first TaskResult
    /// for the given task_id within the given timeout.
    ///
    /// IMPORTANT: Call this BEFORE publish_task_payload to avoid the race
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
            .get_stream("H2AI_RESULTS")
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

    // ── internal ────────────────────────────────────────────────────────────

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
            .get_key_value("H2AI_TASK_CHECKPOINTS")
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

    /// Load and decompress a checkpoint by task_id. Returns `None` if not found.
    pub async fn get_task_checkpoint(
        &self,
        task_id: &str,
    ) -> Result<Option<TaskCheckpoint>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_TASK_CHECKPOINTS")
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
        use futures::TryStreamExt;

        let kv = match self.jetstream.get_key_value("H2AI_TASK_CHECKPOINTS").await {
            Ok(kv) => kv,
            Err(e) => {
                tracing::warn!("list_task_checkpoints: KV open failed: {e}");
                return vec![];
            }
        };
        let keys: Vec<String> = match kv.keys().await {
            Ok(stream) => stream.try_collect().await.unwrap_or_default(),
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
                    .get_object_store("H2AI_CHECKPOINT_PAYLOADS")
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
                    Err(e) => tracing::warn!("failed to open H2AI_CHECKPOINT_PAYLOADS: {e}"),
                }
            }
        }
        // 2. Delete the KV entry
        let kv = self
            .jetstream
            .get_key_value("H2AI_TASK_CHECKPOINTS")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        kv.delete(task_id)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    // ── approval records ────────────────────────────────────────────────────────────

    /// Store an approval record pending human review.
    pub async fn put_approval_record(
        &self,
        record: &h2ai_types::approval::ApprovalRecord,
    ) -> Result<u64, NatsError> {
        let payload =
            serde_json::to_vec(record).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let kv = self
            .jetstream
            .get_key_value("H2AI_APPROVALS")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let rev = kv
            .put(&record.task_id, payload.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(rev)
    }

    /// Load an approval record AND its current KV revision.
    /// The revision is required for the CAS delete in the reaper.
    pub async fn get_approval_record_with_revision(
        &self,
        task_id: &str,
    ) -> Result<Option<(h2ai_types::approval::ApprovalRecord, u64)>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_APPROVALS")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv.entry(task_id).await {
            Ok(Some(entry)) => {
                // Deleted/purged entries still return Some — treat them as not found.
                if entry.operation != async_nats::jetstream::kv::Operation::Put {
                    return Ok(None);
                }
                let revision = entry.revision;
                let record =
                    serde_json::from_slice::<h2ai_types::approval::ApprovalRecord>(&entry.value)
                        .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some((record, revision)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }

    /// List all approval records with their revisions.
    /// Used by the reaper to find expired records.
    pub async fn list_approval_records_with_revision(
        &self,
    ) -> Vec<(h2ai_types::approval::ApprovalRecord, u64)> {
        use futures::TryStreamExt;
        let kv = match self.jetstream.get_key_value("H2AI_APPROVALS").await {
            Ok(kv) => kv,
            Err(e) => {
                tracing::warn!("list_approval_records: KV open failed: {e}");
                return vec![];
            }
        };
        let keys: Vec<String> = match kv.keys().await {
            Ok(stream) => stream.try_collect().await.unwrap_or_default(),
            Err(e) => {
                tracing::warn!("list_approval_records: keys() failed: {e}");
                return vec![];
            }
        };
        let mut result = Vec::with_capacity(keys.len());
        for key in &keys {
            if let Ok(Some(pair)) = self.get_approval_record_with_revision(key).await {
                result.push(pair);
            }
        }
        result
    }

    /// Atomically delete the approval record only if the revision matches.
    ///
    /// Returns `Ok(())` only if this node won the CAS race.
    /// Returns `Err` if another node already deleted or updated the record.
    pub async fn delete_approval_record_if_revision(
        &self,
        task_id: &str,
        expected_revision: u64,
    ) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_APPROVALS")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        // Use update with empty bytes as the atomic claim — only succeeds if revision matches.
        kv.update(task_id, vec![].into(), expected_revision)
            .await
            .map_err(|e| {
                NatsError::KvError(format!("CAS delete failed (revision mismatch): {e}"))
            })?;
        // Clean up the tombstone entry
        kv.delete(task_id)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Write the compiled WikiCache to NATS KV.
    ///
    /// Pass `expected_revision = Some(rev)` for optimistic CAS (prevents concurrent overwrites).
    /// Pass `None` for an unconditional put (first write or forced refresh).
    pub async fn put_wiki_cache(
        &self,
        cache: &h2ai_constraints::wiki::WikiCache,
        expected_revision: Option<u64>,
    ) -> Result<u64, NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_CONSTRAINT_WIKI")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let bytes = serde_json::to_vec(cache).map_err(|e| NatsError::Serialize(e.to_string()))?;
        let revision = match expected_revision {
            Some(rev) => kv
                .update("index", bytes.into(), rev)
                .await
                .map_err(|e| NatsError::KvError(e.to_string()))?,
            None => kv
                .put("index", bytes.into())
                .await
                .map_err(|e| NatsError::KvError(e.to_string()))?,
        };
        Ok(revision)
    }

    /// Read the compiled WikiCache from NATS KV.
    ///
    /// Returns `None` if the wiki has not been bootstrapped yet.
    pub async fn get_wiki_cache(
        &self,
    ) -> Result<Option<(h2ai_constraints::wiki::WikiCache, u64)>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_CONSTRAINT_WIKI")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv
            .entry("index")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?
        {
            Some(entry) => {
                let revision = entry.revision;
                let mut cache: h2ai_constraints::wiki::WikiCache =
                    serde_json::from_slice(&entry.value)
                        .map_err(|e| NatsError::Serialize(e.to_string()))?;
                cache.revision = revision;
                Ok(Some((cache, revision)))
            }
            None => Ok(None),
        }
    }

    /// Store the compact tag→[constraint_id] index in NATS KV.
    ///
    /// Key: "tag_index". Much smaller than the full WikiCache blob.
    pub async fn put_tag_index(
        &self,
        index: &std::collections::HashMap<String, Vec<String>>,
    ) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_CONSTRAINT_WIKI")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let bytes = serde_json::to_vec(index).map_err(|e| NatsError::Serialize(e.to_string()))?;
        kv.put("tag_index", bytes.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Fetch the compact tag→[constraint_id] index from NATS KV.
    pub async fn get_tag_index(
        &self,
    ) -> Result<Option<std::collections::HashMap<String, Vec<String>>>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_CONSTRAINT_WIKI")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv
            .entry("tag_index")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?
        {
            Some(entry) => {
                let index = serde_json::from_slice(&entry.value)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(index))
            }
            None => Ok(None),
        }
    }

    /// Store a single ConstraintMeta by ID. Key = constraint ID.
    pub async fn put_constraint_meta(
        &self,
        meta: &h2ai_constraints::types::ConstraintMeta,
    ) -> Result<(), NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_CONSTRAINT_META")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let bytes = serde_json::to_vec(meta).map_err(|e| NatsError::Serialize(e.to_string()))?;
        kv.put(&meta.id, bytes.into())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Fetch a single ConstraintMeta by ID. Returns `None` if not found.
    pub async fn get_constraint_meta(
        &self,
        id: &str,
    ) -> Result<Option<h2ai_constraints::types::ConstraintMeta>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_CONSTRAINT_META")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        match kv
            .entry(id)
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?
        {
            Some(entry) => {
                let meta = serde_json::from_slice(&entry.value)
                    .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(meta))
            }
            None => Ok(None),
        }
    }

    /// Fetch multiple ConstraintMeta by ID in parallel. Missing IDs are silently skipped.
    pub async fn get_constraint_metas(
        &self,
        ids: &[String],
    ) -> Result<Vec<h2ai_constraints::types::ConstraintMeta>, NatsError> {
        let kv = self
            .jetstream
            .get_key_value("H2AI_CONSTRAINT_META")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let kv = std::sync::Arc::new(kv);
        let futures: Vec<_> = ids
            .iter()
            .map(|id| {
                let kv = kv.clone();
                let id = id.clone();
                async move {
                    match kv.entry(&id).await {
                        Ok(Some(entry)) => serde_json::from_slice::<
                            h2ai_constraints::types::ConstraintMeta,
                        >(&entry.value)
                        .ok(),
                        _ => None,
                    }
                }
            })
            .collect();
        let results = futures::future::join_all(futures).await;
        Ok(results.into_iter().flatten().collect())
    }

    /// Store a ConstraintPayload in the Object Store.
    ///
    /// Key format: `{id}@{version}` — e.g., `GDPR-DPA-001@v2`.
    pub async fn put_constraint_payload(
        &self,
        payload: &h2ai_constraints::types::ConstraintPayload,
    ) -> Result<(), NatsError> {
        let os = self
            .jetstream
            .get_object_store("H2AI_CONSTRAINT_PAYLOADS")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{}@{}", payload.id, payload.version);
        let bytes = serde_json::to_vec(payload).map_err(|e| NatsError::Serialize(e.to_string()))?;
        os.put(key.as_str(), &mut bytes.as_slice())
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        Ok(())
    }

    /// Fetch a ConstraintPayload from the Object Store by (id, version).
    ///
    /// Returns `None` if the payload does not exist.
    pub async fn get_constraint_payload(
        &self,
        id: &str,
        version: &str,
    ) -> Result<Option<h2ai_constraints::types::ConstraintPayload>, NatsError> {
        let os = self
            .jetstream
            .get_object_store("H2AI_CONSTRAINT_PAYLOADS")
            .await
            .map_err(|e| NatsError::KvError(e.to_string()))?;
        let key = format!("{id}@{version}");
        match os.get(&key).await {
            Ok(mut obj) => {
                use tokio::io::AsyncReadExt;
                let mut buf = Vec::new();
                obj.read_to_end(&mut buf)
                    .await
                    .map_err(|e| NatsError::KvError(e.to_string()))?;
                let payload: h2ai_constraints::types::ConstraintPayload =
                    serde_json::from_slice(&buf)
                        .map_err(|e| NatsError::Serialize(e.to_string()))?;
                Ok(Some(payload))
            }
            Err(e) if e.to_string().contains("not found") => Ok(None),
            Err(e) => Err(NatsError::KvError(e.to_string())),
        }
    }
}

#[cfg(test)]
mod wire_protocol_tests {
    // These tests require a running NATS server.
    // Run with: H2AI_INTEGRATION_TEST=1 cargo test -p h2ai-state -- --ignored
    use super::*;
    use h2ai_types::agent::{AgentDescriptor, ContextPayload, TaskPayload, TaskResult};
    use h2ai_types::identity::{AgentId, TaskId};
    use h2ai_types::sizing::TauValue;
    use std::time::Duration;

    #[tokio::test]
    #[ignore]
    async fn publish_and_receive_task_payload() {
        let nats_url = h2ai_config::H2AIConfig::default().nats_url;
        let nats = match NatsClient::connect(&nats_url).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
                return;
            }
        };
        nats.ensure_infrastructure().await.expect("infra");

        let task_id = TaskId::new();
        let agent_id = AgentId::from("test-agent");

        // Subscriber must be set up before publish
        let subject = h2ai_nats::subjects::ephemeral_task_subject(&task_id);
        let mut sub = nats.client.subscribe(subject.clone()).await.unwrap();

        let payload = TaskPayload {
            task_id: task_id.clone(),
            agent_id: agent_id.clone(),
            agent: AgentDescriptor {
                model: "mock".into(),
                tools: vec![],
                cost_tier: h2ai_types::agent::CostTier::Mid,
            },
            instructions: "test".into(),
            context: ContextPayload::Inline("ctx".into()),
            tau: TauValue::new(0.5).unwrap(),
            max_tokens: 256,
            wave_mode: h2ai_types::agent::WaveMode::Normal,
        };
        nats.publish_task_payload(&payload).await.expect("publish");

        use futures::StreamExt;
        let msg = tokio::time::timeout(Duration::from_secs(2), sub.next())
            .await
            .expect("timeout")
            .expect("msg");
        let decoded: TaskPayload = serde_json::from_slice(&msg.payload).unwrap();
        assert_eq!(decoded.task_id, task_id);
        assert_eq!(decoded.agent_id, agent_id);
    }

    #[tokio::test]
    #[ignore]
    async fn await_task_result_round_trip() {
        let nats_url = h2ai_config::H2AIConfig::default().nats_url;
        let nats = match NatsClient::connect(&nats_url).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
                return;
            }
        };
        nats.ensure_infrastructure().await.expect("infra");

        let task_id = TaskId::new();
        let agent_id = AgentId::from("test-agent");

        // Consumer MUST be set up before publish
        let nats2 = NatsClient::connect(&nats_url).await.unwrap();
        let tid = task_id.clone();
        let waiter = tokio::spawn(async move {
            nats2
                .await_task_result_once(&tid, Duration::from_secs(5))
                .await
        });
        // Small yield to let consumer set up
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Publish result as if an edge agent did it
        let result = TaskResult {
            task_id: task_id.clone(),
            agent_id: agent_id.clone(),
            output: "hello".into(),
            token_cost: 10,
            error: None,
            tool_calls: vec![],
        };
        let js = async_nats::jetstream::new(nats.client.clone());
        let result_subject = h2ai_nats::subjects::task_result_subject(&task_id);
        js.publish(result_subject, serde_json::to_vec(&result).unwrap().into())
            .await
            .unwrap()
            .await
            .unwrap();

        let received = waiter.await.unwrap().expect("result");
        assert_eq!(received.output, "hello");
        assert_eq!(received.task_id, task_id);
    }
}
