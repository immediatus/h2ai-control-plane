use async_nats::jetstream::{self, kv, stream};
use async_nats::Client;
use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent, TaskSnapshot};
use h2ai_types::identity::TaskId;
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
        use jetstream::consumer::DeliverPolicy;

        let subject = task_result_subject(task_id);
        let consumer_cfg = jetstream::consumer::pull::OrderedConfig {
            filter_subject: subject,
            deliver_policy: DeliverPolicy::All,
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
}

#[cfg(test)]
mod wire_protocol_tests {
    // These tests require a running NATS server.
    // Run with: H2AI_INTEGRATION_TEST=1 cargo test -p h2ai-state -- --ignored
    use super::*;
    use h2ai_types::agent::{AgentDescriptor, ContextPayload, TaskPayload, TaskResult};
    use h2ai_types::identity::{AgentId, TaskId};
    use h2ai_types::physics::TauValue;
    use std::time::Duration;

    #[tokio::test]
    #[ignore]
    async fn publish_and_receive_task_payload() {
        let nats_url = std::env::var("NATS_URL")
            .unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
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
        let nats_url = std::env::var("NATS_URL")
            .unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
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
