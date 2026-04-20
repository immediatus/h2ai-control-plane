use async_nats::jetstream::{self, kv, stream};
use async_nats::Client;
use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent};
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

    /// Open a push consumer on `h2ai.tasks.{task_id}` and return a stream of events.
    ///
    /// `from_seq = 0` delivers from the beginning; non-zero starts after that sequence.
    pub async fn tail_task_events(
        &self,
        task_id: &TaskId,
        from_seq: u64,
    ) -> Result<impl futures::Stream<Item = Result<(u64, H2AIEvent), NatsError>>, NatsError> {
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
        let consumer_cfg = jetstream::consumer::push::Config {
            filter_subject: subject,
            deliver_policy,
            ..Default::default()
        };
        let stream = self
            .jetstream
            .get_stream("H2AI_TASKS")
            .await
            .map_err(|e| NatsError::StreamError(e.to_string()))?;
        let consumer = stream
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
            drop(msg.ack());
            Ok((seq, event))
        });
        Ok(mapped)
    }

    // ── internal ────────────────────────────────────────────────────────────

    async fn ensure_kv_bucket(&self, config: kv::Config) -> Result<(), NatsError> {
        match self.jetstream.get_key_value(&config.bucket).await {
            Ok(_) => Ok(()),
            Err(_) => self
                .jetstream
                .create_key_value(config)
                .await
                .map(|_| ())
                .map_err(|e| NatsError::KvError(e.to_string())),
        }
    }
}
