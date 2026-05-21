use crate::error::AuditError;
use crate::provider::AuditProvider;
use async_nats::Client;
use async_trait::async_trait;
use h2ai_types::agent::AgentTelemetryEvent;
use std::sync::Arc;

/// Abstracts the NATS publish + flush operations so `BrokerPublisherProvider`
/// can be tested without a live NATS server.
#[async_trait]
pub trait NatsPublishClient: Send + Sync {
    async fn publish_bytes(&self, subject: String, payload: Vec<u8>) -> Result<(), AuditError>;
    async fn flush(&self) -> Result<(), AuditError>;
}

#[async_trait]
impl NatsPublishClient for Client {
    async fn publish_bytes(&self, subject: String, payload: Vec<u8>) -> Result<(), AuditError> {
        self.publish(subject, payload.into())
            .await
            .map_err(|e| AuditError::Transport(e.to_string()))
    }

    async fn flush(&self) -> Result<(), AuditError> {
        self.flush()
            .await
            .map_err(|e| AuditError::Flush(e.to_string()))
    }
}

pub struct BrokerPublisherProvider {
    client: Arc<dyn NatsPublishClient>,
    subject_prefix: String,
}

impl BrokerPublisherProvider {
    /// Production constructor — takes a live `async_nats::Client`.
    pub fn new(client: Client, subject_prefix: impl Into<String>) -> Self {
        Self {
            client: Arc::new(client),
            subject_prefix: subject_prefix.into(),
        }
    }

    /// Test constructor — accepts any `NatsPublishClient` implementation.
    pub fn with_client(
        client: Arc<dyn NatsPublishClient>,
        subject_prefix: impl Into<String>,
    ) -> Self {
        Self {
            client,
            subject_prefix: subject_prefix.into(),
        }
    }
}

#[async_trait]
impl AuditProvider for BrokerPublisherProvider {
    async fn record_event(&self, event: AgentTelemetryEvent) -> Result<(), AuditError> {
        let agent_id = match &event {
            AgentTelemetryEvent::LlmPromptSent { agent_id, .. }
            | AgentTelemetryEvent::LlmResponseReceived { agent_id, .. }
            | AgentTelemetryEvent::ShellCommandExecuted { agent_id, .. }
            | AgentTelemetryEvent::SystemError { agent_id, .. } => agent_id.to_string(),
        };
        let json =
            serde_json::to_string(&event).map_err(|e| AuditError::Serialization(e.to_string()))?;
        let subject = format!("{}.{}", self.subject_prefix, agent_id);
        self.client.publish_bytes(subject, json.into_bytes()).await
    }

    async fn flush(&self) -> Result<(), AuditError> {
        self.client.flush().await
    }
}
