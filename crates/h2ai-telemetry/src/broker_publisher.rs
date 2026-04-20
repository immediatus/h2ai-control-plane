use crate::error::AuditError;
use crate::provider::AuditProvider;
use async_nats::Client;
use async_trait::async_trait;
use h2ai_types::agent::AgentTelemetryEvent;

pub struct BrokerPublisherProvider {
    client: Client,
    subject_prefix: String,
}

impl BrokerPublisherProvider {
    pub fn new(client: Client, subject_prefix: impl Into<String>) -> Self {
        Self {
            client,
            subject_prefix: subject_prefix.into(),
        }
    }
}

#[async_trait]
impl AuditProvider for BrokerPublisherProvider {
    async fn record_event(&self, event: AgentTelemetryEvent) -> Result<(), AuditError> {
        let json =
            serde_json::to_string(&event).map_err(|e| AuditError::Serialization(e.to_string()))?;
        let subject = format!("{}.event", self.subject_prefix);
        self.client
            .publish(subject, json.into())
            .await
            .map_err(|e| AuditError::Transport(e.to_string()))?;
        Ok(())
    }

    async fn flush(&self) -> Result<(), AuditError> {
        self.client
            .flush()
            .await
            .map_err(|e| AuditError::Flush(e.to_string()))?;
        Ok(())
    }
}
