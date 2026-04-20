use crate::error::AuditError;
use crate::provider::AuditProvider;
use async_trait::async_trait;
use h2ai_types::agent::AgentTelemetryEvent;

pub struct DirectLogProvider;

#[async_trait]
impl AuditProvider for DirectLogProvider {
    async fn record_event(&self, event: AgentTelemetryEvent) -> Result<(), AuditError> {
        let json =
            serde_json::to_string(&event).map_err(|e| AuditError::Serialization(e.to_string()))?;
        println!("{json}");
        Ok(())
    }

    async fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }
}
