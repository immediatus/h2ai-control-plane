use crate::error::AuditError;
use async_trait::async_trait;
use h2ai_types::agent::AgentTelemetryEvent;

#[async_trait]
pub trait AuditProvider: Send + Sync {
    async fn record_event(&self, event: AgentTelemetryEvent) -> Result<(), AuditError>;
    async fn flush(&self) -> Result<(), AuditError>;
}
