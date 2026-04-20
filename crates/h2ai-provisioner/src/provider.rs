use crate::error::ProvisionError;
use async_trait::async_trait;
use h2ai_types::agent::AgentDescriptor;
use h2ai_types::identity::AgentId;

#[async_trait]
pub trait AgentProvider: Send + Sync {
    async fn ensure_agent_capacity(
        &self,
        descriptor: &AgentDescriptor,
        task_load: usize,
    ) -> Result<(), ProvisionError>;

    async fn terminate_agent(&self, agent_id: &AgentId) -> Result<(), ProvisionError>;
}
