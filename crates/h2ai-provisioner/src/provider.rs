use crate::error::ProvisionError;
use async_trait::async_trait;
use h2ai_types::agent::{AgentDescriptor, TaskRequirements};
use h2ai_types::identity::AgentId;

#[async_trait]
pub trait AgentProvider: Send + Sync {
    async fn ensure_agent_capacity(
        &self,
        descriptor: &AgentDescriptor,
        task_load: usize,
    ) -> Result<(), ProvisionError>;

    async fn terminate_agent(&self, agent_id: &AgentId) -> Result<(), ProvisionError>;

    async fn select_agent(
        &self,
        requirements: &TaskRequirements,
    ) -> Result<AgentId, ProvisionError>;
}
