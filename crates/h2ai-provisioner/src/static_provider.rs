use crate::error::ProvisionError;
use crate::provider::AgentProvider;
use async_trait::async_trait;
use h2ai_types::agent::AgentDescriptor;
use h2ai_types::identity::AgentId;

pub struct StaticProvider {
    pub max_task_load: usize,
}

impl StaticProvider {
    pub fn new(max_task_load: usize) -> Self {
        Self { max_task_load }
    }
}

#[async_trait]
impl AgentProvider for StaticProvider {
    async fn ensure_agent_capacity(
        &self,
        descriptor: &AgentDescriptor,
        task_load: usize,
    ) -> Result<(), ProvisionError> {
        // TODO: verify via NATS heartbeat subject `h2ai.heartbeat.<model>.<task_load>`
        if task_load > self.max_task_load {
            return Err(ProvisionError::CapacityLimitReached {
                agent_type: descriptor.model.clone(),
            });
        }
        Ok(())
    }

    async fn terminate_agent(&self, agent_id: &AgentId) -> Result<(), ProvisionError> {
        // TODO: publish soft-kill to NATS subject `h2ai.control.terminate.<agent_id>`
        let _ = agent_id; // suppress unused warning until NATS is wired
        Ok(())
    }
}
