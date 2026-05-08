use crate::error::ProvisionError;
use crate::provider::AgentProvider;
use async_trait::async_trait;
use h2ai_types::agent::{AgentDescriptor, TaskRequirements};
use h2ai_types::identity::AgentId;

pub struct StaticProvider {
    pub max_task_load: usize,
    pub nats: Option<async_nats::Client>,
}

impl StaticProvider {
    pub fn new(max_task_load: usize) -> Self {
        Self {
            max_task_load,
            nats: None,
        }
    }

    pub fn with_nats(mut self, nats: async_nats::Client) -> Self {
        self.nats = Some(nats);
        self
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
        if let Some(ref nats) = self.nats {
            let subject = h2ai_nats::subjects::agent_terminate_subject(agent_id);
            nats.publish(subject, bytes::Bytes::new())
                .await
                .map_err(|e| ProvisionError::Transport(e.to_string()))?;
        }
        Ok(())
    }

    async fn select_agent(
        &self,
        requirements: &TaskRequirements,
    ) -> Result<AgentId, ProvisionError> {
        // StaticProvider has no live agent registry — always unavailable.
        Err(ProvisionError::NoAgentsAvailable {
            max_tier: requirements.max_cost_tier.clone(),
            tools: requirements.required_tools.clone(),
        })
    }
}
