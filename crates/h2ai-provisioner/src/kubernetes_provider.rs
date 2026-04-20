use crate::error::ProvisionError;
use crate::provider::AgentProvider;
use async_trait::async_trait;
use h2ai_types::agent::{AgentDescriptor, TaskRequirements};
use h2ai_types::identity::AgentId;

#[cfg(feature = "kubernetes")]
use kube::Client as KubeClient;

/// Kubernetes-backed agent provisioner.
///
/// Provisions ephemeral LLM-based agent pods as Kubernetes Jobs.
/// Image and tool ConfigMaps are derived from the `AgentDescriptor`.
/// Each job receives scoped NATS NKeys via env vars.
///
/// # Phase 2 implementation
/// Currently scaffolded — full Job manifest generation is TODO.
pub struct KubernetesProvider {
    #[cfg(feature = "kubernetes")]
    client: KubeClient,
    #[allow(dead_code)] // used in Phase 2 Job manifest generation
    namespace: String,
}

impl KubernetesProvider {
    /// Create a provider that will spawn jobs in the given namespace.
    #[cfg(feature = "kubernetes")]
    pub fn new(client: KubeClient, namespace: impl Into<String>) -> Self {
        Self {
            client,
            namespace: namespace.into(),
        }
    }

    /// Create a no-op provider for testing without a live cluster.
    #[cfg(not(feature = "kubernetes"))]
    pub fn new_stub(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
        }
    }
}

#[async_trait]
impl AgentProvider for KubernetesProvider {
    async fn ensure_agent_capacity(
        &self,
        descriptor: &AgentDescriptor,
        task_load: usize,
    ) -> Result<(), ProvisionError> {
        // TODO (Phase 2): Generate a Kubernetes Job manifest:
        //   - Image derived from descriptor.model (e.g., "ghcr.io/h2ai/agent:latest")
        //   - Tool ConfigMaps mounted based on descriptor.tools
        //   - Env: NATS_NKEY_SEED injected from ScopedAgentCredentials
        //   - Labels: task_id, model for observability
        //   - Submit via kube::Api<Job>::create(...)
        let _ = (descriptor, task_load);
        Ok(())
    }

    async fn terminate_agent(&self, agent_id: &AgentId) -> Result<(), ProvisionError> {
        // TODO (Phase 2): Delete the Kubernetes Job for this agent_id
        //   - kube::Api<Job>::delete(agent_id, &DeleteParams::background())
        let _ = agent_id;
        Ok(())
    }

    async fn select_agent(
        &self,
        requirements: &TaskRequirements,
    ) -> Result<AgentId, ProvisionError> {
        Err(ProvisionError::NoAgentsAvailable {
            max_tier: requirements.max_cost_tier.clone(),
            tools: requirements.required_tools.clone(),
        })
    }
}
