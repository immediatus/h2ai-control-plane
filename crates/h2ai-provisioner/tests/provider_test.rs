use async_trait::async_trait;
use h2ai_provisioner::error::ProvisionError;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_types::agent::AgentDescriptor;
use h2ai_types::identity::AgentId;

struct AlwaysOkProvider;

#[async_trait]
impl AgentProvider for AlwaysOkProvider {
    async fn ensure_agent_capacity(
        &self,
        _descriptor: &AgentDescriptor,
        _task_load: usize,
    ) -> Result<(), ProvisionError> {
        Ok(())
    }

    async fn terminate_agent(&self, _agent_id: &AgentId) -> Result<(), ProvisionError> {
        Ok(())
    }
}

struct AlwaysFailProvider;

#[async_trait]
impl AgentProvider for AlwaysFailProvider {
    async fn ensure_agent_capacity(
        &self,
        descriptor: &AgentDescriptor,
        _task_load: usize,
    ) -> Result<(), ProvisionError> {
        Err(ProvisionError::CapacityLimitReached {
            agent_type: descriptor.model.clone(),
        })
    }

    async fn terminate_agent(&self, agent_id: &AgentId) -> Result<(), ProvisionError> {
        Err(ProvisionError::AgentUnavailable {
            agent_id: agent_id.to_string(),
        })
    }
}

fn descriptor(model: &str) -> AgentDescriptor {
    AgentDescriptor {
        model: model.to_owned(),
        tools: vec![],
    }
}

#[tokio::test]
async fn agent_provider_ok_impl_returns_ok() {
    let provider = AlwaysOkProvider;
    assert!(provider
        .ensure_agent_capacity(&descriptor("gpt-4o"), 1)
        .await
        .is_ok());
    assert!(provider
        .terminate_agent(&AgentId::from("agent-1"))
        .await
        .is_ok());
}

#[tokio::test]
async fn agent_provider_fail_impl_returns_errors() {
    let provider = AlwaysFailProvider;
    let result = provider
        .ensure_agent_capacity(&descriptor("claude-sonnet-4-6"), 5)
        .await;
    assert!(result.is_err());
    let term_result = provider.terminate_agent(&AgentId::from("agent-x")).await;
    assert!(term_result.is_err());
}

#[tokio::test]
async fn provision_error_display_messages_are_meaningful() {
    let err1 = ProvisionError::AgentUnavailable {
        agent_id: "agent-42".into(),
    };
    assert!(err1.to_string().contains("agent-42"));
    let err2 = ProvisionError::CapacityLimitReached {
        agent_type: "gpt-4o".into(),
    };
    assert!(err2.to_string().contains("gpt-4o"));
}
