use h2ai_provisioner::kubernetes_provider::KubernetesProvider;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_types::agent::{AgentDescriptor, AgentTool};
use h2ai_types::identity::AgentId;

fn descriptor(model: &str, tools: Vec<AgentTool>) -> AgentDescriptor {
    AgentDescriptor {
        model: model.to_owned(),
        tools,
    }
}

#[tokio::test]
async fn kubernetes_provider_stub_returns_ok() {
    let provider = KubernetesProvider::new_stub("default");
    assert!(provider
        .ensure_agent_capacity(&descriptor("claude-sonnet-4-6", vec![AgentTool::Shell]), 1)
        .await
        .is_ok());
    assert!(provider
        .terminate_agent(&AgentId::from("agent-1"))
        .await
        .is_ok());
}

#[tokio::test]
async fn kubernetes_provider_stub_handles_toolless_agent() {
    let provider = KubernetesProvider::new_stub("h2ai-system");
    assert!(provider
        .ensure_agent_capacity(&descriptor("gpt-4o", vec![]), 3)
        .await
        .is_ok());
}
