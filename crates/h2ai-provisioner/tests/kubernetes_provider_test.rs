#![cfg(not(feature = "kubernetes"))]

use h2ai_provisioner::kubernetes_provider::KubernetesProvider;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_types::agent::{AgentDescriptor, AgentTool, CostTier, TaskRequirements};
use h2ai_types::identity::AgentId;

fn descriptor(model: &str, tools: Vec<AgentTool>) -> AgentDescriptor {
    AgentDescriptor {
        model: model.to_owned(),
        tools,
        cost_tier: h2ai_types::agent::CostTier::Mid,
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

#[tokio::test]
async fn kubernetes_provider_select_agent_returns_unavailable() {
    let provider = KubernetesProvider::new_stub("default");
    let req = TaskRequirements {
        max_cost_tier: CostTier::Mid,
        required_tools: vec![],
    };
    let err = provider.select_agent(&req).await;
    assert!(
        err.is_err(),
        "KubernetesProvider always returns NoAgentsAvailable"
    );
}
