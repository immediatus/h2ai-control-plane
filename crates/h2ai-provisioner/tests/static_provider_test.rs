use futures::StreamExt;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_provisioner::static_provider::StaticProvider;
use h2ai_types::agent::{AgentDescriptor, AgentTool};
use h2ai_types::identity::AgentId;

fn descriptor(model: &str) -> AgentDescriptor {
    AgentDescriptor {
        model: model.to_owned(),
        tools: vec![],
        cost_tier: h2ai_types::agent::CostTier::Mid,
    }
}

#[tokio::test]
async fn static_provider_ok_when_within_capacity() {
    let provider = StaticProvider::new(10);
    assert!(provider
        .ensure_agent_capacity(&descriptor("gpt-4o"), 5)
        .await
        .is_ok());
}

#[tokio::test]
async fn static_provider_capacity_error_when_over_limit() {
    let provider = StaticProvider::new(3);
    let result = provider
        .ensure_agent_capacity(&descriptor("claude-sonnet-4-6"), 5)
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn static_provider_terminate_returns_ok() {
    let provider = StaticProvider::new(10);
    assert!(provider
        .terminate_agent(&AgentId::from("agent-1"))
        .await
        .is_ok());
}

#[tokio::test]
async fn static_provider_capacity_boundary_at_limit_is_ok() {
    let provider = StaticProvider::new(5);
    assert!(provider
        .ensure_agent_capacity(
            &AgentDescriptor {
                model: "gpt-4o".into(),
                tools: vec![AgentTool::WebSearch],
                cost_tier: h2ai_types::agent::CostTier::Mid,
            },
            5
        )
        .await
        .is_ok());
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn static_provider_terminate_publishes_to_nats() {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    let nats = async_nats::connect(&url).await.expect("connect");
    let provider = StaticProvider::new(10).with_nats(nats.clone());

    let mut sub = nats.subscribe("h2ai.control.terminate.>").await.unwrap();

    provider
        .terminate_agent(&AgentId::from("agent-123"))
        .await
        .expect("terminate_agent failed");

    let msg = tokio::time::timeout(std::time::Duration::from_secs(1), sub.next())
        .await
        .expect("timeout waiting for nats message")
        .expect("no message on sub");

    assert_eq!(msg.subject.as_str(), "h2ai.control.terminate.agent-123");
}
