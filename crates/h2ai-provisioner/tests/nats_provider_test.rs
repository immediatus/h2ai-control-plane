// Integration tests require live NATS at localhost:4222.
use h2ai_provisioner::provider::AgentProvider;
use h2ai_types::agent::{AgentDescriptor, AgentHeartbeat};
use h2ai_types::identity::AgentId;
use std::time::Duration;

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn heartbeat_registers_agent_and_capacity_check_passes() {
    use h2ai_provisioner::nats_provider::NatsAgentProvider;

    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    let nats = async_nats::connect(&url).await.expect("connect");
    let provider = NatsAgentProvider::new(nats.clone(), Duration::from_secs(10))
        .await
        .expect("create provider");
    let agent_id = AgentId::from(format!("test-agent-{}", uuid::Uuid::new_v4()));
    let descriptor = AgentDescriptor {
        model: "gpt-4o".into(),
        tools: vec![],
        cost_tier: h2ai_types::agent::CostTier::Mid,
    };
    let heartbeat = AgentHeartbeat {
        agent_id: agent_id.clone(),
        descriptor: descriptor.clone(),
        timestamp: chrono::Utc::now(),
        active_tasks: 0,
    };
    nats.publish(
        format!("h2ai.heartbeat.{}", agent_id),
        serde_json::to_vec(&heartbeat).unwrap().into(),
    )
    .await
    .expect("publish heartbeat");
    tokio::time::sleep(Duration::from_millis(100)).await;
    provider
        .ensure_agent_capacity(&descriptor, 1)
        .await
        .expect("one live agent must satisfy load=1");
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn no_heartbeat_means_capacity_limit_reached() {
    use h2ai_provisioner::nats_provider::NatsAgentProvider;

    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    let nats = async_nats::connect(&url).await.expect("connect");
    let provider = NatsAgentProvider::new(nats, Duration::from_secs(10))
        .await
        .expect("create provider");
    let descriptor = AgentDescriptor {
        model: "nonexistent-model-xyz".into(),
        tools: vec![],
        cost_tier: h2ai_types::agent::CostTier::Mid,
    };
    let err = provider.ensure_agent_capacity(&descriptor, 1).await;
    assert!(err.is_err(), "no agents → should error");
}
