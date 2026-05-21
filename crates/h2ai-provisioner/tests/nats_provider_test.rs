use h2ai_config::SchedulerPolicy;
use h2ai_provisioner::nats_provider::NatsAgentProvider;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_types::agent::{AgentDescriptor, AgentHeartbeat, CostTier, TaskRequirements};
use h2ai_types::identity::AgentId;
use std::time::Duration;

async fn connect() -> Option<async_nats::Client> {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    match async_nats::connect(&url).await {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            None
        }
    }
}

#[tokio::test]
async fn heartbeat_registers_agent_and_capacity_check_passes() {
    use h2ai_provisioner::nats_provider::NatsAgentProvider;

    let Some(nats) = connect().await else {
        return;
    };
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
        format!("h2ai.heartbeat.{agent_id}"),
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
async fn no_heartbeat_means_capacity_limit_reached() {
    use h2ai_provisioner::nats_provider::NatsAgentProvider;

    let Some(nats) = connect().await else {
        return;
    };
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

#[tokio::test]
async fn nats_provider_zero_ttl_returns_transport_error() {
    let Some(nats) = connect().await else {
        return;
    };
    let result = NatsAgentProvider::new(nats, Duration::ZERO).await;
    assert!(result.is_err(), "zero TTL must return error");
}

#[tokio::test]
async fn nats_provider_with_policy_from_config_least_loaded() {
    let Some(nats) = connect().await else {
        return;
    };
    let provider = NatsAgentProvider::new(nats, Duration::from_secs(10))
        .await
        .expect("create provider")
        .with_policy_from_config(&SchedulerPolicy::LeastLoaded, 2);
    let req = TaskRequirements {
        max_cost_tier: CostTier::High,
        required_tools: vec![],
    };
    let _ = provider.select_agent(&req).await; // result doesn't matter
}

#[tokio::test]
async fn nats_provider_with_policy_from_config_spillover() {
    let Some(nats) = connect().await else {
        return;
    };
    let provider = NatsAgentProvider::new(nats, Duration::from_secs(10))
        .await
        .expect("create provider")
        .with_policy_from_config(&SchedulerPolicy::CostAwareSpillover, 3);
    let req = TaskRequirements {
        max_cost_tier: CostTier::High,
        required_tools: vec![],
    };
    let _ = provider.select_agent(&req).await;
}

#[tokio::test]
async fn nats_provider_terminate_no_nats_returns_error() {
    let provider = NatsAgentProvider::new_test_only();
    let err = provider.terminate_agent(&AgentId::from("agent-x")).await;
    assert!(err.is_err(), "test-only provider has no NATS → error");
}

#[tokio::test]
async fn nats_provider_capacity_limit_reached_when_load_exceeds_live() {
    let Some(nats) = connect().await else {
        return;
    };
    let provider = NatsAgentProvider::new(nats.clone(), Duration::from_secs(10))
        .await
        .expect("create provider");
    let agent_id = AgentId::from(format!("cap-agent-{}", uuid::Uuid::new_v4()));
    let descriptor = AgentDescriptor {
        model: "cap-test-model".into(),
        tools: vec![],
        cost_tier: CostTier::Mid,
    };
    let heartbeat = AgentHeartbeat {
        agent_id: agent_id.clone(),
        descriptor: descriptor.clone(),
        timestamp: chrono::Utc::now(),
        active_tasks: 0,
    };
    nats.publish(
        format!("h2ai.heartbeat.{agent_id}"),
        serde_json::to_vec(&heartbeat).unwrap().into(),
    )
    .await
    .expect("publish heartbeat");
    tokio::time::sleep(Duration::from_millis(100)).await;
    // live=1 but task_load=5 → CapacityLimitReached
    let err = provider.ensure_agent_capacity(&descriptor, 5).await;
    assert!(err.is_err(), "live=1 < need=5 → CapacityLimitReached");
}
