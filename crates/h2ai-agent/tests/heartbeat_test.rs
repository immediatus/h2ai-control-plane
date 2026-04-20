use futures::StreamExt;
use h2ai_agent::heartbeat::HeartbeatTask;
use h2ai_types::agent::{AgentDescriptor, AgentHeartbeat, CostTier};
use h2ai_types::identity::AgentId;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
#[ignore]
async fn heartbeat_publishes_to_correct_subject() {
    let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    let client = async_nats::connect(&nats_url).await.unwrap();
    let agent_id = AgentId::from(uuid::Uuid::new_v4().to_string());
    let descriptor = AgentDescriptor {
        model: "mock".into(),
        tools: vec![],
        cost_tier: CostTier::Low,
    };
    let active_tasks = Arc::new(AtomicU32::new(3));

    let expected_subject = format!("h2ai.heartbeat.{agent_id}");
    let mut sub = client.subscribe(expected_subject.clone()).await.unwrap();

    let task = HeartbeatTask::new(
        client.clone(),
        agent_id.clone(),
        descriptor.clone(),
        Duration::from_millis(100),
        active_tasks,
    );
    let handle = task.start();

    let msg = tokio::time::timeout(Duration::from_secs(2), sub.next())
        .await
        .expect("timeout waiting for heartbeat")
        .expect("no message");

    let hb: AgentHeartbeat = serde_json::from_slice(&msg.payload).unwrap();
    assert_eq!(hb.agent_id, agent_id);
    assert_eq!(hb.descriptor.model, "mock");
    assert_eq!(hb.active_tasks, 3);

    handle.abort();
}
