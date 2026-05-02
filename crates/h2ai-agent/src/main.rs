use h2ai_agent::dispatch;
use h2ai_agent::heartbeat::HeartbeatTask;
use h2ai_config::H2AIConfig;
use h2ai_types::agent::{AgentDescriptor, CostTier};
use h2ai_types::identity::AgentId;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cfg = H2AIConfig::default();
    let nats_url = cfg.nats_url;

    let agent_id_str = std::env::var("H2AI_AGENT_ID").unwrap_or_else(|_| String::new());
    let agent_id: AgentId = if agent_id_str.is_empty() {
        AgentId::from(uuid::Uuid::new_v4().to_string())
    } else {
        AgentId::from(agent_id_str)
    };

    let model = std::env::var("H2AI_EXPLORER_MODEL").unwrap_or_else(|_| "mock".into());
    let descriptor = AgentDescriptor {
        model: model.clone(),
        tools: vec![],
        cost_tier: CostTier::Mid,
    };

    let client = async_nats::connect(&nats_url).await?;
    tracing::info!(agent_id = %agent_id, model = %model, "h2ai-agent connected to NATS");

    // Shared active task counter (heartbeat reports it, dispatch increments/decrements it)
    let active_tasks = Arc::new(AtomicU32::new(0));

    let hb_task = HeartbeatTask::new(
        client.clone(),
        agent_id.clone(),
        descriptor.clone(),
        Duration::from_secs(10),
        active_tasks.clone(),
    );
    let _hb_handle = hb_task.start();

    // Build a mock adapter for now (Task 6 can wire real adapters)
    let adapter: Arc<dyn h2ai_types::adapter::IComputeAdapter> =
        Arc::new(h2ai_adapters::mock::MockAdapter::new(String::new()));

    dispatch::run(client, agent_id, adapter, active_tasks).await
}
