use h2ai_adapters::factory::AdapterFactory;
use h2ai_adapters::mock::MockAdapter;
use h2ai_agent::dispatch;
use h2ai_agent::heartbeat::HeartbeatTask;
use h2ai_agent::tools::agent_tools;
use h2ai_config::H2AIConfig;
use h2ai_types::adapter::IComputeAdapter;
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

    let config_path = std::env::var("H2AI_CONFIG").ok();
    let cfg = Arc::new(
        H2AIConfig::load_layered(config_path.as_deref().map(std::path::Path::new))
            .expect("config load failed"),
    );
    h2ai_agent::config_validation::validate_tool_configs(&cfg);
    let nats_url = cfg.nats_url.clone();

    let agent_id = AgentId::from(uuid::Uuid::new_v4().to_string());

    let model = cfg
        .adapter_profiles
        .first()
        .map_or_else(|| "local".into(), |p| p.name.clone());
    let descriptor = AgentDescriptor {
        model: model.clone(),
        tools: agent_tools(),
        cost_tier: CostTier::Mid,
    };

    let client = async_nats::connect(&nats_url).await?;
    tracing::info!(agent_id = %agent_id, model = %model, "h2ai-agent connected to NATS");

    let active_tasks = Arc::new(AtomicU32::new(0));

    let hb_task = HeartbeatTask::new(
        client.clone(),
        agent_id.clone(),
        descriptor.clone(),
        Duration::from_secs(10),
        active_tasks.clone(),
    );
    let _hb_handle = hb_task.start();

    let thinking = cfg.adapter_enable_thinking;
    let adapter: Arc<dyn IComputeAdapter> = cfg.adapter_profiles.first().map_or_else(
        || Arc::new(MockAdapter::new(String::new())) as Arc<dyn IComputeAdapter>,
        |p| match AdapterFactory::build_with_thinking(&p.kind, thinking) {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!("adapter build failed ({e}) — falling back to MockAdapter");
                Arc::new(MockAdapter::new(String::new()))
            }
        },
    );

    dispatch::run(client, agent_id, adapter, active_tasks, cfg).await
}
