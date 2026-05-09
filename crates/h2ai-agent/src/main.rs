use h2ai_adapters::factory::AdapterFactory;
use h2ai_adapters::mock::MockAdapter;
use h2ai_agent::dispatch;
use h2ai_agent::heartbeat::HeartbeatTask;
use h2ai_config::H2AIConfig;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::agent::{AgentDescriptor, AgentTool, CostTier};
use h2ai_types::config::AdapterKind;
use h2ai_types::identity::AgentId;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::Duration;

pub fn agent_tools() -> Vec<AgentTool> {
    vec![AgentTool::Shell, AgentTool::FileSystem]
}

fn build_adapter() -> Arc<dyn IComputeAdapter> {
    let provider = std::env::var("H2AI_EXPLORER_PROVIDER")
        .unwrap_or_default()
        .to_lowercase();
    let model = std::env::var("H2AI_EXPLORER_MODEL").unwrap_or_else(|_| "gpt-4o".into());
    let api_key_env =
        std::env::var("H2AI_EXPLORER_API_KEY_ENV").unwrap_or_else(|_| "OPENAI_API_KEY".into());
    let endpoint = std::env::var("H2AI_EXPLORER_ENDPOINT").ok();

    let kind = match provider.as_str() {
        "anthropic" => AdapterKind::Anthropic { api_key_env, model },
        "openai" => AdapterKind::OpenAI { api_key_env, model },
        "ollama" => AdapterKind::Ollama {
            endpoint: endpoint.unwrap_or_else(|| "http://localhost:11434".into()),
            model,
        },
        "cloudgeneric" | "cloud" => AdapterKind::CloudGeneric {
            endpoint: endpoint.unwrap_or_else(|| "http://localhost:8000/v1".into()),
            api_key_env,
        },
        _ => {
            eprintln!("WARN: H2AI_EXPLORER_PROVIDER not set or unknown — using MockAdapter");
            return Arc::new(MockAdapter::new(String::new()));
        }
    };

    match AdapterFactory::build(&kind) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("WARN: adapter build failed ({e}) — falling back to MockAdapter");
            Arc::new(MockAdapter::new(String::new()))
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cfg = Arc::new(H2AIConfig::default());
    h2ai_agent::config_validation::validate_tool_configs(&cfg);
    let nats_url = cfg.nats_url.clone();

    let agent_id_str = std::env::var("H2AI_AGENT_ID").unwrap_or_else(|_| String::new());
    let agent_id: AgentId = if agent_id_str.is_empty() {
        AgentId::from(uuid::Uuid::new_v4().to_string())
    } else {
        AgentId::from(agent_id_str)
    };

    let model = std::env::var("H2AI_EXPLORER_MODEL").unwrap_or_else(|_| "local".into());
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

    let adapter = build_adapter();
    dispatch::run(client, agent_id, adapter, active_tasks, cfg).await
}

#[cfg(test)]
mod tests {
    use h2ai_types::agent::AgentTool;

    #[test]
    fn agent_tools_list_is_complete() {
        let tools = crate::agent_tools();
        assert!(tools.contains(&AgentTool::Shell));
        assert!(tools.contains(&AgentTool::FileSystem));
    }
}
