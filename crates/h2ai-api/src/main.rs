mod error;
mod routes;
mod state;

use axum::Router;
use h2ai_adapters::factory::AdapterFactory;
use h2ai_adapters::mock::MockAdapter;
use h2ai_config::H2AIConfig;
use h2ai_state::nats::NatsClient;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::config::AdapterKind;
use state::AppState;
use std::env;
use std::sync::Arc;

fn adapter_kind_from_env(prefix: &str) -> AdapterKind {
    let provider = env::var(format!("H2AI_{prefix}_PROVIDER"))
        .unwrap_or_else(|_| "mock".into())
        .to_lowercase();
    let model = env::var(format!("H2AI_{prefix}_MODEL")).unwrap_or_else(|_| "gpt-4o".into());
    let api_key_env =
        env::var(format!("H2AI_{prefix}_API_KEY_ENV")).unwrap_or_else(|_| "OPENAI_API_KEY".into());
    let endpoint = env::var(format!("H2AI_{prefix}_ENDPOINT")).ok();

    match provider.as_str() {
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
        _ => AdapterKind::CloudGeneric {
            endpoint: "mock://localhost".into(),
            api_key_env: "MOCK".into(),
        },
    }
}

fn build_adapter(kind: &AdapterKind) -> Arc<dyn IComputeAdapter> {
    match AdapterFactory::build(kind) {
        Ok(a) => a,
        Err(_) => {
            eprintln!(
                "WARN: adapter kind {kind:?} could not be built; falling back to MockAdapter"
            );
            Arc::new(MockAdapter::new("mock fallback output".into()))
        }
    }
}

#[tokio::main]
async fn main() {
    let nats_url = env::var("H2AI_NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    let listen_addr = env::var("H2AI_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());

    let nats = NatsClient::connect(&nats_url).await.expect("NATS connect");
    nats.ensure_infrastructure()
        .await
        .expect("NATS infrastructure setup");

    let cfg = H2AIConfig::default();

    let explorer_kind = adapter_kind_from_env("EXPLORER");
    let auditor_kind = adapter_kind_from_env("AUDITOR");
    let explorer_adapter = build_adapter(&explorer_kind);
    let auditor_adapter = build_adapter(&auditor_kind);

    let scoring_kind_opt = {
        let provider = env::var("H2AI_SCORING_PROVIDER")
            .unwrap_or_else(|_| "none".into())
            .to_lowercase();
        if provider == "none" || provider.is_empty() {
            None
        } else {
            Some(adapter_kind_from_env("SCORING"))
        }
    };
    let scoring_adapter: Option<Arc<dyn IComputeAdapter>> =
        scoring_kind_opt.as_ref().map(build_adapter);

    eprintln!("explorer adapter: {:?}", explorer_kind);
    eprintln!("auditor  adapter: {:?}", auditor_kind);
    eprintln!("scoring  adapter: {:?}", scoring_kind_opt);

    let mut app_state = AppState::new(nats, cfg, explorer_adapter, auditor_adapter);
    if let Some(sa) = scoring_adapter {
        app_state.scoring_adapter = Some(sa);
    }

    let app = Router::new()
        .merge(routes::task_router())
        .merge(routes::calibrate_router())
        .merge(routes::health_router())
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(&listen_addr).await.unwrap();
    eprintln!("listening on {listen_addr}");
    axum::serve(listener, app).await.unwrap();
}
