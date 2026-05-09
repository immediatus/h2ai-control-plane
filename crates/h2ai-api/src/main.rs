mod approval_reaper;
mod constraint_source;
mod error;
mod metrics;
mod oracle;
mod recovery;
mod routes;
mod state;

use axum::Router;
use h2ai_adapters::factory::AdapterFactory;
use h2ai_adapters::mock::MockAdapter;
use h2ai_config::{AdapterProfile, H2AIConfig};
use h2ai_state::nats::NatsClient;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::config::AdapterKind;
use state::AppState;
use std::env;
use std::sync::Arc;
use tracing::warn;

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

/// Resolve adapter kind for a role: prefer env vars; fall back to the config's
/// adapter_profiles (prefers a profile named "local", then the first profile).
fn adapter_kind_for_role(prefix: &str, profiles: &[AdapterProfile]) -> AdapterKind {
    let provider = env::var(format!("H2AI_{prefix}_PROVIDER"))
        .unwrap_or_default()
        .to_lowercase();
    if !provider.is_empty() && provider != "mock" {
        return adapter_kind_from_env(prefix);
    }
    // No explicit env var — use config adapter_profiles if available.
    if let Some(profile) = profiles
        .iter()
        .find(|p| p.name == "local")
        .or_else(|| profiles.first())
    {
        return profile.kind.clone();
    }
    adapter_kind_from_env(prefix)
}

fn adapter_family(kind: &AdapterKind) -> &'static str {
    match kind {
        AdapterKind::Anthropic { .. } => "anthropic",
        AdapterKind::OpenAI { .. } => "openai",
        AdapterKind::Ollama { .. } => "ollama",
        AdapterKind::LocalLlamaCpp { .. } => "llamacpp",
        AdapterKind::CloudGeneric { .. } => "cloudgeneric",
        AdapterKind::A2a { .. } => "a2a",
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
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let listen_addr = env::var("H2AI_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());

    let cfg = {
        use std::path::Path;
        if let Ok(path) = env::var("H2AI_CONFIG") {
            let p = Path::new(&path);
            eprintln!("INFO: loading config from H2AI_CONFIG={path}");
            H2AIConfig::load_layered(Some(p))
                .unwrap_or_else(|e| panic!("H2AI_CONFIG={path} failed to load: {e}"))
        } else if Path::new("h2ai.toml").exists() {
            eprintln!("INFO: loading config from ./h2ai.toml");
            H2AIConfig::load_layered(Some(Path::new("h2ai.toml")))
                .unwrap_or_else(|e| panic!("h2ai.toml failed to load: {e}"))
        } else {
            eprintln!("INFO: no override config found — using reference defaults");
            H2AIConfig::load_layered(None).expect("embedded reference.toml is always valid")
        }
    };

    let nats = NatsClient::connect(&cfg.nats_url)
        .await
        .unwrap_or_else(|e| {
            eprintln!(
                "ERROR: cannot connect to NATS at {} — {e}\n       Start a NATS server first:  nats-server -js",
                cfg.nats_url
            );
            std::process::exit(1);
        });
    nats.ensure_infrastructure()
        .await
        .expect("NATS infrastructure setup");

    let profiles = &cfg.adapter_profiles;
    let explorer_kind = adapter_kind_for_role("EXPLORER", profiles);
    let auditor_kind = adapter_kind_for_role("AUDITOR", profiles);
    let explorer_adapter = build_adapter(&explorer_kind);
    let auditor_adapter = build_adapter(&auditor_kind);

    let scoring_kind_opt = {
        let provider = env::var("H2AI_SCORING_PROVIDER")
            .unwrap_or_else(|_| "none".into())
            .to_lowercase();
        if provider == "none" || provider.is_empty() {
            None
        } else {
            Some(adapter_kind_for_role("SCORING", profiles))
        }
    };
    let scoring_adapter: Option<Arc<dyn IComputeAdapter>> =
        scoring_kind_opt.as_ref().map(build_adapter);

    let explorer2_kind_opt = {
        let provider = env::var("H2AI_EXPLORER2_PROVIDER")
            .unwrap_or_else(|_| "same".into())
            .to_lowercase();
        if provider == "same" || provider.is_empty() {
            None
        } else {
            Some(adapter_kind_for_role("EXPLORER2", profiles))
        }
    };
    let explorer2_adapter: Arc<dyn IComputeAdapter> = explorer2_kind_opt
        .as_ref()
        .map(build_adapter)
        .unwrap_or_else(|| explorer_adapter.clone());

    eprintln!("explorer  adapter: {:?}", explorer_kind);
    eprintln!(
        "explorer2 adapter: {:?}",
        explorer2_kind_opt.as_ref().unwrap_or(&explorer_kind)
    );
    eprintln!("auditor   adapter: {:?}", auditor_kind);
    eprintln!("scoring   adapter: {:?}", scoring_kind_opt);

    if adapter_family(&explorer_kind) == adapter_family(&auditor_kind) {
        warn!(
            target: "h2ai.verification",
            family = adapter_family(&explorer_kind),
            "verification_adapter and explorer_adapter are the same family — \
             self-preference bias likely. Configure a different model family for verification."
        );
    }

    let mut app_state = AppState::new(nats, cfg, explorer_adapter, auditor_adapter)
        .with_explorer2(explorer2_adapter);
    if let Some(sa) = scoring_adapter {
        app_state.scoring_adapter = Some(sa);
    }

    app_state.load_tao_estimator().await;
    app_state.load_bandit_state().await;
    app_state.load_wiki_cache().await;

    // Recover in-flight tasks from checkpoints persisted before last restart
    crate::recovery::recover_in_flight_tasks(Arc::new(app_state.clone())).await;

    // Spawn Phase 6 oracle accumulator — subscribes to h2ai.oracle.results
    // and accumulates calibration residuals in NATS KV.
    {
        let accumulator = crate::oracle::OracleAccumulator {
            nats_raw: app_state.nats.client.clone(),
            nats_state: app_state.nats.clone(),
            bandit: app_state.bandit_state.clone(),
            metrics: app_state.metrics.clone(),
        };
        tokio::spawn(accumulator.run());
    }

    // Spawn HITL approval reaper — scans for timed-out approvals every 60s
    {
        let reaper_state = Arc::new(app_state.clone());
        tokio::spawn(crate::approval_reaper::run_approval_reaper(reaper_state));
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
