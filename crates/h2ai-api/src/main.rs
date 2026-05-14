mod approval_reaper;
mod bootstrap;
mod constraint_source;
mod debug_record;
mod error;
mod metrics;
mod opro;
mod oracle;
mod oracle_worker;
mod recovery;
mod rho_ema;
mod routes;
mod shadow_auditor;
mod state;

use axum::Router;
use h2ai_adapters::factory::AdapterFactory;
use h2ai_adapters::mock::MockAdapter;
use h2ai_config::{AdapterProfile, FamilyConstraint, H2AIConfig};
use h2ai_provisioner::nats_provider::NatsAgentProvider;
use h2ai_state::nats::NatsClient;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::config::AdapterKind;
use state::AppState;
use std::env;
use std::sync::Arc;
use std::time::Duration;
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
            model: env::var(format!("H2AI_{prefix}_MODEL")).ok(),
        },
        _ => AdapterKind::CloudGeneric {
            endpoint: "mock://localhost".into(),
            api_key_env: "MOCK".into(),
            model: None,
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

fn build_adapter(kind: &AdapterKind, enable_thinking: bool) -> Arc<dyn IComputeAdapter> {
    match AdapterFactory::build_with_thinking(kind, enable_thinking) {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!(target: "h2ai.startup", adapter = ?kind, error = %e,
                "adapter could not be built; falling back to MockAdapter");
            Arc::new(MockAdapter::new("mock fallback output".into()))
        }
    }
}

/// Resolve the config file path: H2AI_CONFIG env var takes priority (test/task override),
/// then /etc/h2ai/h2ai.toml (deployment default), then None (embedded reference.toml).
fn resolve_config_path() -> Option<String> {
    if let Ok(path) = std::env::var("H2AI_CONFIG") {
        return Some(path);
    }
    let default = "/etc/h2ai/h2ai.toml";
    if std::path::Path::new(default).exists() {
        return Some(default.to_owned());
    }
    None
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // --validate-config: load config, print startup report, exit without starting servers.
    if std::env::args().any(|a| a == "--validate-config") {
        let config_path = resolve_config_path();
        match H2AIConfig::load_layered(config_path.as_deref().map(std::path::Path::new)) {
            Ok(cfg) => {
                h2ai_config::log_startup_config_report(&cfg);
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("Config validation failed: {e}");
                std::process::exit(1);
            }
        }
    }

    let listen_addr = env::var("H2AI_LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());

    let cfg = {
        let config_path = resolve_config_path();
        match &config_path {
            Some(path) => {
                tracing::info!(target: "h2ai.startup", config_path = %path, "loading config");
                H2AIConfig::load_layered(Some(std::path::Path::new(path)))
                    .unwrap_or_else(|e| panic!("config {path} failed to load: {e}"))
            }
            None => {
                tracing::info!(target: "h2ai.startup", "no config file found — using embedded reference defaults");
                H2AIConfig::load_layered(None).expect("embedded reference.toml is always valid")
            }
        }
    };

    h2ai_config::log_startup_config_report(&cfg);

    let nats = NatsClient::connect_with_cfg(&cfg.nats_url, cfg.state.clone())
        .await
        .unwrap_or_else(|e| {
            tracing::error!(target: "h2ai.startup", nats_url = %cfg.nats_url, error = %e,
                "cannot connect to NATS — start a NATS server first: nats-server -js");
            std::process::exit(1);
        });
    nats.ensure_infrastructure()
        .await
        .expect("NATS infrastructure setup");

    // Seed Bayesian bootstrap priors for all adapter profiles so Thompson sampling
    // doesn't start cold. Idempotent — skips adapters that already have OPRO state.
    bootstrap::seed_all_bootstrap_priors(&cfg.adapter_profiles, &cfg.calibration_bootstrap, &nats)
        .await;

    if let Err(e) = nats.put_safety_profile_snapshot(&cfg.safety).await {
        tracing::warn!(target: "h2ai.startup", error = %e, "failed to publish safety profile snapshot to NATS KV");
    }

    let profiles = &cfg.adapter_profiles;
    let explorer_kind = adapter_kind_for_role("EXPLORER", profiles);
    let auditor_kind = adapter_kind_for_role("AUDITOR", profiles);
    let thinking = cfg.adapter_enable_thinking;
    let explorer_adapter = build_adapter(&explorer_kind, thinking);
    let auditor_adapter = build_adapter(&auditor_kind, thinking);

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
    let scoring_adapter: Option<Arc<dyn IComputeAdapter>> = scoring_kind_opt
        .as_ref()
        .map(|k| build_adapter(k, thinking));

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
        .map(|k| build_adapter(k, thinking))
        .unwrap_or_else(|| explorer_adapter.clone());

    let shadow_auditor_kind_opt = {
        let provider = env::var("H2AI_SHADOW_AUDITOR_PROVIDER")
            .unwrap_or_else(|_| "none".into())
            .to_lowercase();
        if provider == "none" || provider.is_empty() {
            None
        } else {
            Some(adapter_kind_for_role("SHADOW_AUDITOR", profiles))
        }
    };
    let shadow_auditor_adapter: Option<Arc<dyn IComputeAdapter>> = shadow_auditor_kind_opt
        .as_ref()
        .map(|k| build_adapter(k, thinking));

    let researcher_kind_opt = {
        let provider = env::var("H2AI_RESEARCHER_PROVIDER")
            .unwrap_or_else(|_| "none".into())
            .to_lowercase();
        if provider == "none" || provider.is_empty() {
            None
        } else {
            Some(adapter_kind_for_role("RESEARCHER", profiles))
        }
    };
    let researcher_adapter: Option<Arc<dyn IComputeAdapter>> = researcher_kind_opt
        .as_ref()
        .map(|k| build_adapter(k, thinking));

    tracing::info!(target: "h2ai.startup", adapter = ?explorer_kind, "explorer adapter");
    tracing::info!(target: "h2ai.startup", adapter = ?explorer2_kind_opt.as_ref().unwrap_or(&explorer_kind), "explorer2 adapter");
    tracing::info!(target: "h2ai.startup", adapter = ?auditor_kind, "auditor adapter");
    tracing::info!(target: "h2ai.startup", adapter = ?scoring_kind_opt, "scoring adapter");
    tracing::info!(target: "h2ai.startup", adapter = ?shadow_auditor_kind_opt, "shadow adapter");
    tracing::info!(target: "h2ai.startup", adapter = ?researcher_kind_opt, "researcher adapter");

    if let Some(ref sk) = shadow_auditor_kind_opt {
        if adapter_family(sk) == adapter_family(&auditor_kind) {
            tracing::error!(target: "h2ai.startup",
                family = %adapter_family(sk),
                "shadow_auditor and auditor are the same family — shadow mode requires a different family. \
                 Set H2AI_SHADOW_AUDITOR_PROVIDER to a different provider.");
            std::process::exit(1);
        }
    }

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
    if let Some(shadow) = shadow_auditor_adapter {
        app_state = app_state.with_shadow_auditor(shadow);
    }
    if let Some(researcher) = researcher_adapter {
        app_state.researcher_adapter = Some(researcher);
    }

    // Populate startup-only safety gauges from config.
    {
        let mut m = app_state.metrics.write().await;
        m.safety_profile_name = app_state.cfg.safety.profile.as_str().to_string();
        m.safety_krum_fault_tolerance = app_state.cfg.safety.krum_fault_tolerance as u64;
        m.safety_diversity_threshold = app_state.cfg.safety.diversity_threshold;
        m.safety_shadow_auditor_enabled = app_state.cfg.safety.shadow_auditor.enabled as u8;
        m.safety_require_bivariate_cg = app_state.cfg.safety.require_bivariate_cg as u8;
    }

    {
        use h2ai_orchestrator::srani_grounding::{
            LlmResearcherGrounder, SpecAnchorGrounder, SraniGroundingChain,
        };
        let srani_cfg = &app_state.cfg.srani;
        let mut tiers: Vec<Box<dyn h2ai_orchestrator::srani_grounding::GroundingProvider>> =
            vec![Box::new(SpecAnchorGrounder)];
        if let Some(ref r) = app_state.researcher_adapter {
            tiers.push(Box::new(LlmResearcherGrounder::new(r.clone())));
        }
        let chain = SraniGroundingChain::new(tiers);
        // Wire distiller from the researcher adapter if distillation is enabled.
        let chain = if let Some(ref r) = app_state.researcher_adapter {
            chain.with_distiller(
                r.clone(),
                srani_cfg.grounding_raw_max_chars,
                srani_cfg.grounding_hint_max_chars,
                srani_cfg.grounding_distill,
            )
        } else {
            chain
        };
        app_state.srani_grounding_chain = Some(std::sync::Arc::new(chain));
        tracing::info!(
            target: "h2ai.startup",
            distill = srani_cfg.grounding_distill,
            raw_max = srani_cfg.grounding_raw_max_chars,
            hint_max = srani_cfg.grounding_hint_max_chars,
            "SRANI grounding chain built"
        );
    }

    app_state.load_tao_estimator().await;
    app_state.load_bandit_state().await;
    app_state.load_srani_state().await;
    app_state.load_calibration().await;

    // Restore promoted shadow-audit domains persisted before last restart.
    {
        match app_state.nats.get_shadow_promoted_domains().await {
            Ok(domains) if !domains.is_empty() => {
                *app_state.promoted_audit_domains.write().await = domains;
                tracing::info!(target: "h2ai.shadow_auditor", "restored promoted domains from NATS KV");
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(target: "h2ai.shadow_auditor", error = %e, "failed to load promoted domains from NATS");
            }
        }
    }

    // Always run calibration at startup so USL coefficients reflect current hardware.
    // Persisted calibration (loaded above) remains as fallback if the LLM is unreachable.
    {
        use h2ai_types::adapter::AdapterFamily;
        use std::collections::HashSet;

        let pre_families: HashSet<AdapterFamily> = [
            app_state.explorer_adapter.family(),
            app_state.explorer2_adapter.family(),
            app_state.verification_adapter.family(),
        ]
        .into_iter()
        .filter(|f| *f != AdapterFamily::Mock)
        .collect();
        let single_family_warning = pre_families.len() == 1;
        if single_family_warning {
            match app_state.cfg.safety.family_constraint {
                FamilyConstraint::RequireDiverse => {
                    tracing::error!("all non-Mock adapters share one family — aborting");
                    std::process::exit(1);
                }
                FamilyConstraint::SingleFamilyOk => {
                    tracing::warn!(
                        "single-family adapter pool: correlated hallucination protection degraded"
                    );
                }
                FamilyConstraint::Disabled => {}
            }
        }
        let mut adapter_families: Vec<String> =
            pre_families.iter().map(|f| f.to_string()).collect();
        adapter_families.sort();
        let explorer_verification_family_match = app_state.explorer_adapter.family()
            == app_state.verification_adapter.family()
            && app_state.explorer_adapter.family() != AdapterFamily::Mock;

        let had_calibration = app_state.calibration.read().await.is_some();
        tracing::info!(target: "h2ai.startup", "running startup calibration");
        crate::routes::calibrate::run_calibration_core(
            app_state.clone(),
            single_family_warning,
            explorer_verification_family_match,
            adapter_families,
            None,
        )
        .await;

        if app_state.calibration.read().await.is_none() {
            if had_calibration {
                tracing::warn!(target: "h2ai.startup", "startup calibration failed (LLM unreachable?); using persisted calibration");
            } else {
                tracing::warn!(target: "h2ai.startup",
                    "startup calibration did not complete (LLM unreachable?); tasks will be rejected until POST /calibrate succeeds");
            }
        } else {
            tracing::info!(target: "h2ai.startup", "startup calibration complete — ready to accept tasks");
        }
    }

    // Wire NATS dispatch when H2AI_NATS_DISPATCH=true.
    // When enabled, explorer slots are dispatched to TaoAgent processes via NATS
    // rather than calling the in-process LLM adapter.
    if std::env::var("H2AI_NATS_DISPATCH")
        .unwrap_or_default()
        .eq_ignore_ascii_case("true")
    {
        let ttl = Duration::from_secs(
            std::env::var("H2AI_AGENT_TTL_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30),
        );
        match NatsAgentProvider::new(app_state.nats.client.clone(), ttl).await {
            Ok(provider) => {
                tracing::info!(target: "h2ai.startup", ttl = ?ttl, "NATS agent dispatch enabled");
                app_state = app_state.with_agent_provider(Arc::new(provider));
            }
            Err(e) => {
                tracing::warn!(target: "h2ai.startup", error = %e, "NatsAgentProvider init failed — falling back to direct adapters");
            }
        }
    }

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
            oracle_window_size: app_state.cfg.oracle_window_size,
            oracle_ece_alert_threshold: app_state.cfg.oracle_ece_alert_threshold,
            oracle_pass_rate_floor: app_state.cfg.oracle_pass_rate_floor,
            calibration: app_state.calibration.clone(),
            calibration_max_ensemble_size: app_state.cfg.calibration_max_ensemble_size,
        };
        tokio::spawn(accumulator.run());
    }

    // Spawn oracle worker — subscribes to h2ai.oracle.pending, runs tests via
    // ShellExecutor, and publishes OracleResultEvent to h2ai.oracle.results.
    {
        let oracle_worker = crate::oracle_worker::OracleWorker::new(app_state.nats.client.clone());
        tokio::spawn(oracle_worker.run());
    }

    // Wire shadow auditor accumulator — driven by process() calls from tasks.rs.
    if app_state.shadow_auditor_adapter.is_some() && app_state.cfg.safety.shadow_auditor.enabled {
        let shadow_state = Arc::new(app_state.clone());
        let acc = crate::shadow_auditor::ShadowAuditorAccumulator::new(shadow_state);
        app_state.shadow_accumulator = Some(Arc::new(tokio::sync::Mutex::new(acc)));
        tracing::info!(target: "h2ai.startup", "shadow auditor accumulator wired");
    }

    // Spawn HITL approval reaper — scans for timed-out approvals every 60s
    {
        let reaper_state = Arc::new(app_state.clone());
        tokio::spawn(crate::approval_reaper::run_approval_reaper(reaper_state));
    }

    // Route versions are permanent API contracts — hardcoded, not runtime config.
    // `api_version` in config declares which version is current/stable (for docs,
    // logging, and client defaults). To add v2: keep /v1 nests and add /v2 nests;
    // all versions are served simultaneously so clients migrate on their own schedule.
    {
        let current = &app_state.cfg.api_version;
        assert!(
            current == "v1",
            "api_version={current:?} but only v1 routes are compiled; \
             add /v2 route handlers before changing api_version"
        );
        tracing::info!(target: "h2ai.startup", api_version = %current, "stable API ready");
    }
    let app = Router::new()
        // v1 routes — permanent; never remove, only add new version nests alongside
        .nest("/v1", routes::task_router())
        .nest("/v1", routes::calibrate_router())
        // health/ready/metrics are always at root — never versioned
        .merge(routes::health_router())
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(&listen_addr).await.unwrap();
    tracing::info!(target: "h2ai.startup", addr = %listen_addr, "listening");
    axum::serve(listener, app).await.unwrap();
}
