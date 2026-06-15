#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::option_if_let_else,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::doc_markdown,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::literal_string_with_formatting_args,
    clippy::missing_fields_in_debug,
    clippy::redundant_else,
    clippy::map_unwrap_or,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::large_futures,
    clippy::suboptimal_flops,
    clippy::needless_type_cast,
    clippy::elidable_lifetime_names
)]
mod bootstrap;
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
mod task_pipeline;
mod tenant_registry;

use axum::Router;
use h2ai_adapters::factory::AdapterFactory;
use h2ai_config::{FamilyConstraint, H2AIConfig};
use h2ai_provisioner::nats_provider::NatsAgentProvider;
use h2ai_state::nats::NatsClient;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::config::AdapterKind;
use state::AppState;
use std::sync::Arc;
use std::time::Duration;

fn build_adapter(kind: &AdapterKind, enable_thinking: bool) -> Arc<dyn IComputeAdapter> {
    AdapterFactory::build_with_thinking(kind, enable_thinking).unwrap_or_else(|e| {
        tracing::error!(target: "h2ai.startup", adapter = ?kind, error = %e,
                "adapter could not be built; cannot start without a valid adapter");
        std::process::exit(1);
    })
}

/// Resolve the config file path: `H2AI_CONFIG` env var takes priority (test/task override),
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
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
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

    let cfg = {
        let config_path = resolve_config_path();
        if let Some(path) = &config_path {
            tracing::info!(target: "h2ai.startup", config_path = %path, "loading config");
            H2AIConfig::load_layered(Some(std::path::Path::new(path)))
                .unwrap_or_else(|e| panic!("config {path} failed to load: {e}"))
        } else {
            tracing::info!(target: "h2ai.startup", "no config file found — using embedded reference defaults");
            H2AIConfig::load_layered(None).expect("embedded reference.toml is always valid")
        }
    };

    let listen_addr = cfg.listen_addr.clone();

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

    nats.provision_signals_stream()
        .await
        .expect("failed to provision H2AI_SIGNALS JetStream stream");

    // Seed Bayesian bootstrap priors for all adapter profiles so Thompson sampling
    // doesn't start cold. Idempotent — skips adapters that already have OPRO state.
    bootstrap::seed_all_bootstrap_priors(&cfg.adapter_profiles, &cfg.calibration_bootstrap, &nats)
        .await;

    if let Err(e) = nats.put_safety_profile_snapshot(&cfg.safety).await {
        tracing::warn!(target: "h2ai.startup", error = %e, "failed to publish safety profile snapshot to NATS KV");
    }

    let profiles = &cfg.adapter_profiles;
    let thinking = cfg.adapter_enable_thinking;

    // Build adapter_pool from config profiles. If profiles is empty, pool is empty —
    // that is the correct failure mode (config is wrong).
    let adapter_pool: Vec<Arc<dyn IComputeAdapter>> = profiles
        .iter()
        .map(|p| build_adapter(&p.kind, thinking))
        .collect();

    let auditor_kind = profiles
        .iter()
        .find(|p| p.name == "auditor")
        .or_else(|| profiles.first())
        .map_or_else(
            || AdapterKind::CloudGeneric {
                endpoint: "mock://localhost".into(),
                api_key_env: "MOCK".into(),
                model: None,
                provider: Default::default(),
            },
            |p| p.kind.clone(),
        );
    // Reuse the pool Arc when the auditor kind matches every explorer's kind so that
    // the pointer-based skip_audit check in phases/audit.rs fires correctly.
    let auditor_adapter: Arc<dyn IComputeAdapter> =
        if adapter_pool.iter().all(|a| *a.kind() == auditor_kind) && !adapter_pool.is_empty() {
            adapter_pool[0].clone()
        } else {
            build_adapter(&auditor_kind, thinking)
        };

    let scoring_kind_opt = profiles
        .iter()
        .find(|p| p.name == "scoring")
        .map(|p| p.kind.clone());
    let scoring_adapter: Option<Arc<dyn IComputeAdapter>> = scoring_kind_opt
        .as_ref()
        .map(|k| build_adapter(k, thinking));

    let shadow_auditor_kind_opt = if cfg.safety.shadow_auditor.enabled {
        profiles
            .iter()
            .find(|p| p.name == "shadow_auditor" || p.name == "shadow")
            .map(|p| p.kind.clone())
    } else {
        None
    };
    let shadow_auditor_adapter: Option<Arc<dyn IComputeAdapter>> = shadow_auditor_kind_opt
        .as_ref()
        .map(|k| build_adapter(k, thinking));

    let researcher_kind_opt = profiles
        .iter()
        .find(|p| p.name == "researcher")
        .map(|p| p.kind.clone());
    let researcher_adapter: Option<Arc<dyn IComputeAdapter>> = researcher_kind_opt
        .as_ref()
        .map(|k| build_adapter(k, thinking));

    tracing::info!(target: "h2ai.startup", pool_size = adapter_pool.len(), "adapter pool");
    tracing::info!(target: "h2ai.startup", adapter = ?auditor_kind, "auditor adapter");
    tracing::info!(target: "h2ai.startup", adapter = ?scoring_kind_opt, "scoring adapter");
    tracing::info!(target: "h2ai.startup", adapter = ?shadow_auditor_kind_opt, "shadow adapter");
    tracing::info!(target: "h2ai.startup", adapter = ?researcher_kind_opt, "researcher adapter");

    if cfg.safety.shadow_auditor.strict && !cfg.safety.shadow_auditor.enabled {
        tracing::warn!(
            target: "h2ai.startup",
            "shadow_auditor.strict = true but shadow_auditor.enabled = false; \
             strict mode has no effect without an enabled shadow auditor"
        );
    }

    let mut app_state = AppState::new(nats, cfg, adapter_pool, auditor_adapter);
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
        m.safety_shadow_auditor_enabled = u8::from(app_state.cfg.safety.shadow_auditor.enabled);
        m.safety_require_bivariate_cg = u8::from(app_state.cfg.safety.require_bivariate_cg);
    }

    {
        use h2ai_orchestrator::srani_grounding::{
            LlmResearcherGrounder, SpecAnchorGrounder, SraniGroundingChain,
        };
        let srani_cfg = &app_state.cfg.srani;
        let mut tiers: Vec<Box<dyn h2ai_orchestrator::srani_grounding::GroundingProvider>> =
            vec![Box::new(SpecAnchorGrounder)];
        if let Some(ref r) = app_state.researcher_adapter {
            tiers.push(Box::new(LlmResearcherGrounder::new(
                r.clone(),
                srani_cfg.researcher_max_tokens,
            )));
        }
        let chain = SraniGroundingChain::new(tiers)
            .with_compress_threshold(srani_cfg.grounding_compress_threshold);
        // Wire distiller from the researcher adapter if distillation is enabled.
        let chain = if let Some(ref r) = app_state.researcher_adapter {
            chain.with_distiller(
                r.clone(),
                srani_cfg.grounding_distill,
                srani_cfg.distill_max_tokens,
            )
        } else {
            chain
        };
        app_state.srani_grounding_chain = Some(std::sync::Arc::new(chain));
        tracing::info!(
            target: "h2ai.startup",
            distill = srani_cfg.grounding_distill,
            "SRANI grounding chain built"
        );
    }

    // ── gap research chain: StackOverflow + LLM distiller ───────────────
    // DuckDuckGo lite is CAPTCHA-blocked on cloud/devcontainer IPs; SO is not.
    {
        use h2ai_orchestrator::srani_grounding::{SraniGroundingChain, WebSearchGrounder};
        use h2ai_tools::web_search::StackOverflowSearchBackend;
        let backend = std::sync::Arc::new(StackOverflowSearchBackend::new());
        let web_grounder = WebSearchGrounder::new(backend, 5);
        let providers: Vec<Box<dyn h2ai_orchestrator::srani_grounding::GroundingProvider>> =
            vec![Box::new(web_grounder)];
        let srani_cfg = &app_state.cfg.srani;
        let chain = SraniGroundingChain::new(providers)
            .with_compress_threshold(srani_cfg.grounding_compress_threshold);
        let chain = if let Some(ref r) = app_state.researcher_adapter {
            chain.with_distiller(
                r.clone(),
                srani_cfg.grounding_distill,
                srani_cfg.distill_max_tokens,
            )
        } else {
            chain
        };
        app_state.gap_research_chain = Some(std::sync::Arc::new(chain));
        tracing::info!(target: "h2ai.startup", "gap research chain built (StackOverflow + distiller)");
    }

    // Wire knowledge provider — always a CompositeProvider that fans to [wiki, skill_provider]
    // so that extracted skills reach the thinking loop on every query.
    app_state.knowledge_provider = {
        use h2ai_config::ConstraintWikiConfig;
        use h2ai_knowledge::factory::KnowledgeProviderFactory;
        use h2ai_knowledge::provider::{KnowledgeProvider, PassthroughProvider};
        use h2ai_knowledge::skill_provider::CompositeProvider;
        // Build the base wiki/passthrough provider first.
        let base: Arc<dyn KnowledgeProvider> = if let Some(k_cfg) = &app_state.cfg.knowledge {
            tracing::info!(target: "h2ai.startup", "building knowledge provider (Bm25Wiki)");
            KnowledgeProviderFactory::build_provider(k_cfg).await
        } else {
            // When [knowledge] is absent, check if [constraint_wiki] corpus has a wiki/
            // subdirectory. If so, build a Bm25WikiProvider from it so the thinking loop
            // can surface domain articles during brainstorm iterations.
            let constraint_wiki_path = if let ConstraintWikiConfig::Fs { corpus_path, .. } =
                &app_state.cfg.constraint_wiki
            {
                let wiki_sub = std::path::Path::new(corpus_path).join("wiki");
                if wiki_sub.exists() {
                    Some(corpus_path.clone())
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(corpus_path) = constraint_wiki_path {
                tracing::info!(
                    target: "h2ai.startup",
                    corpus_path = %corpus_path,
                    "building knowledge provider from constraint wiki (no [knowledge] config)"
                );
                KnowledgeProviderFactory::build_from_constraint_corpus(std::path::Path::new(
                    &corpus_path,
                ))
                .await
            } else {
                tracing::info!(
                    target: "h2ai.startup",
                    "knowledge provider: passthrough (no [knowledge] config, no wiki dir)"
                );
                Arc::new(PassthroughProvider::new(
                    (*app_state.constraint_resolver).clone(),
                ))
            }
        };
        // Compose base wiki provider with the live skill_provider so extracted skills reach
        // the thinking loop on every knowledge query.
        CompositeProvider::new(
            vec![
                base,
                Arc::clone(&app_state.skill_provider) as Arc<dyn KnowledgeProvider>,
            ],
            app_state.cfg.knowledge_domain_scoping,
        )
    };

    app_state
        .load_tenant_state(&h2ai_types::identity::TenantId::default_tenant())
        .await;

    // Restore promoted shadow-audit domains persisted before last restart.
    {
        match app_state
            .nats
            .as_ref()
            .expect("NATS required in production")
            .get_shadow_promoted_domains()
            .await
        {
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

    // Spawn calibration in the background so the TCP listener can bind immediately.
    // Persisted calibration (loaded above) remains as fallback if the LLM is unreachable.
    // Tasks submitted before calibration completes are rejected with CalibrationRequired.
    {
        let single_family_warning = app_state.adapter_pool.len() == 1;
        if single_family_warning {
            match app_state.cfg.safety.family_constraint {
                FamilyConstraint::RequireDiverse => {
                    tracing::error!("adapter pool has only one adapter — RequireDiverse policy requires multiple adapters with distinct diversity IDs; aborting");
                    std::process::exit(1);
                }
                FamilyConstraint::SingleFamilyOk => {
                    tracing::warn!(
                        "single-adapter pool: correlated hallucination protection degraded"
                    );
                }
                FamilyConstraint::Disabled => {}
            }
        }
        let adapter_families: Vec<String> = app_state
            .adapter_pool
            .iter()
            .map(|a| format!("{:?}", a.kind()))
            .collect();
        let explorer_verification_family_match = false; // enforced via modulo IDs now

        let calibration_app_state = app_state.clone();
        let default_tenant = h2ai_types::identity::TenantId::default_tenant();
        let default_ts = calibration_app_state.tenant_state(&default_tenant);
        let had_calibration = default_ts.calibration.read().await.is_some();
        tracing::info!(target: "h2ai.startup", "spawning startup calibration in background");
        tokio::spawn(async move {
            crate::routes::calibrate::run_calibration_core(
                calibration_app_state.clone(),
                single_family_warning,
                explorer_verification_family_match,
                adapter_families,
                None,
            )
            .await;
            let default_ts = calibration_app_state
                .tenant_state(&h2ai_types::identity::TenantId::default_tenant());
            if default_ts.calibration.read().await.is_none() {
                if had_calibration {
                    tracing::warn!(target: "h2ai.startup", "startup calibration failed (LLM unreachable?); using persisted calibration");
                } else {
                    tracing::warn!(target: "h2ai.startup",
                        "startup calibration did not complete (LLM unreachable?); tasks will be rejected until POST /calibrate succeeds");
                }
            } else {
                tracing::info!(target: "h2ai.startup", "startup calibration complete — ready to accept tasks");
            }
        });
    }

    // Wire NATS dispatch when nats_dispatch_enabled = true in config.
    // When enabled, explorer slots are dispatched to TaoAgent processes via NATS
    // rather than calling the in-process LLM adapter.
    if app_state.cfg.nats_dispatch_enabled {
        let ttl = Duration::from_secs(app_state.cfg.nats_agent_ttl_secs);
        match NatsAgentProvider::new(
            app_state
                .nats_raw_client
                .clone()
                .expect("NATS required in production"),
            ttl,
        )
        .await
        {
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
        let default_oracle_ts =
            app_state.tenant_state(&h2ai_types::identity::TenantId::default_tenant());
        let accumulator = crate::oracle::OracleAccumulator {
            nats_raw: app_state
                .nats_raw_client
                .clone()
                .expect("NATS required in production"),
            nats_state: app_state
                .nats_concrete
                .clone()
                .expect("NATS required in production"),
            bandit: default_oracle_ts.bandit_state.clone(),
            metrics: app_state.metrics.clone(),
            oracle_window_size: app_state.cfg.oracle_window_size,
            oracle_ece_alert_threshold: app_state.cfg.oracle_ece_alert_threshold,
            oracle_pass_rate_floor: app_state.cfg.oracle_pass_rate_floor,
            calibration: default_oracle_ts.calibration.clone(),
            calibration_max_ensemble_size: app_state.cfg.calibration_max_ensemble_size,
        };
        tokio::spawn(accumulator.run());
    }

    // Spawn oracle worker — subscribes to h2ai.oracle.*.pending, runs tests via
    // ShellExecutor, and publishes OracleResultEvent to h2ai.oracle.results.
    {
        let oracle_worker = crate::oracle_worker::OracleWorker::new(
            app_state
                .nats_raw_client
                .clone()
                .expect("NATS required in production"),
        );
        tokio::spawn(oracle_worker.run());
    }

    // Wire shadow auditor accumulator — driven by process() calls from tasks.rs.
    if app_state.shadow_auditor_adapter.is_some() && app_state.cfg.safety.shadow_auditor.enabled {
        let shadow_state = Arc::new(app_state.clone());
        let acc = crate::shadow_auditor::ShadowAuditorAccumulator::new(shadow_state);
        app_state.shadow_accumulator = Some(Arc::new(tokio::sync::Mutex::new(acc)));
        tracing::info!(target: "h2ai.startup", "shadow auditor accumulator wired");
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
        .nest("/v1", routes::admin_router())
        // health/ready/metrics are always at root — never versioned
        .merge(routes::health_router())
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(&listen_addr).await.unwrap();
    tracing::info!(target: "h2ai.startup", addr = %listen_addr, "listening");
    axum::serve(listener, app).await.unwrap();
}
