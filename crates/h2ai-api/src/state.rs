use crate::tenant_registry::{TenantRegistry, TenantState};
use h2ai_config::H2AIConfig;
use h2ai_constraints::resolver::ConstraintResolver;
use h2ai_constraints::source::{FsConstraintIndex, FsConstraintStore};
use h2ai_context::embedding::EmbeddingModel;
use h2ai_knowledge::provider::KnowledgeProvider;
use h2ai_knowledge::skill_provider::{CompositeProvider, SkillProvider};
use h2ai_orchestrator::payload_store::{MemoryPayloadStore, PayloadStore};
use h2ai_orchestrator::session_journal::SessionJournal;
use h2ai_orchestrator::task_runner::{
    Decomposer, DefaultDecomposer, DefaultEngineRunner, DefaultThinkingLoopRunner, EngineRunner,
    ThinkingLoopRunner,
};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_state::backend::{NatsBackend, TaskDispatchBackend};
use h2ai_state::nats::NatsClient;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::identity::TenantId;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::{Notify, Semaphore};

/// Answer slot and wakeup notifier for a single pending clarification.
pub type ClarificationEntry = (Arc<Notify>, Arc<Mutex<Option<String>>>);

/// Shared application state injected into every Axum handler via `State<AppState>`.
///
/// All fields are `Clone`-cheap (wrapped in `Arc` or copy types) so the state can be
/// cloned per-request without allocating.  Mutable runtime state uses `Arc<RwLock<…>>`.
#[derive(Clone)]
pub struct AppState {
    /// NATS client used for event publishing and KV persistence.
    ///
    /// `None` in unit-test builds that use `new_for_tests()`.
    pub nats: Option<Arc<dyn NatsBackend>>,
    /// Raw async-nats client for oracle gate and thinking-loop oracle calls.
    /// Populated from `NatsClient.client` at startup; `None` in test builds.
    pub nats_raw_client: Option<async_nats::Client>,
    /// Narrow dispatch-only view of the same NATS connection, for `NatsDispatchConfig`.
    /// Coerced from `Arc<NatsClient>` at startup; `None` in test builds.
    pub task_dispatch_nats: Option<Arc<dyn TaskDispatchBackend>>,
    /// Concrete `NatsClient` for subsystems that need inherent methods not covered by traits
    /// (e.g. `OracleAccumulator.nats_state`). `None` in test builds.
    pub nats_concrete: Option<Arc<NatsClient>>,
    /// Resolved runtime configuration, loaded once at startup via `H2AIConfig::load_layered`.
    pub cfg: Arc<H2AIConfig>,
    /// In-memory task registry; tracks phase, status, and proposal counts per task ID.
    pub store: TaskStore,
    /// Per-tenant estimator registry. Single-tenant: always the "default" tenant.
    /// Multi-tenant: lazy-created on first task submission per tenant.
    pub tenant_registry: Arc<TenantRegistry>,
    /// Append-only event journal; persists task-event sequences to NATS for replay.
    pub journal: Arc<SessionJournal>,
    /// Ordered pool of compute adapters. Explorer slots are routed to `pool[diversity_id % pool.len()]`.
    /// With a single adapter all diversity IDs collapse to the same slot (graceful degradation).
    pub adapter_pool: Vec<Arc<dyn IComputeAdapter>>,
    /// Scores proposals in Phase 3.5. Returns `{"score": float, "reason": "..."}`.
    pub verification_adapter: Arc<dyn IComputeAdapter>,
    /// Approves/rejects proposals in Phase 4. Returns `{"approved": bool, "reason": "..."}`.
    pub auditor_adapter: Arc<dyn IComputeAdapter>,
    /// Optional dedicated adapter for `TaskProfile::Scoring`.  When `None`, the
    /// explorer adapter handles scoring tasks, which may share quota with exploration.
    pub scoring_adapter: Option<Arc<dyn IComputeAdapter>>,
    /// Optional shadow auditor for Phase 4 bias measurement.
    /// Must be from a different adapter family than `auditor_adapter`.
    /// `None` = shadow mode off regardless of config.
    pub shadow_auditor_adapter: Option<Arc<dyn IComputeAdapter>>,
    /// Domains currently in two-auditor AND-vote mode, maintained by `ShadowAuditorAccumulator`.
    /// Read at task dispatch time; updated in place by the accumulator via this shared ref.
    pub promoted_audit_domains: Arc<tokio::sync::RwLock<std::collections::HashSet<String>>>,
    /// Semaphore that caps concurrency at `cfg.max_concurrent_tasks`.
    /// `submit_task` acquires a permit before spawning; the permit is held until the
    /// engine finishes so back-pressure is applied at the HTTP layer.
    pub task_semaphore: Arc<Semaphore>,
    /// Optional embedding model for semantic similarity and Weiszfeld merge path.
    /// `None` disables embedding-dependent features (Weiszfeld, CG cosine agreement).
    pub embedding_model: Option<Arc<dyn EmbeddingModel>>,
    /// In-memory Prometheus metrics state.
    pub metrics: std::sync::Arc<tokio::sync::RwLock<crate::metrics::MetricsState>>,
    /// When `Some`, `tasks.rs` builds a `NatsDispatchConfig` so explorer slots are
    /// dispatched to `TaoAgent` via NATS instead of calling LLM adapters directly.
    pub agent_provider: Option<Arc<dyn AgentProvider>>,
    /// Content-addressed store for large task-context offloading.
    /// Passed through to `NatsDispatchConfig::payload_store`.
    pub payload_store: Arc<dyn PayloadStore>,
    /// Shadow auditor accumulator. When `Some`, `tasks.rs` calls `process()` after
    /// each engine run to update per-domain disagreement windows.
    pub shadow_accumulator:
        Option<Arc<tokio::sync::Mutex<crate::shadow_auditor::ShadowAuditorAccumulator>>>,
    /// Optional researcher adapter for C1 grounding.
    /// When `Some`, search-enabled slots get a pre-step and low-CV retries fetch contradiction evidence.
    /// When `None`, C1 falls back to hint-only without external web grounding.
    pub researcher_adapter: Option<Arc<dyn IComputeAdapter>>,
    /// Dedicated grounding chain gap researcher: DuckDuckGo web search + LLM distiller.
    /// `None` when researcher adapter is not configured.
    pub gap_research_chain:
        Option<std::sync::Arc<h2ai_orchestrator::grounding_chain::GapResearchChain>>,
    /// Pending human clarification waiters.
    /// Maps `task_id` → (Notify to wake the waiting engine, slot for the answer text).
    /// Populated by the engine when it suspends for clarification; resolved by POST /tasks/{id}/clarify.
    pub clarification_waiters: Arc<Mutex<HashMap<String, ClarificationEntry>>>,
    /// FS-backed constraint resolver built once at startup from `constraint_wiki` config.
    /// Used to load `Vec<ConstraintDoc>` for the engine's `constraint_corpus`.
    pub constraint_resolver: Arc<ConstraintResolver>,
    /// Knowledge provider for hierarchical constraint retrieval.
    /// Always a `CompositeProvider` that fans queries to both the wiki/passthrough provider
    /// and the `skill_provider`, so extracted skills reach the thinking loop.
    pub knowledge_provider: Arc<CompositeProvider>,
    /// Calibration drift monitor: tracks consensus_agreement_rate,
    /// fires DDM warnings and BOCPD changepoints, holds ORCA conformal margin.
    pub drift_monitor: std::sync::Arc<tokio::sync::Mutex<h2ai_autonomic::drift::DriftMonitor>>,
    /// Thinking loop stage runner. Real: DefaultThinkingLoopRunner. Test: MockThinkingLoopRunner.
    pub thinking_loop_runner: Arc<dyn ThinkingLoopRunner>,
    /// Decomposition stage runner. Real: DefaultDecomposer. Test: MockDecomposer.
    pub decomposer: Arc<dyn Decomposer>,
    /// Engine execution runner. Real: DefaultEngineRunner. Test: MockEngineRunner.
    pub engine_runner: Arc<dyn EngineRunner>,
    /// Live skill node store, populated by post_run after each resolved task.
    pub skill_provider: Arc<SkillProvider>,
}

impl AppState {
    /// Construct the initial `AppState` with an adapter pool and auditor adapter.
    ///
    /// `adapter_pool` must be non-empty; panics at startup if empty.
    /// `verification_adapter` is set to `auditor_adapter`.
    /// No family bias guard is applied here; that check lives in `main.rs` before
    /// the adapters are wired together.
    pub fn new(
        nats: NatsClient,
        cfg: H2AIConfig,
        adapter_pool: Vec<Arc<dyn IComputeAdapter>>,
        auditor_adapter: Arc<dyn IComputeAdapter>,
    ) -> Self {
        assert!(!adapter_pool.is_empty(), "adapter_pool must be non-empty");
        let raw_client = nats.client.clone();
        let nats_arc = Arc::new(nats);
        let snapshot_interval = cfg.snapshot_interval_events;
        let journal = Arc::new(
            SessionJournal::new(nats_arc.clone()).with_snapshot_interval(snapshot_interval),
        );
        let task_dispatch: Arc<dyn TaskDispatchBackend> = nats_arc.clone();
        let nats_concrete = nats_arc.clone();
        let nats: Arc<dyn NatsBackend> = nats_arc;
        let max_tasks = cfg.max_concurrent_tasks;
        let drift_monitor = std::sync::Arc::new(tokio::sync::Mutex::new(
            h2ai_autonomic::drift::DriftMonitor::from_config(&cfg),
        ));
        let (constraint_resolver, _) = {
            use h2ai_config::ConstraintWikiConfig;
            use h2ai_constraints::ambiguity::seed_scorecards;
            let (resolver, flagged) = match &cfg.constraint_wiki {
                ConstraintWikiConfig::Fs { corpus_path, .. } => {
                    let (index, store) = FsConstraintStore::load(corpus_path).unwrap_or_else(|e| {
                        tracing::warn!(
                            error = %e,
                            corpus_path = %corpus_path,
                            "constraint corpus load failed; using empty corpus"
                        );
                        (
                            FsConstraintIndex::from_docs(&[]),
                            FsConstraintStore::from_docs(vec![]),
                        )
                    });
                    // Pre-validate corpus for static ambiguity at startup so operators see
                    // issues before the first task runs, not buried inside a MAPE-K wave.
                    let flagged = if cfg.ambiguity_detection.enabled {
                        let docs = store.all_docs_sorted();
                        let cards = seed_scorecards(&docs, &cfg.ambiguity_detection);
                        let mut ids = std::collections::HashSet::new();
                        for ((cid, check_idx), card) in &cards {
                            let evidence_str = card
                                .evidence
                                .iter()
                                .map(|e| e.to_string())
                                .collect::<Vec<_>>()
                                .join("; ");
                            if card.score >= cfg.ambiguity_detection.score_threshold {
                                tracing::error!(
                                    target: "h2ai.constraints",
                                    constraint_id = %cid,
                                    check_idx = %check_idx,
                                    score = card.score,
                                    evidence = %evidence_str,
                                    "CONSTRAINT NON-TRUSTABLE: ambiguity score crossed threshold \
                                     at corpus load — verdicts for this check are unreliable \
                                     and auto-repair is suppressed"
                                );
                                ids.insert(cid.clone());
                            } else {
                                tracing::warn!(
                                    target: "h2ai.constraints",
                                    constraint_id = %cid,
                                    check_idx = %check_idx,
                                    score = card.score,
                                    evidence = %evidence_str,
                                    "constraint ambiguity evidence detected at corpus load"
                                );
                            }
                        }
                        ids
                    } else {
                        std::collections::HashSet::new()
                    };
                    (
                        ConstraintResolver::new(Arc::new(index), Arc::new(store)),
                        flagged,
                    )
                }
                ConstraintWikiConfig::Disabled => (
                    ConstraintResolver::new(
                        Arc::new(FsConstraintIndex::from_docs(&[])),
                        Arc::new(FsConstraintStore::from_docs(vec![])),
                    ),
                    std::collections::HashSet::new(),
                ),
            };
            (Arc::new(resolver), Arc::new(flagged))
        };
        // Build skill_provider before the struct literal so we can Arc::clone it into the
        // composite.  main.rs replaces knowledge_provider with a real wiki+skill composite.
        let skill_provider = SkillProvider::new();
        let knowledge_provider = {
            use h2ai_knowledge::provider::PassthroughProvider;
            let base: Arc<dyn KnowledgeProvider> = Arc::new(PassthroughProvider::new_from_path(
                std::path::Path::new("."),
            ));
            CompositeProvider::new(
                vec![
                    base,
                    Arc::clone(&skill_provider) as Arc<dyn KnowledgeProvider>,
                ],
                cfg.knowledge_domain_scoping,
            )
        };
        Self {
            nats: Some(nats),
            nats_raw_client: Some(raw_client),
            task_dispatch_nats: Some(task_dispatch),
            nats_concrete: Some(nats_concrete),
            cfg: Arc::new(cfg),
            store: TaskStore::new(),
            tenant_registry: Arc::new(TenantRegistry::new()),
            journal,
            adapter_pool,
            verification_adapter: auditor_adapter.clone(),
            auditor_adapter,
            scoring_adapter: None,
            shadow_auditor_adapter: None,
            promoted_audit_domains: Arc::new(tokio::sync::RwLock::new(
                std::collections::HashSet::new(),
            )),
            task_semaphore: Arc::new(Semaphore::new(max_tasks)),
            embedding_model: None,
            metrics: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::metrics::MetricsState::default(),
            )),
            agent_provider: None,
            payload_store: Arc::new(MemoryPayloadStore::new()),
            shadow_accumulator: None,
            researcher_adapter: None,
            gap_research_chain: None,
            clarification_waiters: Arc::new(Mutex::new(HashMap::new())),
            constraint_resolver,
            skill_provider,
            knowledge_provider,
            drift_monitor,
            thinking_loop_runner: Arc::new(DefaultThinkingLoopRunner),
            decomposer: Arc::new(DefaultDecomposer),
            engine_runner: Arc::new(DefaultEngineRunner),
        }
    }

    /// Construct a minimal `AppState` for unit tests. Does not connect to NATS.
    #[allow(dead_code)]
    pub fn new_for_tests(
        cfg: H2AIConfig,
        adapter_pool: Vec<Arc<dyn IComputeAdapter>>,
        auditor_adapter: Arc<dyn IComputeAdapter>,
    ) -> Self {
        use h2ai_knowledge::provider::PassthroughProvider;
        use h2ai_orchestrator::payload_store::MemoryPayloadStore;
        assert!(!adapter_pool.is_empty(), "adapter_pool must be non-empty");
        let max_tasks = cfg.max_concurrent_tasks;
        let drift_monitor = std::sync::Arc::new(tokio::sync::Mutex::new(
            h2ai_autonomic::drift::DriftMonitor::from_config(&cfg),
        ));
        let constraint_resolver = Arc::new(ConstraintResolver::new(
            Arc::new(FsConstraintIndex::from_docs(&[])),
            Arc::new(FsConstraintStore::from_docs(vec![])),
        ));
        let skill_provider = SkillProvider::new();
        let knowledge_provider = {
            let base: Arc<dyn KnowledgeProvider> = Arc::new(PassthroughProvider::new_from_path(
                std::path::Path::new("."),
            ));
            CompositeProvider::new(
                vec![
                    base,
                    Arc::clone(&skill_provider) as Arc<dyn KnowledgeProvider>,
                ],
                cfg.knowledge_domain_scoping,
            )
        };
        Self {
            nats: None,
            nats_raw_client: None,
            task_dispatch_nats: None,
            nats_concrete: None,
            cfg: Arc::new(cfg),
            store: TaskStore::new(),
            tenant_registry: Arc::new(TenantRegistry::new()),
            journal: Arc::new(SessionJournal::new_noop()),
            adapter_pool,
            verification_adapter: auditor_adapter.clone(),
            auditor_adapter,
            scoring_adapter: None,
            shadow_auditor_adapter: None,
            promoted_audit_domains: Arc::new(tokio::sync::RwLock::new(
                std::collections::HashSet::new(),
            )),
            task_semaphore: Arc::new(Semaphore::new(max_tasks)),
            embedding_model: None,
            metrics: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::metrics::MetricsState::default(),
            )),
            agent_provider: None,
            payload_store: Arc::new(MemoryPayloadStore::new()),
            shadow_accumulator: None,
            researcher_adapter: None,
            gap_research_chain: None,
            clarification_waiters: Arc::new(Mutex::new(HashMap::new())),
            constraint_resolver,
            skill_provider,
            knowledge_provider,
            drift_monitor,
            thinking_loop_runner: Arc::new(DefaultThinkingLoopRunner),
            decomposer: Arc::new(DefaultDecomposer),
            engine_runner: Arc::new(DefaultEngineRunner),
        }
    }

    /// Return isolated estimator state for the given tenant, creating it on first access.
    ///
    /// For single-tenant deployments: always call with `TenantId::default_tenant()`.
    #[must_use]
    pub fn tenant_state(&self, id: &TenantId) -> Arc<TenantState> {
        self.tenant_registry.get_or_create(id, &self.cfg)
    }

    #[cfg(feature = "fastembed-embed")]
    #[allow(dead_code)]
    pub fn with_embedding_model(mut self, model: Arc<dyn EmbeddingModel>) -> Self {
        self.embedding_model = Some(model);
        self
    }

    /// Configure a shadow auditor adapter for Phase 4 disagreement measurement.
    ///
    /// The shadow adapter MUST be from a different family than `auditor_adapter`.
    /// Callers are responsible for the family check — this method stores whatever is passed.
    pub fn with_shadow_auditor(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.shadow_auditor_adapter = Some(adapter);
        self
    }

    /// Override the agent provider used for NATS-based explorer dispatch.
    ///
    /// When `None` (the default), explorer slots call the in-process LLM adapter directly.
    /// When `Some`, each explorer slot is dispatched via NATS to a `TaoAgent` process;
    /// `tasks.rs` builds a `NatsDispatchConfig` from this provider at task submission time.
    /// Must be set before the server starts accepting requests.
    pub fn with_agent_provider(mut self, provider: Arc<dyn AgentProvider>) -> Self {
        self.agent_provider = Some(provider);
        self
    }

    /// Override the content-addressed payload store used for large context offloading.
    ///
    /// Defaults to `MemoryPayloadStore` (in-process, zero-dependency). Supply a
    /// NATS-backed or object-store-backed implementation in production when contexts
    /// exceed the inline threshold and must survive process restarts.
    #[allow(dead_code)]
    pub fn with_payload_store(mut self, store: Arc<dyn PayloadStore>) -> Self {
        self.payload_store = store;
        self
    }

    /// Build an [`AdapterRegistry`] from this state.
    ///
    /// The reasoning adapter is always `adapter_pool[0]`.  When `scoring_adapter` is
    /// `Some`, `TaskProfile::Scoring` requests are routed to it instead of the pool,
    /// preventing scoring load from competing with exploration quota.  All other profiles
    /// fall through to `adapter_pool[0]`.
    #[must_use]
    pub fn registry(&self) -> AdapterRegistry {
        let reg = AdapterRegistry::new(self.adapter_pool[0].clone());
        match &self.scoring_adapter {
            Some(scoring) => reg.with_scoring(scoring.clone()),
            None => reg,
        }
    }

    /// Seed calibration for a non-default tenant from the default tenant's calibration.
    ///
    /// Called at task submission time for non-default tenants. If the tenant has no
    /// calibration yet and the default tenant does, copy the default calibration so the
    /// new tenant can submit tasks immediately without a separate calibration run.
    /// No-op if: the tenant already has calibration, the tenant IS the default, or
    /// the default tenant has no calibration.
    pub async fn seed_calibration_from_default_if_needed(&self, tenant_id: &TenantId) {
        if *tenant_id == TenantId::default_tenant() {
            return;
        }
        let ts = self.tenant_state(tenant_id);
        {
            let cal = ts.calibration.read().await;
            if cal.is_some() {
                return;
            }
        }
        let default_ts = self.tenant_state(&TenantId::default_tenant());
        let default_cal = default_ts.calibration.read().await.clone();
        if let Some(cal) = default_cal {
            *ts.calibration.write().await = Some(cal);
            tracing::info!(
                target: "h2ai.calibration",
                %tenant_id,
                "seeded calibration from default tenant"
            );
        }
    }

    /// Load persisted estimator state for `tenant_id` from NATS KV into the registry.
    ///
    /// Called at startup for the "default" tenant. New tenants start from cold defaults
    /// and learn from their own tasks. Silently falls back to defaults on any error.
    pub async fn load_tenant_state(&self, tenant_id: &TenantId) {
        use h2ai_orchestrator::bandit::BanditState;
        use h2ai_orchestrator::tao_loop::TaoMultiplierEstimator;
        use h2ai_types::events::CalibrationSource;

        let Some(nats) = &self.nats else {
            return;
        };
        let ts = self.tenant_state(tenant_id);

        // TAO multiplier
        match nats.get_tao_estimator_state(tenant_id).await {
            Ok(Some((ema, count))) => {
                let alpha = self.cfg.tao_estimator_ema_alpha;
                // serde round-trip: TaoMultiplierEstimator serializes only {ema, count};
                // alpha/warmup are #[serde(skip)] and restored via with_alpha().
                let json = serde_json::json!({ "ema": ema, "count": count });
                match serde_json::from_value::<TaoMultiplierEstimator>(json) {
                    Ok(est) => *ts.tao_multiplier_estimator.write().await = est.with_alpha(alpha),
                    Err(e) => {
                        tracing::warn!(error = %e, "tao_estimator deserialize failed; using default");
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(error = %e, "tao_estimator load from NATS failed; using default");
            }
        }

        // Bandit state
        match nats.get_bandit_state(tenant_id).await {
            Ok(Some(bytes)) => match serde_json::from_slice::<BanditState>(&bytes) {
                Ok(state) => *ts.bandit_state.write().await = state,
                Err(e) => {
                    tracing::warn!(error = %e, "bandit state deserialize failed; using default");
                }
            },
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(error = %e, "bandit state load from NATS failed; using default");
            }
        }

        // Skill nodes
        {
            use h2ai_knowledge::types::KnowledgeNode;
            use h2ai_state::backend::SkillStore;
            match nats.get_skill_nodes(tenant_id).await {
                Ok(bytes) if !bytes.is_empty() => {
                    match serde_json::from_slice::<Vec<KnowledgeNode>>(&bytes) {
                        Ok(nodes) => {
                            let n = nodes.len();
                            self.skill_provider.push_all(nodes);
                            tracing::info!(target: "h2ai.startup", n, "skill nodes restored from NATS KV");
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "skill nodes deserialize failed; starting empty");
                        }
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "skill nodes load from NATS failed; starting empty");
                }
            }
        }

        // Calibration snapshot
        // Calibration is currently a single global dataset per server — not per-tenant.
        // A per-tenant calibration API is a future enhancement.
        match nats.get_calibration().await {
            Ok(Some(cal)) => {
                let label = match cal.calibration_source {
                    CalibrationSource::Measured => "measured",
                    CalibrationSource::PartialFit => "partial_fit",
                    CalibrationSource::SyntheticPriors => "synthetic_priors",
                };
                {
                    let mut metrics = self.metrics.write().await;
                    metrics.calibration_source_label = label.to_string();
                }
                if cal.calibration_source == CalibrationSource::SyntheticPriors {
                    tracing::warn!(
                        target: "h2ai.calibration",
                        "stored calibration uses SyntheticPriors. Run POST /calibrate to replace."
                    );
                }
                *ts.calibration.write().await = Some(cal);
                tracing::info!(target: "h2ai.calibration", "calibration restored from NATS KV");
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(target: "h2ai.calibration", error = %e, "calibration load from NATS failed");
            }
        }
    }
}
