use crate::constraint_source::{NatsConstraintIndex, NatsConstraintStore};
use crate::rho_ema::RhoEmaState;
use h2ai_config::H2AIConfig;
use h2ai_constraints::resolver::ConstraintResolver;
use h2ai_constraints::source::{FsConstraintIndex, FsConstraintStore};
use h2ai_context::embedding::EmbeddingModel;
use h2ai_orchestrator::bandit::BanditState;
use h2ai_orchestrator::payload_store::{MemoryPayloadStore, PayloadStore};
use h2ai_orchestrator::self_optimizer::TauSpreadEstimator;
use h2ai_orchestrator::session_journal::SessionJournal;
use h2ai_orchestrator::tao_loop::TaoMultiplierEstimator;
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_state::nats::NatsClient;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::events::CalibrationCompletedEvent;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::{Notify, RwLock, Semaphore};

/// Answer slot and wakeup notifier for a single pending clarification.
pub type ClarificationEntry = (Arc<Notify>, Arc<Mutex<Option<String>>>);

/// Shared application state injected into every Axum handler via `State<AppState>`.
///
/// All fields are `Clone`-cheap (wrapped in `Arc` or copy types) so the state can be
/// cloned per-request without allocating.  Mutable runtime state uses `Arc<RwLock<…>>`.
#[derive(Clone)]
pub struct AppState {
    /// NATS client used for event publishing and KV persistence.
    pub nats: Arc<NatsClient>,
    /// Resolved runtime configuration, loaded once at startup via `H2AIConfig::load_layered`.
    pub cfg: Arc<H2AIConfig>,
    /// In-memory task registry; tracks phase, status, and proposal counts per task ID.
    pub store: TaskStore,
    /// Most recent calibration snapshot; populated by the calibration background loop.
    /// Handlers return `ApiError::CalibrationRequired` when this is `None`.
    pub calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>>,
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
    /// Optional shadow auditor for Phase 4 bias measurement (GAP-C2).
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
    /// Online estimator for the TAO loop per-turn quality factor. Accumulates
    /// q_after/q_before ratios from Tier 1 verified tasks; returns heuristic 0.6 until 20 samples.
    pub tao_multiplier_estimator: Arc<RwLock<TaoMultiplierEstimator>>,
    /// EMA-tracked τ spread hint. Updated when SelfOptimizer suggests τ adjustments on
    /// wasteful-but-successful tasks. User-specified τ bounds in task manifests always override.
    pub tau_spread_estimator: Arc<RwLock<TauSpreadEstimator>>,
    /// Thompson Sampling bandit for adaptive N selection. Learns optimal ensemble size
    /// from task outcomes across runs. Persisted to NATS KV `H2AI_ESTIMATOR/bandit_state`.
    pub bandit_state: Arc<RwLock<BanditState>>,
    /// In-memory Prometheus metrics state.
    pub metrics: std::sync::Arc<tokio::sync::RwLock<crate::metrics::MetricsState>>,
    /// When `Some`, `tasks.rs` builds a `NatsDispatchConfig` so explorer slots are
    /// dispatched to TaoAgent via NATS instead of calling LLM adapters directly.
    pub agent_provider: Option<Arc<dyn AgentProvider>>,
    /// Content-addressed store for large task-context offloading.
    /// Passed through to `NatsDispatchConfig::payload_store`.
    pub payload_store: Arc<dyn PayloadStore>,
    /// Shadow auditor accumulator. When `Some`, `tasks.rs` calls `process()` after
    /// each engine run to update per-domain disagreement windows.
    pub shadow_accumulator:
        Option<Arc<tokio::sync::Mutex<crate::shadow_auditor::ShadowAuditorAccumulator>>>,
    /// Optional researcher adapter for C1 grounding (GAP-C1).
    /// When `Some`, search-enabled slots get a pre-step and low-CV retries fetch contradiction evidence.
    /// When `None`, C1 falls back to hint-only without external web grounding.
    pub researcher_adapter: Option<Arc<dyn IComputeAdapter>>,
    /// SRANI adaptive EMA state: (ema_cfi, count).
    /// Loaded from NATS KV at startup; updated by tasks.rs after each engine run.
    pub srani_state: Arc<tokio::sync::RwLock<(f64, usize)>>,
    /// SRANI grounding chain: spec anchor → LLM researcher → web search escalation.
    /// Built at startup from available adapters; `None` = spec anchor only (inline, no chain).
    pub srani_grounding_chain:
        Option<std::sync::Arc<h2ai_orchestrator::srani_grounding::SraniGroundingChain>>,
    /// Online ρ EMA tracker for INNOVATION-3 (GAP-A3).
    /// Updated after each engine run with pairwise centered score products.
    pub rho_ema: Arc<RwLock<RhoEmaState>>,
    /// Pending human clarification waiters.
    /// Maps task_id → (Notify to wake the waiting engine, slot for the answer text).
    /// Populated by the engine when it suspends for clarification; resolved by POST /tasks/{id}/clarify.
    pub clarification_waiters: Arc<Mutex<HashMap<String, ClarificationEntry>>>,
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
        let nats = Arc::new(nats);
        let snapshot_interval = cfg.snapshot_interval_events;
        let journal =
            Arc::new(SessionJournal::new(nats.clone()).with_snapshot_interval(snapshot_interval));
        let max_tasks = cfg.max_concurrent_tasks;
        let tau_spread = cfg.calibration_tau_spread;
        let tao_ema_alpha = cfg.tao_estimator_ema_alpha;
        let tao_warmup = cfg.tao_estimator_warmup;
        let n_max_init = cfg.bandit_n_max_initial;
        let bandit_n_max_arms = cfg.bandit_n_max_arms;
        let bandit_prior_sigma = cfg.bandit_prior_sigma;
        let bandit_prior_strength = cfg.bandit_prior_strength;
        Self {
            nats,
            cfg: Arc::new(cfg),
            store: TaskStore::new(),
            calibration: Arc::new(RwLock::new(None)),
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
            tao_multiplier_estimator: Arc::new(RwLock::new(
                TaoMultiplierEstimator::new_with_alpha(tao_ema_alpha).with_warmup(tao_warmup),
            )),
            tau_spread_estimator: Arc::new(RwLock::new(TauSpreadEstimator::new(
                tau_spread[0],
                tau_spread[1],
            ))),
            bandit_state: Arc::new(RwLock::new(BanditState::new(
                n_max_init,
                0,
                bandit_n_max_arms,
                bandit_prior_sigma,
                bandit_prior_strength,
            ))),
            metrics: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::metrics::MetricsState::default(),
            )),
            agent_provider: None,
            payload_store: Arc::new(MemoryPayloadStore::new()),
            shadow_accumulator: None,
            researcher_adapter: None,
            srani_state: Arc::new(tokio::sync::RwLock::new((0.0, 0))),
            srani_grounding_chain: None,
            rho_ema: Arc::new(RwLock::new(RhoEmaState::default())),
            clarification_waiters: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    #[cfg(feature = "fastembed-embed")]
    #[allow(dead_code)]
    pub fn with_embedding_model(mut self, model: Arc<dyn EmbeddingModel>) -> Self {
        self.embedding_model = Some(model);
        self
    }

    /// Configure a shadow auditor adapter for Phase 4 disagreement measurement (GAP-C2).
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
    /// When `Some`, each explorer slot is dispatched via NATS to a TaoAgent process;
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
    pub fn registry(&self) -> AdapterRegistry {
        let reg = AdapterRegistry::new(self.adapter_pool[0].clone());
        match &self.scoring_adapter {
            Some(scoring) => reg.with_scoring(scoring.clone()),
            None => reg,
        }
    }

    /// Load persisted TaoMultiplierEstimator state from NATS, if available.
    /// Called once after construction. Silently falls back to the in-memory default on error.
    pub async fn load_tao_estimator(&self) {
        match self.nats.get_tao_estimator_state().await {
            Ok(Some((ema, count))) => {
                let alpha = self.cfg.tao_estimator_ema_alpha;
                // Reconstruct by deserializing the persisted fields, then restore alpha.
                // warmup_sum is skipped in serde, so warm-up cannot be resumed mid-stream —
                // documented in TaoMultiplierEstimator: put only persists when count >= 20.
                let json = serde_json::json!({ "ema": ema, "count": count });
                match serde_json::from_value::<h2ai_orchestrator::tao_loop::TaoMultiplierEstimator>(
                    json,
                ) {
                    Ok(est) => {
                        *self.tao_multiplier_estimator.write().await = est.with_alpha(alpha);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to deserialize tao_estimator; using default")
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(error = %e, "failed to load tao_estimator from NATS; using default")
            }
        }
    }

    /// Load persisted bandit state from NATS KV on startup.
    /// No-op on first run (no entry yet) or on deserialization failure (uses default prior).
    pub async fn load_bandit_state(&self) {
        match self.nats.get_bandit_state().await {
            Ok(Some(bytes)) => match serde_json::from_slice::<BanditState>(&bytes) {
                Ok(state) => {
                    *self.bandit_state.write().await = state;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to deserialize bandit state; using default prior")
                }
            },
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(error = %e, "failed to load bandit state from NATS; using default prior")
            }
        }
    }

    /// Load persisted SRANI adaptive EMA state from NATS KV on startup.
    /// On first run (no entry) or deserialization failure, falls back to cold-start defaults.
    pub async fn load_srani_state(&self) {
        match self.nats.get_srani_state().await {
            Ok(Some((ema_cfi, count))) => {
                *self.srani_state.write().await = (ema_cfi, count);
                tracing::info!(
                    target: "h2ai.startup",
                    ema_cfi,
                    count,
                    "srani adaptive state restored from NATS KV"
                );
            }
            Ok(None) => {
                let midpoint = self.cfg.srani.cold_start_midpoint();
                *self.srani_state.write().await = (midpoint, 0);
            }
            Err(e) => {
                tracing::warn!(
                    target: "h2ai.startup",
                    error = %e,
                    "failed to load srani state from NATS; using cold-start defaults"
                );
                let midpoint = self.cfg.srani.cold_start_midpoint();
                *self.srani_state.write().await = (midpoint, 0);
            }
        }
    }

    /// Restore persisted calibration from NATS KV into the in-memory field.
    ///
    /// Called once at startup. When calibration exists in NATS the server can accept
    /// tasks immediately without requiring a POST /calibrate. When nothing is stored
    /// the field stays `None` and the caller should run eager calibration before opening
    /// for traffic.
    pub async fn load_calibration(&self) {
        match self.nats.get_calibration().await {
            Ok(Some(cal)) => {
                let label = match cal.calibration_source {
                    h2ai_types::events::CalibrationSource::Measured => "measured",
                    h2ai_types::events::CalibrationSource::PartialFit => "partial_fit",
                    h2ai_types::events::CalibrationSource::SyntheticPriors => "synthetic_priors",
                };
                {
                    let mut metrics = self.metrics.write().await;
                    metrics.calibration_source_label = label.to_string();
                }
                if cal.calibration_source == h2ai_types::events::CalibrationSource::SyntheticPriors
                {
                    tracing::warn!(
                        target: "h2ai.calibration",
                        "stored calibration uses SyntheticPriors — N_max based on config defaults. \
                         Run POST /calibrate to replace with real measurements."
                    );
                }
                *self.calibration.write().await = Some(cal);
                tracing::info!(target: "h2ai.calibration", "calibration restored from NATS KV");
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(target: "h2ai.calibration", error = %e, "failed to load calibration from NATS")
            }
        }
    }

    /// Build a per-request [`ConstraintResolver`] for the current config.
    ///
    /// When `constraint_wiki.enabled = true`: NATS-backed lazy resolver (never bulk-loads).
    /// When `constraint_wiki.enabled = false`: FS-backed resolver loaded from `corpus_path`.
    pub async fn constraint_resolver(&self) -> ConstraintResolver {
        if self.cfg.constraint_wiki.enabled {
            ConstraintResolver::new(
                Arc::new(NatsConstraintIndex::new(self.nats.clone())),
                Arc::new(NatsConstraintStore::new(self.nats.clone())),
            )
        } else {
            let corpus_path = self
                .cfg
                .constraint_wiki
                .corpus_path
                .clone()
                .unwrap_or_else(|| "/constraints".to_string());
            let (index, store) = FsConstraintStore::load(&corpus_path)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        error = %e,
                        "constraint corpus load failed at path {corpus_path}; using empty corpus"
                    );
                    let store = FsConstraintStore::from_docs(vec![]);
                    let index = FsConstraintIndex::from_docs(&[]);
                    (index, store)
                });
            ConstraintResolver::new(Arc::new(index), Arc::new(store))
        }
    }
}
