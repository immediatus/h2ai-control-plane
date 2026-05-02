use h2ai_config::H2AIConfig;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_orchestrator::self_optimizer::TauSpreadEstimator;
use h2ai_orchestrator::session_journal::SessionJournal;
use h2ai_orchestrator::tao_loop::TaoMultiplierEstimator;
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_state::nats::NatsClient;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::events::CalibrationCompletedEvent;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

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
    /// Primary LLM adapter used for Phase 1 exploration and as the default for all profiles.
    pub explorer_adapter: Arc<dyn IComputeAdapter>,
    /// Second explorer for USL timing Phase B. Defaults to `explorer_adapter` if not set.
    pub explorer2_adapter: Arc<dyn IComputeAdapter>,
    /// Scores proposals in Phase 3.5. Returns `{"score": float, "reason": "..."}`.
    pub verification_adapter: Arc<dyn IComputeAdapter>,
    /// Approves/rejects proposals in Phase 4. Returns `{"approved": bool, "reason": "..."}`.
    pub auditor_adapter: Arc<dyn IComputeAdapter>,
    /// Optional dedicated adapter for `TaskProfile::Scoring`.  When `None`, the
    /// explorer adapter handles scoring tasks, which may share quota with exploration.
    pub scoring_adapter: Option<Arc<dyn IComputeAdapter>>,
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
}

impl AppState {
    /// Construct the initial `AppState` with a single explorer and auditor adapter.
    ///
    /// Sets `explorer2_adapter` and `verification_adapter` to the same values as
    /// `explorer_adapter` and `auditor_adapter` respectively; call [`with_explorer2`][Self::with_explorer2]
    /// afterwards when the runtime needs a distinct second-explorer endpoint.
    /// No same-family bias guard is applied here; that check lives in `main.rs` before
    /// the adapters are wired together.
    pub fn new(
        nats: NatsClient,
        cfg: H2AIConfig,
        explorer_adapter: Arc<dyn IComputeAdapter>,
        auditor_adapter: Arc<dyn IComputeAdapter>,
    ) -> Self {
        let nats = Arc::new(nats);
        let snapshot_interval = cfg.snapshot_interval_events;
        let journal =
            Arc::new(SessionJournal::new(nats.clone()).with_snapshot_interval(snapshot_interval));
        let max_tasks = cfg.max_concurrent_tasks;
        let tau_spread = cfg.calibration_tau_spread;
        let tao_ema_alpha = cfg.tao_estimator_ema_alpha;
        Self {
            nats,
            cfg: Arc::new(cfg),
            store: TaskStore::new(),
            calibration: Arc::new(RwLock::new(None)),
            journal,
            explorer2_adapter: explorer_adapter.clone(),
            explorer_adapter,
            verification_adapter: auditor_adapter.clone(),
            auditor_adapter,
            scoring_adapter: None,
            task_semaphore: Arc::new(Semaphore::new(max_tasks)),
            embedding_model: None,
            tao_multiplier_estimator: Arc::new(RwLock::new(
                TaoMultiplierEstimator::new_with_alpha(tao_ema_alpha),
            )),
            tau_spread_estimator: Arc::new(RwLock::new(TauSpreadEstimator::new(
                tau_spread[0],
                tau_spread[1],
            ))),
        }
    }

    #[cfg(feature = "fastembed-embed")]
    pub fn with_embedding_model(mut self, model: Arc<dyn EmbeddingModel>) -> Self {
        self.embedding_model = Some(model);
        self
    }

    /// Override the second explorer adapter used in USL timing Phase B.
    ///
    /// USL Phase B measures inter-adapter latency to separate the σ (contention) and
    /// κ (coherency) coefficients; using an adapter from the same provider family as
    /// `explorer_adapter` would conflate the two signals.  Supply a distinct endpoint
    /// here to ensure the calibration sees genuine cross-adapter round-trip cost.
    pub fn with_explorer2(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.explorer2_adapter = adapter;
        self
    }

    /// Build an [`AdapterRegistry`] from this state.
    ///
    /// The reasoning adapter is always `explorer_adapter`.  When `scoring_adapter` is
    /// `Some`, `TaskProfile::Scoring` requests are routed to it instead of the explorer,
    /// preventing scoring load from competing with exploration quota.  All other profiles
    /// fall through to the explorer adapter.
    pub fn registry(&self) -> AdapterRegistry {
        let reg = AdapterRegistry::new(self.explorer_adapter.clone());
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
}
