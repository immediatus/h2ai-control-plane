use h2ai_config::H2AIConfig;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_orchestrator::payload_store::{MemoryPayloadStore, PayloadStore};
use h2ai_orchestrator::self_optimizer::TauSpreadEstimator;
use h2ai_orchestrator::session_journal::SessionJournal;
use h2ai_orchestrator::tao_loop::TaoMultiplierEstimator;
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_state::nats::NatsClient;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::events::CalibrationCompletedEvent;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

#[derive(Clone)]
pub struct AppState {
    pub nats: Arc<NatsClient>,
    pub cfg: Arc<H2AIConfig>,
    pub store: TaskStore,
    pub calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>>,
    pub journal: Arc<SessionJournal>,
    pub explorer_adapter: Arc<dyn IComputeAdapter>,
    /// Second explorer for USL timing Phase B. Defaults to `explorer_adapter` if not set.
    pub explorer2_adapter: Arc<dyn IComputeAdapter>,
    /// Scores proposals in Phase 3.5. Returns `{"score": float, "reason": "..."}`.
    pub verification_adapter: Arc<dyn IComputeAdapter>,
    /// Approves/rejects proposals in Phase 4. Returns `{"approved": bool, "reason": "..."}`.
    pub auditor_adapter: Arc<dyn IComputeAdapter>,
    /// Optional dedicated adapter for TaskProfile::Scoring. When None, uses explorer_adapter.
    pub scoring_adapter: Option<Arc<dyn IComputeAdapter>>,
    /// Limits concurrent task executions to cfg.max_concurrent_tasks.
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
    /// Content-addressed store for large context offloading in NATS dispatch mode.
    /// Default: in-process `MemoryPayloadStore`. Swap for `NatsObjectStoreBackend` in production.
    pub payload_store: Arc<dyn PayloadStore>,
}

impl AppState {
    pub fn new(
        nats: NatsClient,
        cfg: H2AIConfig,
        explorer_adapter: Arc<dyn IComputeAdapter>,
        auditor_adapter: Arc<dyn IComputeAdapter>,
    ) -> Self {
        // Cross-family bias warning: same model family for verification and exploration means
        // the LLM judge likely has self-preference bias toward its own output style.
        if std::mem::discriminant(explorer_adapter.kind())
            == std::mem::discriminant(auditor_adapter.kind())
        {
            tracing::warn!(
                target: "h2ai.verification",
                explorer_family = ?explorer_adapter.kind(),
                "verification_adapter and explorer_adapter are the same family \
                 — self-preference bias likely. Configure a different model family for verification."
            );
        }
        let nats = Arc::new(nats);
        let snapshot_interval = cfg.snapshot_interval_events;
        let journal =
            Arc::new(SessionJournal::new(nats.clone()).with_snapshot_interval(snapshot_interval));
        let max_tasks = cfg.max_concurrent_tasks;
        let tau_spread = cfg.calibration_tau_spread;
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
            tao_multiplier_estimator: Arc::new(RwLock::new(TaoMultiplierEstimator::new())),
            tau_spread_estimator: Arc::new(RwLock::new(TauSpreadEstimator::new(
                tau_spread[0],
                tau_spread[1],
            ))),
            payload_store: Arc::new(MemoryPayloadStore::new()),
        }
    }

    pub fn with_embedding_model(mut self, model: Arc<dyn EmbeddingModel>) -> Self {
        self.embedding_model = Some(model);
        self
    }

    /// Override the second explorer adapter (for USL timing Phase B).
    pub fn with_explorer2(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.explorer2_adapter = adapter;
        self
    }

    /// Build an `AdapterRegistry` from this state.
    ///
    /// The reasoning adapter is always `explorer_adapter`. The scoring adapter
    /// is used for `TaskProfile::Scoring` if configured; otherwise the explorer
    /// adapter handles all profiles.
    pub fn registry(&self) -> AdapterRegistry {
        let reg = AdapterRegistry::new(self.explorer_adapter.clone());
        match &self.scoring_adapter {
            Some(scoring) => reg.with_scoring(scoring.clone()),
            None => reg,
        }
    }
}
