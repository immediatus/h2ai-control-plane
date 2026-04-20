use h2ai_config::H2AIConfig;
use h2ai_orchestrator::session_journal::SessionJournal;
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
    /// Scores proposals in Phase 3.5. Returns `{"score": float, "reason": "..."}`.
    pub verification_adapter: Arc<dyn IComputeAdapter>,
    /// Approves/rejects proposals in Phase 4. Returns `{"approved": bool, "reason": "..."}`.
    pub auditor_adapter: Arc<dyn IComputeAdapter>,
    /// Optional dedicated adapter for TaskProfile::Scoring (semantic similarity, short JSON
    /// scoring). When None, the explorer adapter is used for all profiles.
    pub scoring_adapter: Option<Arc<dyn IComputeAdapter>>,
    /// Limits concurrent task executions to cfg.max_concurrent_tasks.
    pub task_semaphore: Arc<Semaphore>,
}

impl AppState {
    pub fn new(
        nats: NatsClient,
        cfg: H2AIConfig,
        explorer_adapter: Arc<dyn IComputeAdapter>,
        auditor_adapter: Arc<dyn IComputeAdapter>,
    ) -> Self {
        let nats = Arc::new(nats);
        let journal = Arc::new(SessionJournal::new(nats.clone()));
        let max_tasks = cfg.max_concurrent_tasks;
        Self {
            nats,
            cfg: Arc::new(cfg),
            store: TaskStore::new(),
            calibration: Arc::new(RwLock::new(None)),
            journal,
            explorer_adapter,
            verification_adapter: auditor_adapter.clone(),
            auditor_adapter,
            scoring_adapter: None,
            task_semaphore: Arc::new(Semaphore::new(max_tasks)),
        }
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
