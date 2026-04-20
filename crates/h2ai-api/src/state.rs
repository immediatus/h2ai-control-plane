use h2ai_config::H2AIConfig;
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_state::nats::NatsClient;
use h2ai_types::events::CalibrationCompletedEvent;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub nats: Arc<NatsClient>,
    pub cfg: Arc<H2AIConfig>,
    pub store: TaskStore,
    pub calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>>,
}

impl AppState {
    pub fn new(nats: NatsClient, cfg: H2AIConfig) -> Self {
        Self {
            nats: Arc::new(nats),
            cfg: Arc::new(cfg),
            store: TaskStore::new(),
            calibration: Arc::new(RwLock::new(None)),
        }
    }
}
