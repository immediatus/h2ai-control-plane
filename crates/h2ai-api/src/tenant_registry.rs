use crate::rho_ema::RhoEmaState;
use dashmap::DashMap;
use h2ai_config::H2AIConfig;
use h2ai_orchestrator::bandit::BanditState;
use h2ai_orchestrator::self_optimizer::TauSpreadEstimator;
use h2ai_orchestrator::tao_loop::TaoMultiplierEstimator;
use h2ai_types::events::CalibrationCompletedEvent;
use h2ai_types::identity::TenantId;
use std::sync::Arc;
use tokio::sync::RwLock;

/// All per-tenant mutable runtime estimators.
///
/// Created lazily on first use of a tenant ID. Single-tenant deployments always access
/// the "default" tenant — zero behavioral change from pre-registry code.
pub struct TenantState {
    pub calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>>,
    pub tao_multiplier_estimator: Arc<RwLock<TaoMultiplierEstimator>>,
    pub tau_spread_estimator: Arc<RwLock<TauSpreadEstimator>>,
    pub bandit_state: Arc<RwLock<BanditState>>,
    /// `(ema_cfi, count)` — SRANI adaptive EMA: current EMA of CFI scores and observation count.
    pub srani_state: Arc<RwLock<(f64, usize)>>,
    pub rho_ema: Arc<RwLock<RhoEmaState>>,
}

impl TenantState {
    /// Construct a cold-start `TenantState` from config defaults.
    ///
    /// All estimators start at their prior values; `load_tenant_state` in `AppState`
    /// should be called afterward to restore persisted values from NATS KV.
    #[must_use]
    pub fn new(cfg: &H2AIConfig) -> Self {
        let tau_spread = cfg.calibration_tau_spread;
        let tao_ema_alpha = cfg.tao_estimator_ema_alpha;
        let tao_warmup = cfg.tao_estimator_warmup;
        let n_max_init = cfg.bandit_n_max_initial;
        let bandit_n_max_arms = cfg.bandit_n_max_arms;
        let bandit_prior_sigma = cfg.bandit_prior_sigma;
        let bandit_prior_strength = cfg.bandit_prior_strength;
        let srani_midpoint = cfg.srani.cold_start_midpoint();
        Self {
            calibration: Arc::new(RwLock::new(None)),
            tao_multiplier_estimator: Arc::new(RwLock::new(
                TaoMultiplierEstimator::new_with_alpha(tao_ema_alpha).with_warmup(tao_warmup),
            )),
            tau_spread_estimator: Arc::new(RwLock::new(TauSpreadEstimator::new(
                tau_spread[0],
                tau_spread[1],
            ))),
            bandit_state: Arc::new(RwLock::new(BanditState::new(
                n_max_init,
                0, // initial completed-task count
                bandit_n_max_arms,
                bandit_prior_sigma,
                bandit_prior_strength,
            ))),
            srani_state: Arc::new(RwLock::new((srani_midpoint, 0))),
            rho_ema: Arc::new(RwLock::new(RhoEmaState::default())),
        }
    }
}

/// Maps tenant IDs to isolated runtime estimator state.
///
/// `get_or_create` minimises redundant construction: the fast path is a lock-free
/// `DashMap` read; the slow path holds a shard lock via the entry API so only one
/// value is ever *stored*, even if two concurrent callers both race past the fast
/// path and call `TenantState::new` simultaneously.
#[derive(Clone, Default)]
pub struct TenantRegistry(Arc<DashMap<TenantId, Arc<TenantState>>>);

impl TenantRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(DashMap::new()))
    }

    #[must_use]
    pub fn get_or_create(&self, id: &TenantId, cfg: &H2AIConfig) -> Arc<TenantState> {
        if let Some(existing) = self.0.get(id) {
            return existing.clone();
        }
        self.0
            .entry(id.clone())
            .or_insert_with(|| Arc::new(TenantState::new(cfg)))
            .clone()
    }
}
