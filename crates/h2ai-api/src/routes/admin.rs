use crate::state::AppState;
use axum::{
    extract::{Path, State},
    response::IntoResponse,
    Json,
};
use h2ai_orchestrator::bandit::BanditState;
use h2ai_orchestrator::self_optimizer::TauSpreadEstimator;
use h2ai_orchestrator::tao_loop::TaoMultiplierEstimator;
use h2ai_types::identity::TenantId;

/// Resets all per-tenant adaptive estimators to their config-derived cold-start priors.
///
/// Intended exclusively for experiment harness use before each experimental arm.
/// Does NOT reset calibration (α, β₀ are structural measurements, not learned state).
/// Does NOT flush the NATS event log.
pub async fn reset_experiment_state(
    Path(tenant_id): Path<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let tid = TenantId::from(tenant_id.as_str());
    let ts = state.tenant_state(&tid);
    let cfg = &state.cfg;

    let tau_spread = cfg.calibration_tau_spread;
    *ts.tau_spread_estimator.write().await = TauSpreadEstimator::new(tau_spread[0], tau_spread[1]);

    *ts.bandit_state.write().await = BanditState::new(
        cfg.bandit_n_max_initial,
        0,
        cfg.bandit_n_max_arms,
        cfg.bandit_prior_sigma,
        cfg.bandit_prior_strength,
    );

    *ts.tao_multiplier_estimator.write().await =
        TaoMultiplierEstimator::new_with_alpha(cfg.tao_estimator_ema_alpha)
            .with_warmup(cfg.tao_estimator_warmup);

    *ts.rho_ema.write().await = crate::rho_ema::RhoEmaState::default();

    Json(reset_response_body_value(&tenant_id))
}

pub fn reset_response_body_value(tenant_id: &str) -> serde_json::Value {
    serde_json::json!({
        "tenant_id": tenant_id,
        "reset": true,
        "fields_reset": [
            "tau_spread_estimator",
            "bandit_state",
            "tao_multiplier_estimator",
            "rho_ema"
        ]
    })
}
