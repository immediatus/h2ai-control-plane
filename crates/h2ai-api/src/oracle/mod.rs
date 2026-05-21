pub mod client;

use async_nats::Client as RawNats;
use futures::StreamExt;
use h2ai_orchestrator::bandit::BanditState;
use h2ai_state::nats::NatsClient;
use h2ai_types::events::{
    CalibrationCompletedEvent, CalibrationDriftWarning, OracleCalibrationPatchedEvent,
    OracleResultEvent, OracleSuspectEvent,
};
use h2ai_types::sizing::{EnsembleCalibration, OracleObservation};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Compute ECE (Expected Calibration Error) over oracle observations.
///
/// ECE = (1/n) × Σ |q_confidence_i − y_oracle_i|
#[must_use]
pub fn ece_from_observations(observations: &[OracleObservation]) -> f64 {
    if observations.is_empty() {
        return 0.0;
    }
    let sum: f64 = observations.iter().map(|o| o.residual).sum();
    sum / observations.len() as f64
}

/// Compute oracle pass rate (fraction of observations where y_oracle = true).
#[must_use]
pub fn pass_rate_from_observations(observations: &[OracleObservation]) -> f64 {
    if observations.is_empty() {
        return 0.0;
    }
    let passed = observations.iter().filter(|o| o.y_oracle).count();
    passed as f64 / observations.len() as f64
}

/// Compute the P90 of residuals. Uses Angelopoulos-Bates Theorem 1: ⌈(n+1)×0.9⌉ − 1.
#[must_use]
pub fn residual_p90(observations: &[OracleObservation]) -> f64 {
    if observations.is_empty() {
        return 0.0;
    }
    let mut residuals: Vec<f64> = observations.iter().map(|o| o.residual).collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (((residuals.len() + 1) as f64 * 0.9).ceil() as usize)
        .saturating_sub(1)
        .min(residuals.len() - 1);
    residuals[idx]
}

/// Summary of the current calibration window state.
#[derive(Debug, Clone)]
pub struct OracleCalibrationStatus {
    pub n_observations: usize,
    pub ece: f64,
    pub pass_rate: f64,
    pub residual_p90: f64,
    /// 0=Heuristic, 1=Bootstrap, 2=Conformal.
    pub basis: u8,
}

/// Determine calibration basis from the observation window.
///
/// Thresholds are CLT-grounded constants (Lehmann & Romano 2005; DiCiccio & Efron 1996):
/// - n < 10: Heuristic
/// - 10 ≤ n < 30: Bootstrap
/// - n ≥ 30, ECE < 0.15: Conformal
/// - n ≥ 30, ECE ≥ 0.15: Heuristic (quality regression)
#[must_use]
pub fn determine_calibration_basis(observations: &[OracleObservation]) -> OracleCalibrationStatus {
    let n = observations.len();
    let ece = ece_from_observations(observations);
    let pass_rate = pass_rate_from_observations(observations);
    let p90 = residual_p90(observations);
    let basis = if n >= 30 {
        if ece < 0.15 {
            2
        } else {
            0
        }
    } else {
        u8::from(n >= 10)
    };
    OracleCalibrationStatus {
        n_observations: n,
        ece,
        pass_rate,
        residual_p90: p90,
        basis,
    }
}

/// Enforce a FIFO cap on the observation window.
pub fn enforce_fifo_cap(observations: &mut Vec<OracleObservation>, cap: usize) {
    if observations.len() > cap {
        let excess = observations.len() - cap;
        observations.drain(0..excess);
    }
}

/// Patch ensemble p_mean from empirical oracle pass rate. Only when n ≥ 10.
pub async fn patch_ensemble_p_from_oracle(
    calibration: &Arc<RwLock<Option<CalibrationCompletedEvent>>>,
    pass_rate: f64,
    n_observations: usize,
    cg_mean: f64,
    calibration_max_ensemble_size: usize,
) -> Option<(f64, f64, f64)> {
    if n_observations < 10 {
        return None;
    }
    let mut cal = calibration.write().await;
    if let Some(ref mut event) = *cal {
        if let Some(ref existing_ec) = event.ensemble {
            let new_ec = EnsembleCalibration::from_measured_p(
                pass_rate.clamp(0.5, 1.0),
                cg_mean,
                calibration_max_ensemble_size,
            );
            let result = (existing_ec.p_mean, new_ec.p_mean, existing_ec.rho_mean);
            event.ensemble = Some(new_ec);
            return Some(result);
        }
    }
    None
}

/// Background task: consumes oracle results from NATS and updates calibration state.
pub struct OracleAccumulator {
    pub nats_raw: RawNats,
    pub nats_state: Arc<NatsClient>,
    pub bandit: Arc<RwLock<BanditState>>,
    pub metrics: Arc<RwLock<crate::metrics::MetricsState>>,
    pub oracle_window_size: usize,
    pub oracle_ece_alert_threshold: f64,
    pub oracle_pass_rate_floor: f64,
    pub calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>>,
    pub calibration_max_ensemble_size: usize,
}

impl OracleAccumulator {
    pub async fn run(self) {
        let mut sub = match self.nats_raw.subscribe("h2ai.oracle.results").await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "OracleAccumulator: failed to subscribe");
                return;
            }
        };
        while let Some(msg) = sub.next().await {
            if let Ok(result) = serde_json::from_slice::<OracleResultEvent>(&msg.payload) {
                self.handle_result(result).await;
            }
        }
    }

    async fn handle_result(&self, result: OracleResultEvent) {
        // 1. Load existing observations from KV
        let mut observations = self
            .nats_state
            .get_oracle_observations()
            .await
            .unwrap_or_default();

        // 2. Append and enforce FIFO cap
        observations.push(OracleObservation {
            task_id: result.task_id.to_string(),
            q_confidence: result.q_confidence,
            y_oracle: result.passed,
            residual: result.residual,
            domain: result.domain.clone(),
            timestamp_ms: result.timestamp_ms,
        });
        enforce_fifo_cap(&mut observations, self.oracle_window_size);

        // 3. Persist
        if let Err(e) = self.nats_state.put_oracle_observations(&observations).await {
            tracing::warn!(error = %e, "OracleAccumulator: failed to persist observations");
        }

        // 4. Recompute calibration status
        let status = determine_calibration_basis(&observations);

        // 4a. Patch ensemble p_mean when n ≥ 10
        if status.n_observations >= 10 {
            let cg_mean = {
                let cal = self.calibration.read().await;
                cal.as_ref()
                    .and_then(|c| c.ensemble.as_ref())
                    .map_or(0.3, |ec| (1.0 - ec.rho_mean).clamp(f64::EPSILON, 1.0))
            };
            if let Some((p_before, p_after, rho_mean)) = patch_ensemble_p_from_oracle(
                &self.calibration,
                status.pass_rate,
                status.n_observations,
                cg_mean,
                self.calibration_max_ensemble_size,
            )
            .await
            {
                let patch_ev = OracleCalibrationPatchedEvent {
                    task_id: result.task_id.clone(),
                    oracle_pass_rate: status.pass_rate,
                    n_observations: status.n_observations,
                    p_mean_before: p_before,
                    p_mean_after: p_after,
                    rho_mean,
                    timestamp: chrono::Utc::now(),
                };
                use h2ai_types::events::H2AIEvent;
                let ev = H2AIEvent::OracleCalibrationPatched(patch_ev);
                if let Err(e) = self.nats_state.publish_event(&result.task_id, &ev).await {
                    tracing::warn!(error = %e, "failed to publish OracleCalibrationPatchedEvent");
                }
            }
        }

        // 4b. Health alerts (n ≥ 30)
        if status.n_observations >= 30 && status.ece > self.oracle_ece_alert_threshold {
            if let Ok(payload) = serde_json::to_vec(&CalibrationDriftWarning {
                n_observations: status.n_observations,
                ece: status.ece,
                timestamp_ms: result.timestamp_ms,
            }) {
                let _ = self
                    .nats_raw
                    .publish("h2ai.oracle.calibration_drift", payload.into())
                    .await;
                tracing::warn!(
                    n_obs = status.n_observations,
                    ece = status.ece,
                    "CalibrationDrift: ECE exceeded alert threshold"
                );
            }
        }
        if status.n_observations >= 30 && status.pass_rate < self.oracle_pass_rate_floor {
            if let Ok(payload) = serde_json::to_vec(&OracleSuspectEvent {
                pass_rate: status.pass_rate,
                n_observations: status.n_observations,
                reason: format!(
                    "pass_rate < {:.2} over 30+ observations",
                    self.oracle_pass_rate_floor
                ),
                timestamp_ms: result.timestamp_ms,
            }) {
                let _ = self
                    .nats_raw
                    .publish("h2ai.oracle.suspect", payload.into())
                    .await;
                tracing::warn!(
                    pass_rate = status.pass_rate,
                    n_obs = status.n_observations,
                    "OracleSuspect: pass rate below floor threshold"
                );
            }
        }

        // 5. Update bandit
        {
            let mut bandit = self.bandit.write().await;
            bandit.update(result.n_used, Some(result.passed), None);
        }

        // 6. Update Prometheus metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.oracle_ece = status.ece;
            metrics.oracle_n_observations = status.n_observations as u64;
            metrics.oracle_pass_rate = status.pass_rate;
            metrics.oracle_residual_p90 = status.residual_p90;
            metrics.oracle_calibration_basis = status.basis;
        }

        tracing::debug!(
            task_id = %result.task_id,
            passed = result.passed,
            residual = result.residual,
            n_obs = status.n_observations,
            ece = status.ece,
            basis = status.basis,
            "oracle result processed"
        );
    }
}
