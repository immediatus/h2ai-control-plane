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
/// ECE = mean(residuals) = (1/n) × Σ |q_confidence_i − y_oracle_i|
/// Returns 0.0 for empty input.
pub fn ece_from_observations(observations: &[OracleObservation]) -> f64 {
    if observations.is_empty() {
        return 0.0;
    }
    let sum: f64 = observations.iter().map(|o| o.residual).sum();
    sum / observations.len() as f64
}

/// Compute oracle pass rate (fraction of observations where y_oracle = true).
/// Returns 0.0 for empty input.
pub fn pass_rate_from_observations(observations: &[OracleObservation]) -> f64 {
    if observations.is_empty() {
        return 0.0;
    }
    let passed = observations.iter().filter(|o| o.y_oracle).count();
    passed as f64 / observations.len() as f64
}

/// Compute the P90 of residuals across observations.
///
/// Sorts residuals and returns the value at index `floor(0.9 × n)`.
/// Returns 0.0 for empty input.
pub fn residual_p90(observations: &[OracleObservation]) -> f64 {
    if observations.is_empty() {
        return 0.0;
    }
    let mut residuals: Vec<f64> = observations.iter().map(|o| o.residual).collect();
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((residuals.len() as f64 * 0.9).ceil() as usize)
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

/// Determine the current calibration basis from the observation window.
///
/// Thresholds (n < 10, 10 ≤ n < 30, n ≥ 30) are CLT-grounded mathematical constants,
/// NOT operator-configurable. Changing them would misclaim statistical guarantees.
/// - n ≥ 30: CLT holds (Lehmann & Romano 2005); conformal intervals valid
/// - n ≥ 10: bootstrap minimum (DiCiccio & Efron 1996)
///
/// The operator-configurable alert thresholds (oracle_ece_alert_threshold,
/// oracle_pass_rate_floor) in `handle_result` are separate concerns.
///
/// Rules (from design spec):
/// - n < 10                      → basis=0 (Heuristic — insufficient data)
/// - 10 ≤ n < 30                 → basis=1 (Bootstrap — coarse intervals, not conformal)
/// - n ≥ 30 AND ECE < 0.15      → basis=2 (Conformal)
/// - n ≥ 30 AND ECE ≥ 0.15      → basis=0 (Heuristic — quality regression)
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
    } else if n >= 10 {
        1
    } else {
        0
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
///
/// If `observations.len() > cap`, drops the oldest entries from the front.
pub fn enforce_fifo_cap(observations: &mut Vec<OracleObservation>, cap: usize) {
    if observations.len() > cap {
        let excess = observations.len() - cap;
        observations.drain(0..excess);
    }
}

/// Patch the ensemble calibration's p_mean with the empirically measured oracle pass rate.
/// Only called when n_observations >= 10 (Bootstrap threshold).
/// Returns `Some((p_before, p_after, rho_mean))` when a patch was applied, `None` otherwise.
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
            tracing::debug!(
                pass_rate,
                n_observations,
                old_p = existing_ec.p_mean,
                new_p = new_ec.p_mean,
                old_basis = ?existing_ec.prediction_basis,
                "oracle: patching ensemble p_mean from pass_rate"
            );
            let result = (existing_ec.p_mean, new_ec.p_mean, existing_ec.rho_mean);
            event.ensemble = Some(new_ec);
            return Some(result);
        }
    }
    None
}

/// Background task that consumes oracle results and updates calibration state.
pub struct OracleAccumulator {
    pub nats_raw: RawNats,
    pub nats_state: Arc<NatsClient>,
    pub bandit: Arc<RwLock<BanditState>>,
    pub metrics: Arc<RwLock<crate::metrics::MetricsState>>,
    /// Rolling window cap from `cfg.oracle_window_size`.
    pub oracle_window_size: usize,
    /// ECE alert threshold from `cfg.oracle_ece_alert_threshold`.
    pub oracle_ece_alert_threshold: f64,
    /// Pass-rate floor from `cfg.oracle_pass_rate_floor`.
    pub oracle_pass_rate_floor: f64,
    /// Shared calibration snapshot — updated in-place when oracle n >= 10.
    pub calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>>,
    /// From `cfg.calibration_max_ensemble_size`; passed to `EnsembleCalibration::from_measured_p`.
    pub calibration_max_ensemble_size: usize,
}

impl OracleAccumulator {
    /// Subscribe to `h2ai.oracle.results` and process events in a loop.
    ///
    /// This is a long-running background task; spawn with `tokio::spawn`.
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

        // 2. Append new observation and enforce FIFO cap (operator-configurable window size)
        observations.push(OracleObservation {
            task_id: result.task_id.to_string(),
            q_confidence: result.q_confidence,
            y_oracle: result.passed,
            residual: result.residual,
            domain: result.domain.clone(),
            oracle_type: result.oracle_type.clone(),
            timestamp_ms: result.timestamp_ms,
        });
        enforce_fifo_cap(&mut observations, self.oracle_window_size);

        // 3. Persist updated window
        if let Err(e) = self.nats_state.put_oracle_observations(&observations).await {
            tracing::warn!(error = %e, "OracleAccumulator: failed to persist observations");
        }

        // 4. Recompute calibration status
        let status = determine_calibration_basis(&observations);

        // 4b. Wire oracle pass_rate → EnsembleCalibration (INNOVATION-1, GAP-B2).
        if status.n_observations >= 10 {
            let cg_mean = {
                let cal = self.calibration.read().await;
                cal.as_ref()
                    .and_then(|c| c.ensemble.as_ref())
                    .map(|ec| (1.0 - ec.rho_mean).clamp(f64::EPSILON, 1.0))
                    .unwrap_or(0.3)
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

        // 4a. Publish health alerts when thresholds breached
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
                    threshold = self.oracle_ece_alert_threshold,
                    "CalibrationDrift: ECE exceeded alert threshold"
                );
            }
        }
        if status.n_observations >= 30 && status.pass_rate < self.oracle_pass_rate_floor {
            if let Ok(payload) = serde_json::to_vec(&OracleSuspectEvent {
                pass_rate: status.pass_rate,
                n_observations: status.n_observations,
                reason: format!(
                    "pass_rate < {:.2} over 30+ observations — oracle may be misconfigured",
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
                    floor = self.oracle_pass_rate_floor,
                    "OracleSuspect: pass rate below floor threshold"
                );
            }
        }

        // 5. Update bandit (tier1_passed)
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

#[cfg(test)]
mod innovation1_tests {
    use super::*;
    use h2ai_types::events::{CalibrationCompletedEvent, CalibrationSource, CgMode};
    use h2ai_types::sizing::{
        CoherencyCoefficients, CoordinationThreshold, EnsembleCalibration, PredictionBasis,
    };
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn mock_calibration_event() -> CalibrationCompletedEvent {
        let cc = CoherencyCoefficients::new(0.1, 0.04, vec![0.7]).unwrap();
        let ct = CoordinationThreshold::from_calibration(&cc, 0.9);
        CalibrationCompletedEvent {
            calibration_id: h2ai_types::identity::TaskId::new(),
            coefficients: cc,
            coordination_threshold: ct,
            ensemble: Some(EnsembleCalibration::from_cg_mean(0.7, 9)),
            eigen: None,
            timestamp: chrono::Utc::now(),
            pairwise_beta: None,
            cg_mode: CgMode::default(),
            adapter_families: vec![],
            explorer_verification_family_match: false,
            single_family_warning: false,
            n_max_lo: 0.0,
            n_max_hi: 0.0,
            n_eff_cosine_prior: 0.0,
            calibration_quality: Default::default(),
            calibration_source: CalibrationSource::Measured,
            beta_quality: None,
        }
    }

    #[tokio::test]
    async fn handle_result_patches_ensemble_to_empirical_after_10_observations() {
        let calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>> =
            Arc::new(RwLock::new(Some(mock_calibration_event())));

        // Verify initial state is Heuristic.
        {
            let cal = calibration.read().await;
            assert_eq!(
                cal.as_ref()
                    .unwrap()
                    .ensemble
                    .as_ref()
                    .unwrap()
                    .prediction_basis,
                PredictionBasis::Heuristic
            );
        }

        let initial_rho = calibration
            .read()
            .await
            .as_ref()
            .unwrap()
            .ensemble
            .as_ref()
            .unwrap()
            .rho_mean;
        let initial_cg = (1.0 - initial_rho).clamp(f64::EPSILON, 1.0);

        let pass_rate = 0.65_f64;
        let n = 10_usize;
        let max_n = 9_usize;
        patch_ensemble_p_from_oracle(&calibration, pass_rate, n, initial_cg, max_n).await;

        let cal = calibration.read().await;
        let ec = cal.as_ref().unwrap().ensemble.as_ref().unwrap();
        assert_eq!(
            ec.prediction_basis,
            PredictionBasis::Empirical,
            "ensemble must switch to Empirical after oracle patches p_mean"
        );
        assert!(
            (ec.p_mean - 0.65_f64.clamp(0.5, 1.0)).abs() < 1e-9,
            "p_mean must be oracle pass_rate clamped to [0.5, 1.0]"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_types::sizing::{OracleDomain, OracleType};

    #[test]
    fn enforce_fifo_cap_uses_custom_size() {
        let mut obs: Vec<OracleObservation> = (0..10)
            .map(|i| OracleObservation {
                task_id: format!("t{i}"),
                q_confidence: 0.5,
                y_oracle: true,
                residual: 0.1,
                domain: OracleDomain::Unknown,
                oracle_type: OracleType::TestSuite,
                timestamp_ms: i as u64,
            })
            .collect();
        enforce_fifo_cap(&mut obs, 5);
        assert_eq!(obs.len(), 5);
        assert_eq!(obs[0].task_id, "t5");
    }
}
