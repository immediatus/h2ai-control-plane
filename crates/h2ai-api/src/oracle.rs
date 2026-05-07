use async_nats::Client as RawNats;
use futures::StreamExt;
use h2ai_orchestrator::bandit::BanditState;
use h2ai_state::nats::NatsClient;
use h2ai_types::events::{CalibrationDriftWarning, OracleResultEvent, OracleSuspectEvent};
use h2ai_types::sizing::OracleObservation;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Rolling calibration window cap.
const ORACLE_CAP: usize = 200;

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

/// Background task that consumes oracle results and updates calibration state.
pub struct OracleAccumulator {
    pub nats_raw: RawNats,
    pub nats_state: Arc<NatsClient>,
    pub bandit: Arc<RwLock<BanditState>>,
    pub metrics: Arc<RwLock<crate::metrics::MetricsState>>,
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

        // 2. Append new observation and enforce 200-cap FIFO
        observations.push(OracleObservation {
            task_id: result.task_id.to_string(),
            q_confidence: result.q_confidence,
            y_oracle: result.passed,
            residual: result.residual,
            domain: result.domain.clone(),
            oracle_type: result.oracle_type.clone(),
            timestamp_ms: result.timestamp_ms,
        });
        enforce_fifo_cap(&mut observations, ORACLE_CAP);

        // 3. Persist updated window
        if let Err(e) = self.nats_state.put_oracle_observations(&observations).await {
            tracing::warn!(error = %e, "OracleAccumulator: failed to persist observations");
        }

        // 4. Recompute calibration status
        let status = determine_calibration_basis(&observations);

        // 4a. Publish health alerts when thresholds breached
        if status.n_observations >= 30 && status.ece > 0.15 {
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
                    "CalibrationDrift: ECE exceeded 0.15 threshold"
                );
            }
        }
        if status.n_observations >= 30 && status.pass_rate < 0.3 {
            if let Ok(payload) = serde_json::to_vec(&OracleSuspectEvent {
                pass_rate: status.pass_rate,
                n_observations: status.n_observations,
                reason: "pass_rate < 0.30 over 30+ observations — oracle may be misconfigured"
                    .to_owned(),
                timestamp_ms: result.timestamp_ms,
            }) {
                let _ = self
                    .nats_raw
                    .publish("h2ai.oracle.suspect", payload.into())
                    .await;
                tracing::warn!(
                    pass_rate = status.pass_rate,
                    n_obs = status.n_observations,
                    "OracleSuspect: pass rate below 0.30 threshold"
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
