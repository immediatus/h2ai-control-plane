//! Calibration drift detection: DDM fast-layer + BOCPD structural shift + ORCA margin.

use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_types::calibration::{CalibrationChangepoint, CalibrationDriftWarning};
use std::collections::VecDeque;

// ── DDM (Drift Detection Method, Gama et al. 2004) ────────────────────────────

/// O(1) per observation. Fires a warning when the sliding-window mean deviates more than
/// k × std_reference from the reference mean established on the first full window.
pub struct DdmDetector {
    window: VecDeque<f64>,
    window_size: usize,
    k_ddm: f64,
    reference_mean: f64,
    reference_std: f64,
    initialized: bool,
}

impl DdmDetector {
    pub fn new(window_size: usize, k_ddm: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size.max(2)),
            window_size: window_size.max(2),
            k_ddm,
            reference_mean: 0.0,
            reference_std: 0.0,
            initialized: false,
        }
    }

    /// Feed one observation. Returns `Some(warning)` when drift is detected; `None` otherwise.
    pub fn observe(&mut self, x: f64) -> Option<CalibrationDriftWarning> {
        self.window.push_back(x);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
        if !self.initialized && self.window.len() == self.window_size {
            let mean = self.window.iter().sum::<f64>() / self.window_size as f64;
            let variance = self.window.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (self.window_size - 1) as f64;
            self.reference_mean = mean;
            // Use a minimum std of 0.01 to handle nearly-constant reference windows
            self.reference_std = variance.sqrt().max(0.01);
            self.initialized = true;
            return None;
        }
        if !self.initialized || self.window.len() < self.window_size {
            return None;
        }
        let recent_mean = self.window.iter().sum::<f64>() / self.window.len() as f64;
        let deviation = (recent_mean - self.reference_mean).abs() / self.reference_std;
        if deviation > self.k_ddm {
            Some(CalibrationDriftWarning {
                detected_at: Utc::now(),
                metric: "consensus_agreement_rate".to_string(),
                recent_mean,
                reference_mean: self.reference_mean,
                deviation_sigmas: deviation,
            })
        } else {
            None
        }
    }

    /// Reset reference distribution (call after successful recalibration).
    pub fn reset(&mut self) {
        self.window.clear();
        self.initialized = false;
    }
}

// ── BOCPD (Adams & MacKay, 2007) ──────────────────────────────────────────────
// Normal-Inverse-Gamma conjugate prior. Student-t predictive distribution.
// lgamma implemented via Stirling's approximation + recursion (no external crates).

/// Stirling-series lgamma: accurate to ~1e-10 for x > 0.5.
fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        std::f64::consts::PI.ln() - ((std::f64::consts::PI * x).sin().abs()).ln() - lgamma(1.0 - x)
    } else if x < 8.0 {
        lgamma(x + 1.0) - x.ln()
    } else {
        let z = 1.0 / (x * x);
        (x - 0.5) * x.ln() - x
            + 0.5 * (2.0 * std::f64::consts::PI).ln()
            + (1.0 / 12.0 - z * (1.0 / 360.0 - z / 1260.0)) / x
    }
}

/// Student-t log-PDF with df degrees of freedom, location loc, scale scale.
fn student_t_log_pdf(x: f64, df: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    lgamma((df + 1.0) / 2.0)
        - lgamma(df / 2.0)
        - 0.5 * (df * std::f64::consts::PI).ln()
        - scale.ln()
        - (df + 1.0) / 2.0 * (1.0 + z * z / df).ln()
}

/// Numerically stable log-sum-exp.
fn logsumexp(log_probs: &[f64]) -> f64 {
    let max = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    max + log_probs
        .iter()
        .map(|&lp| (lp - max).exp())
        .sum::<f64>()
        .ln()
}

/// Normal-Inverse-Gamma conjugate parameters for one BOCPD run-length segment.
#[derive(Clone)]
struct NigParams {
    mu: f64,
    kappa: f64,
    alpha: f64,
    beta: f64,
}

impl NigParams {
    fn update_one(&self, x: f64) -> Self {
        let kappa_new = self.kappa + 1.0;
        NigParams {
            mu: (self.kappa * self.mu + x) / kappa_new,
            kappa: kappa_new,
            alpha: self.alpha + 0.5,
            beta: self.beta + self.kappa * (x - self.mu).powi(2) / (2.0 * kappa_new),
        }
    }

    fn predictive_log_prob(&self, x: f64) -> f64 {
        let df = 2.0 * self.alpha;
        let scale_sq = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa);
        let scale = scale_sq.sqrt().max(1e-9);
        student_t_log_pdf(x, df, self.mu, scale)
    }
}

#[derive(Clone)]
struct RunState {
    nig: NigParams,
    log_mass: f64,
}

/// Maximum number of run-length states retained (memory bound).
const MAX_RUN_LENGTH: usize = 500;

/// Bayesian Online Changepoint Detection (Adams & MacKay, 2007) with NIG conjugate prior.
///
/// Fires `CalibrationChangepoint` when P(r_t ≤ 4 | x_{1:t}) > `changepoint_threshold`.
pub struct BocpdDetector {
    prior: NigParams,
    hazard_rate: f64,
    changepoint_threshold: f64,
    run_states: Vec<RunState>,
}

impl BocpdDetector {
    pub fn new(hazard_rate: f64, changepoint_threshold: f64) -> Self {
        let prior = NigParams {
            mu: 0.5,
            kappa: 1.0,
            alpha: 1.0,
            beta: 0.1,
        };
        Self {
            prior: prior.clone(),
            hazard_rate,
            changepoint_threshold,
            run_states: vec![RunState {
                nig: prior,
                log_mass: 0.0,
            }],
        }
    }

    /// Number of active run-length states (for testing the memory bound).
    pub fn run_states_len(&self) -> usize {
        self.run_states.len()
    }

    /// Feed one observation. Returns `Some(changepoint)` when P(r_t ≤ 4) > threshold.
    pub fn observe(&mut self, x: f64) -> Option<CalibrationChangepoint> {
        let log_h = self.hazard_rate.ln();
        let log_1mh = (1.0 - self.hazard_rate).ln();

        let mut new_run_states: Vec<RunState> = Vec::with_capacity(self.run_states.len() + 1);
        let mut cp_log_masses: Vec<f64> = Vec::with_capacity(self.run_states.len());

        for state in &self.run_states {
            let log_pred = state.nig.predictive_log_prob(x);
            let updated_nig = state.nig.update_one(x);
            new_run_states.push(RunState {
                nig: updated_nig,
                log_mass: state.log_mass + log_pred + log_1mh,
            });
            cp_log_masses.push(state.log_mass + log_pred + log_h);
        }

        new_run_states.insert(
            0,
            RunState {
                nig: self.prior.clone(),
                log_mass: logsumexp(&cp_log_masses),
            },
        );

        if new_run_states.len() > MAX_RUN_LENGTH {
            new_run_states.truncate(MAX_RUN_LENGTH);
        }

        let log_norm = logsumexp(
            &new_run_states
                .iter()
                .map(|s| s.log_mass)
                .collect::<Vec<_>>(),
        );
        for state in &mut new_run_states {
            state.log_mass -= log_norm;
        }

        self.run_states = new_run_states;

        // Only test for a changepoint once we have accumulated enough run-length states
        // to distinguish "recently reset" from "genuinely changed". Before 6 observations
        // all probability mass is in the first few states regardless of the data.
        if self.run_states.len() <= 5 {
            return None;
        }

        let short_run_mass: f64 = self
            .run_states
            .iter()
            .take(5)
            .map(|s| s.log_mass.exp())
            .sum();

        if short_run_mass > self.changepoint_threshold {
            Some(CalibrationChangepoint {
                detected_at: Utc::now(),
                bocpd_run_length_posterior_mass: short_run_mass,
                conformal_margin_applied: 0.0, // filled in by DriftMonitor
            })
        } else {
            None
        }
    }

    /// Reset the detector (call after successful recalibration).
    pub fn reset(&mut self) {
        self.run_states = vec![RunState {
            nig: self.prior.clone(),
            log_mass: 0.0,
        }];
    }
}

// ── DriftMonitor: combines DDM + BOCPD + ORCA conformal margin ────────────────

/// Drift event produced by `DriftMonitor::observe`.
pub enum DriftEvent {
    /// DDM fast-layer warning: deviation exceeded k × σ_reference.
    Warning(CalibrationDriftWarning),
    /// BOCPD structural changepoint: P(recent shift) > threshold.
    Changepoint(CalibrationChangepoint),
}

/// Stateful monitor combining DDM + BOCPD + ORCA conformal margin management.
///
/// Store one instance in `AppState` behind `Arc<tokio::sync::Mutex<...>>`.
/// Feed `consensus_agreement_rate` from each completed task via `observe()`.
pub struct DriftMonitor {
    ddm: DdmDetector,
    bocpd: BocpdDetector,
    conformal_margin: f64,
    #[allow(dead_code)]
    auto_recalibrate: bool,
    staleness_ttl_secs: u64,
    changepoint_active: bool,
    changepoint_detected_at: Option<std::time::Instant>,
}

impl DriftMonitor {
    /// Build from config. Used by `AppState::new()`.
    pub fn from_config(cfg: &H2AIConfig) -> Self {
        Self::new(
            DdmDetector::new(cfg.drift_ddm_window, cfg.drift_ddm_k),
            BocpdDetector::new(
                cfg.drift_bocpd_hazard_rate,
                cfg.drift_bocpd_changepoint_threshold,
            ),
            cfg.drift_conformal_margin,
            cfg.auto_recalibrate_on_drift,
            cfg.drift_staleness_ttl_secs,
        )
    }

    /// Direct constructor (used in tests for custom parameters).
    pub fn new(
        ddm: DdmDetector,
        bocpd: BocpdDetector,
        conformal_margin: f64,
        auto_recalibrate: bool,
        staleness_ttl_secs: u64,
    ) -> Self {
        Self {
            ddm,
            bocpd,
            conformal_margin,
            auto_recalibrate,
            staleness_ttl_secs,
            changepoint_active: false,
            changepoint_detected_at: None,
        }
    }

    /// Feed one `consensus_agreement_rate` observation (expected in [0.0, 1.0]).
    /// Returns all drift events produced in this step (may be 0, 1, or 2).
    pub fn observe(&mut self, agreement_rate: f64) -> Vec<DriftEvent> {
        let mut events = Vec::new();

        if let Some(warning) = self.ddm.observe(agreement_rate) {
            events.push(DriftEvent::Warning(warning));
        }

        if let Some(mut cp) = self.bocpd.observe(agreement_rate) {
            cp.conformal_margin_applied = self.conformal_margin;
            self.changepoint_active = true;
            self.changepoint_detected_at = Some(std::time::Instant::now());
            tracing::warn!(
                target: "h2ai.calibration.drift",
                bocpd_mass = cp.bocpd_run_length_posterior_mass,
                conformal_margin = self.conformal_margin,
                auto_recalibrate = self.auto_recalibrate,
                "CalibrationChangepoint detected — ORCA conformal margin active"
            );
            events.push(DriftEvent::Changepoint(cp));
        }

        events
    }

    /// Returns `conformal_margin` when a changepoint is active and within TTL; `0.0` otherwise.
    ///
    /// Subtract from `VerificationConfig::threshold` before running verification (ORCA guarantee).
    pub fn active_conformal_margin(&self) -> f64 {
        if !self.changepoint_active {
            return 0.0;
        }
        match self.changepoint_detected_at {
            Some(t) if t.elapsed().as_secs() < self.staleness_ttl_secs => self.conformal_margin,
            _ => 0.0,
        }
    }

    /// Reset after successful recalibration. Clears both detectors and drift state.
    pub fn reset_after_recalibration(&mut self) {
        self.ddm.reset();
        self.bocpd.reset();
        self.changepoint_active = false;
        self.changepoint_detected_at = None;
    }
}
