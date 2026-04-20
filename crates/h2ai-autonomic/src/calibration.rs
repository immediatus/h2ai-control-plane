use chrono::Utc;
use futures::future::join_all;
use h2ai_config::H2AIConfig;
use h2ai_context::jaccard::{jaccard, tokenize};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::events::CalibrationCompletedEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::physics::{
    CoherencyCoefficients, CoordinationThreshold, EnsembleCalibration, PhysicsError, TauValue,
    tau_alignment,
};
use thiserror::Error;
use tokio::time::Instant;

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("adapter error: {0}")]
    Adapter(String),
    #[error("physics error: {0}")]
    Physics(#[from] PhysicsError),
    #[error("need at least 1 adapter to calibrate")]
    NoAdapters,
}

pub struct CalibrationInput<'a> {
    pub calibration_id: TaskId,
    pub task_prompts: Vec<String>,
    pub adapters: Vec<&'a dyn IComputeAdapter>,
    pub cfg: &'a H2AIConfig,
}

pub struct CalibrationHarness;

impl CalibrationHarness {
    pub async fn run(
        input: CalibrationInput<'_>,
    ) -> Result<CalibrationCompletedEvent, CalibrationError> {
        if input.adapters.is_empty() {
            return Err(CalibrationError::NoAdapters);
        }
        let tau = TauValue::new(input.cfg.calibration_tau)
            .expect("calibration_tau must be in [0,1]");
        let m = input.adapters.len();

        // Phase A: run the first 2 adapters in parallel to get T₂ and per-adapter times for T₁.
        // Phase B: run all M adapters in parallel to get T_M and all outputs for CG_mean.
        // When M < 2, skip Phase A and use a single run as both phases.
        let (t1_proxy, t2_parallel, t_m_parallel, adapter_outputs) = if m >= 2 {
            let (phase_a_outputs, t2_wall) =
                Self::run_adapters_parallel(&input.adapters[..2], &input.task_prompts, tau, input.cfg)
                    .await?;
            // T₁ = mean per-adapter serial time (approximation of single-adapter cost)
            let t1 = phase_a_outputs.iter().map(|(_, t)| *t).sum::<f64>() / 2.0;

            let (all_outputs, t_m_wall) =
                Self::run_adapters_parallel(&input.adapters, &input.task_prompts, tau, input.cfg)
                    .await?;
            let outputs: Vec<Vec<String>> = all_outputs.into_iter().map(|(o, _)| o).collect();
            (t1, t2_wall, t_m_wall, outputs)
        } else {
            // M == 1: no parallelism to measure; use fallback parameters.
            let (single_out, t_single) =
                Self::run_adapters_parallel(&input.adapters, &input.task_prompts, tau, input.cfg)
                    .await?;
            let outputs: Vec<Vec<String>> = single_out.into_iter().map(|(o, _)| o).collect();
            (t_single, t_single, t_single, outputs)
        };

        // Derive α and β₀ analytically from USL linearization.
        let (alpha, beta_base) = Self::usl_fit(
            t1_proxy,
            t2_parallel,
            m,
            t_m_parallel,
            input.cfg.alpha_contention,
            input.cfg.beta_base_default,
        );

        // CG_mean from pairwise Jaccard across all adapter output pairs.
        let (cg_samples, ensemble) = if adapter_outputs.len() < 2 {
            (vec![input.cfg.calibration_cg_fallback], None)
        } else {
            let cal_tau = TauValue::new(input.cfg.calibration_tau)
                .expect("calibration_tau must be in [0,1]");
            let align = tau_alignment(cal_tau, cal_tau); // = 1.0 when all taus equal

            let mut pairs = Vec::new();
            for i in 0..adapter_outputs.len() {
                for j in (i + 1)..adapter_outputs.len() {
                    let oi = adapter_outputs[i].join(" ");
                    let oj = adapter_outputs[j].join(" ");
                    let ki = tokenize(&oi);
                    let kj = tokenize(&oj);
                    pairs.push(jaccard(&ki, &kj) * align);
                }
            }
            let cg_mean_val: f64 = pairs.iter().sum::<f64>() / pairs.len() as f64;
            let ec = if input.cfg.baseline_accuracy_proxy > 0.0 {
                EnsembleCalibration::from_measured_p(
                    input.cfg.baseline_accuracy_proxy,
                    cg_mean_val,
                    9,
                )
            } else {
                EnsembleCalibration::from_cg_mean(cg_mean_val, 9)
            };
            (pairs, Some(ec))
        };

        let cc = CoherencyCoefficients::new(alpha, beta_base, cg_samples)?;
        let coordination_threshold =
            CoordinationThreshold::from_calibration(&cc, input.cfg.coordination_threshold_max);

        Ok(CalibrationCompletedEvent {
            calibration_id: input.calibration_id,
            coefficients: cc,
            coordination_threshold,
            ensemble,
            timestamp: Utc::now(),
        })
    }

    /// Derive USL parameters α and β₀ analytically from two parallel timing measurements.
    ///
    /// Uses the linearisation z(N) = N·T_parallel(N)/T₁ − 1 = α(N−1) + β₀·N(N−1).
    /// With two data points at N=2 (Phase A) and N=M (Phase B):
    ///   β₀ = (z_M − z₂·(M−1)) / ((M−1)(M−2))
    ///   α  = z₂ − 2·β₀
    ///
    /// Falls back to (alpha_fallback, beta_fallback) when:
    /// - M < 3 (denominator (M−1)(M−2) is zero at M=2)
    /// - any timing is degenerate (≤ 0)
    /// - derived α or β₀ are negative (super-linear speedup or measurement noise)
    pub fn usl_fit(
        t1: f64,
        t2_parallel: f64,
        m: usize,
        t_m_parallel: f64,
        alpha_fallback: f64,
        beta_fallback: f64,
    ) -> (f64, f64) {
        if m < 3 || t1 < 1e-9 || t2_parallel < 1e-9 || t_m_parallel < 1e-9 {
            return (alpha_fallback, beta_fallback);
        }
        let m_f = m as f64;
        let z2 = 2.0 * t2_parallel / t1 - 1.0;
        let z_m = m_f * t_m_parallel / t1 - 1.0;

        let beta_denom = (m_f - 1.0) * (m_f - 2.0);
        // m >= 3 (integer), so denom >= 2.0; no zero-division possible.
        let beta0 = (z_m - z2 * (m_f - 1.0)) / beta_denom;
        let alpha = z2 - 2.0 * beta0;

        // Negative params indicate degenerate measurement (e.g. super-linear speedup).
        // Must check before clamping — clamping would mask the degenerate case.
        if beta0 < 0.0 || alpha < 0.0 {
            return (alpha_fallback, beta_fallback);
        }

        (alpha.clamp(0.05, 0.5), beta0.clamp(1e-6, 0.1))
    }

    /// Run a slice of adapters concurrently on all prompts.
    /// Returns (per-adapter (outputs, elapsed_secs), wall_clock_secs).
    async fn run_adapters_parallel(
        adapters: &[&dyn IComputeAdapter],
        prompts: &[String],
        tau: TauValue,
        cfg: &H2AIConfig,
    ) -> Result<(Vec<(Vec<String>, f64)>, f64), CalibrationError> {
        let t_wall_start = Instant::now();
        let futures: Vec<_> = adapters
            .iter()
            .map(|adapter| async move {
                let t0 = Instant::now();
                let mut outputs = Vec::new();
                for prompt in prompts.iter() {
                    let req = ComputeRequest {
                        system_context: String::new(),
                        task: prompt.clone(),
                        tau,
                        max_tokens: cfg.calibration_max_tokens,
                    };
                    let resp = adapter
                        .execute(req)
                        .await
                        .map_err(|e| CalibrationError::Adapter(e.to_string()))?;
                    outputs.push(resp.output);
                }
                Ok::<_, CalibrationError>((outputs, t0.elapsed().as_secs_f64()))
            })
            .collect();

        let results: Vec<Result<(Vec<String>, f64), CalibrationError>> = join_all(futures).await;
        let t_wall = t_wall_start.elapsed().as_secs_f64();

        let mut per_adapter = Vec::with_capacity(results.len());
        for r in results {
            per_adapter.push(r?);
        }
        Ok((per_adapter, t_wall))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usl_throughput(n: f64, alpha: f64, beta: f64) -> f64 {
        n / (1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0))
    }

    #[test]
    fn usl_fit_recovers_ai_agent_params() {
        // Ground truth: α=0.15, β₀=0.01, CG_mean=0.4 → β_eff=0.025
        let true_alpha = 0.15_f64;
        let true_beta0 = 0.01_f64;
        let t1 = 1.0_f64;
        // Simulate wall-clock times from USL formula
        let t2 = t1 / usl_throughput(2.0, true_alpha, true_beta0);
        let t4 = t1 / usl_throughput(4.0, true_alpha, true_beta0);

        let (alpha, beta0) = CalibrationHarness::usl_fit(t1, t2, 4, t4, 0.12, 0.01);
        assert!(
            (alpha - true_alpha).abs() < 0.005,
            "α recovery: expected {true_alpha}, got {alpha:.4}"
        );
        assert!(
            (beta0 - true_beta0).abs() < 0.001,
            "β₀ recovery: expected {true_beta0}, got {beta0:.6}"
        );
    }

    #[test]
    fn usl_fit_recovers_human_team_params() {
        let true_alpha = 0.10_f64;
        let true_beta0 = 0.005_f64;
        let t1 = 1.0_f64;
        let t2 = t1 / usl_throughput(2.0, true_alpha, true_beta0);
        let t5 = t1 / usl_throughput(5.0, true_alpha, true_beta0);

        let (alpha, beta0) = CalibrationHarness::usl_fit(t1, t2, 5, t5, 0.12, 0.01);
        assert!(
            (alpha - true_alpha).abs() < 0.005,
            "α: expected {true_alpha}, got {alpha:.4}"
        );
        assert!(
            (beta0 - true_beta0).abs() < 0.001,
            "β₀: expected {true_beta0}, got {beta0:.6}"
        );
    }

    #[test]
    fn usl_fit_fallback_when_m_less_than_3() {
        let (alpha, beta0) = CalibrationHarness::usl_fit(1.0, 0.8, 2, 0.8, 0.12, 0.01);
        assert_eq!(alpha, 0.12, "fallback α when M=2");
        assert_eq!(beta0, 0.01, "fallback β₀ when M=2");
    }

    #[test]
    fn usl_fit_fallback_when_m_is_1() {
        let (alpha, beta0) = CalibrationHarness::usl_fit(1.0, 1.0, 1, 1.0, 0.12, 0.01);
        assert_eq!(alpha, 0.12);
        assert_eq!(beta0, 0.01);
    }

    #[test]
    fn usl_fit_fallback_on_degenerate_timing() {
        // t1 = 0 → degenerate
        let (alpha, beta0) = CalibrationHarness::usl_fit(0.0, 0.5, 4, 0.5, 0.12, 0.01);
        assert_eq!(alpha, 0.12);
        assert_eq!(beta0, 0.01);
    }

    #[test]
    fn usl_fit_fallback_on_negative_derived_params() {
        // Super-linear speedup at N=2 → negative alpha → use fallback
        let t1 = 1.0_f64;
        let t2_superlinear = 0.3; // X(2) ≈ 3.33 > 2 → super-linear → degenerate
        let t4 = 0.5;
        let (alpha, beta0) = CalibrationHarness::usl_fit(t1, t2_superlinear, 4, t4, 0.12, 0.01);
        assert_eq!(alpha, 0.12, "super-linear speedup must trigger fallback");
        assert_eq!(beta0, 0.01);
    }

    #[test]
    fn usl_fit_clamps_extreme_values() {
        // Ground truth: α=0.8, β₀=0.02 — both positive but α > 0.5 clamp ceiling.
        // β₀ = 0.02 is already within [1e-6, 0.1] so only α gets clamped.
        let true_alpha = 0.8_f64;
        let true_beta0 = 0.02_f64;
        let t1 = 1.0_f64;
        let t2 = t1 / usl_throughput(2.0, true_alpha, true_beta0);
        let t4 = t1 / usl_throughput(4.0, true_alpha, true_beta0);

        let (alpha, beta0) = CalibrationHarness::usl_fit(t1, t2, 4, t4, 0.12, 0.01);
        // α pre-clamp = 0.8 → clamped to 0.5; fallback not taken (both params positive)
        assert_eq!(alpha, 0.5, "α=0.8 must be clamped to 0.5");
        // beta0 = 0.02 is within range, not the fallback value 0.01
        assert!(beta0 >= 1e-6 && beta0 <= 0.1, "beta0 out of clamped range: {beta0}");
        assert!(
            (beta0 - 0.01).abs() > 1e-6,
            "beta0 must not be the fallback value — clamp path must be taken"
        );
    }
}
