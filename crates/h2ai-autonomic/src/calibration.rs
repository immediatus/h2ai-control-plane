use chrono::Utc;
use futures::future::join_all;
use h2ai_config::H2AIConfig;
use h2ai_context::embedding::{cosine_similarity, semantic_jaccard, EmbeddingModel};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::events::CalibrationCompletedEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::physics::{
    tau_alignment, CoherencyCoefficients, CoordinationThreshold, EigenCalibration,
    EnsembleCalibration, PhysicsError, TauValue,
};
use nalgebra::DMatrix;
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
    /// Optional embedding model for semantic CG measurement.
    /// When `None`, falls back to token-level Jaccard (zero extra cost).
    pub embedding_model: Option<&'a dyn EmbeddingModel>,
}

pub struct CalibrationHarness;

impl CalibrationHarness {
    pub async fn run(
        input: CalibrationInput<'_>,
    ) -> Result<CalibrationCompletedEvent, CalibrationError> {
        if input.adapters.is_empty() {
            return Err(CalibrationError::NoAdapters);
        }
        let m = input.adapters.len();
        let taus_m = Self::tau_spread(m, input.cfg);

        // Phase A: run the first 2 adapters in parallel to get T₂ and per-adapter times for T₁.
        // Phase B: run all M adapters in parallel to get T_M and all outputs for CG_mean.
        // When M < 2, skip Phase A and use a single run as both phases.
        let (t1_proxy, t2_parallel, t_m_parallel, adapter_outputs) = if m >= 2 {
            let taus_a = Self::tau_spread(2, input.cfg);
            let (phase_a_outputs, t2_wall) = Self::run_adapters_parallel(
                &input.adapters[..2],
                &input.task_prompts,
                &taus_a,
                input.cfg,
            )
            .await?;
            // T₁ = mean per-adapter serial time (approximation of single-adapter cost)
            let t1 = phase_a_outputs.iter().map(|(_, t)| *t).sum::<f64>() / 2.0;

            let (all_outputs, t_m_wall) = Self::run_adapters_parallel(
                &input.adapters,
                &input.task_prompts,
                &taus_m,
                input.cfg,
            )
            .await?;
            let outputs: Vec<Vec<String>> = all_outputs.into_iter().map(|(o, _)| o).collect();
            (t1, t2_wall, t_m_wall, outputs)
        } else {
            // M == 1: no parallelism to measure; use fallback parameters.
            let (single_out, t_single) = Self::run_adapters_parallel(
                &input.adapters,
                &input.task_prompts,
                &taus_m,
                input.cfg,
            )
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
        // Record one timestamp for the entire calibration run — all pairs are computed
        // simultaneously so a single timestamp per run is the right granularity.
        let calibration_ts = Utc::now().timestamp() as u64;
        let (cg_samples, cg_timestamps, ensemble, pairwise_beta) = if adapter_outputs.len() < 2 {
            (
                vec![input.cfg.calibration_cg_fallback],
                vec![calibration_ts],
                None,
                None,
            )
        } else {
            let cal_tau =
                TauValue::new(input.cfg.calibration_tau).expect("calibration_tau must be in [0,1]");
            let align = tau_alignment(cal_tau, cal_tau); // = 1.0 when all taus equal

            let mut pairs = Vec::new();
            let pairwise_start = Instant::now();
            for i in 0..adapter_outputs.len() {
                for j in (i + 1)..adapter_outputs.len() {
                    pairs.push(Self::adapter_pair_cg(
                        &adapter_outputs[i],
                        &adapter_outputs[j],
                        input.embedding_model,
                        align,
                        input.cfg.cg_agreement_threshold,
                    ));
                }
            }
            let pairwise_elapsed = pairwise_start.elapsed().as_secs_f64();
            let n_pairs = pairs.len();
            let pairwise_beta = if n_pairs > 0 && t1_proxy > 1e-9 {
                let per_pair = pairwise_elapsed / n_pairs as f64;
                Some((per_pair / t1_proxy).clamp(1e-9, 0.1))
            } else {
                None
            };
            let cg_mean_val: f64 = pairs.iter().sum::<f64>() / n_pairs as f64;
            let ec = if input.cfg.baseline_accuracy_proxy > 0.0 {
                EnsembleCalibration::from_measured_p(
                    input.cfg.baseline_accuracy_proxy,
                    cg_mean_val,
                    9,
                )
            } else {
                EnsembleCalibration::from_cg_mean(cg_mean_val, 9)
            };
            (
                pairs,
                vec![calibration_ts; n_pairs],
                Some(ec),
                pairwise_beta,
            )
        };

        // Compute eigenvalue calibration from the full pairwise CG matrix (N×N).
        let eigen: Option<EigenCalibration> = if adapter_outputs.len() >= 2 {
            let n = adapter_outputs.len();
            let cal_tau = TauValue::new(input.cfg.calibration_tau).expect("calibration_tau valid");
            let align = tau_alignment(cal_tau, cal_tau);
            let mut sigma = DMatrix::<f64>::identity(n, n);
            for i in 0..n {
                for j in (i + 1)..n {
                    let cg_ij = Self::adapter_pair_cg(
                        &adapter_outputs[i],
                        &adapter_outputs[j],
                        input.embedding_model,
                        align,
                        input.cfg.cg_agreement_threshold,
                    );
                    sigma[(i, j)] = cg_ij;
                    sigma[(j, i)] = cg_ij;
                }
            }
            Some(EigenCalibration::from_cg_matrix(&sigma))
        } else {
            None
        };

        let cc = CoherencyCoefficients::new_with_timestamps(
            alpha,
            beta_base,
            cg_samples,
            cg_timestamps,
        )?;
        let coordination_threshold =
            CoordinationThreshold::from_calibration(&cc, input.cfg.coordination_threshold_max);

        Ok(CalibrationCompletedEvent {
            calibration_id: input.calibration_id,
            coefficients: cc,
            coordination_threshold,
            ensemble,
            eigen,
            timestamp: Utc::now(),
            pairwise_beta,
            cg_mode: if input.embedding_model.is_some() {
                h2ai_types::events::CgMode::EmbeddingCosine
            } else {
                h2ai_types::events::CgMode::TokenJaccard
            },
            adapter_families: Vec::new(),
            explorer_verification_family_match: false,
            single_family_warning: false,
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

    /// Compute CG(i,j) as the embedding cosine agreement rate between two adapters.
    ///
    /// **With embedding model** (blog measurement): fraction of calibration prompts where
    /// `cosine(embed_i[k], embed_j[k]) > agreement_threshold`. This matches the blog's
    /// "agreement rate on calibration set" definition.
    ///
    /// **Without model** (fallback): mean per-prompt token Jaccard similarity, preserving
    /// existing behavior when no embedding model is configured.
    fn adapter_pair_cg(
        outputs_i: &[String],
        outputs_j: &[String],
        model: Option<&dyn EmbeddingModel>,
        align: f64,
        agreement_threshold: f64,
    ) -> f64 {
        if outputs_i.is_empty() || outputs_i.len() != outputs_j.len() {
            let oi = outputs_i.join(" ");
            let oj = outputs_j.join(" ");
            return semantic_jaccard(&oi, &oj, model) * align;
        }
        let cg = match model {
            Some(m) => {
                let agree = outputs_i
                    .iter()
                    .zip(outputs_j.iter())
                    .filter(|(oi, oj)| {
                        let vi = m.embed(oi);
                        let vj = m.embed(oj);
                        cosine_similarity(&vi, &vj) > agreement_threshold
                    })
                    .count();
                agree as f64 / outputs_i.len() as f64
            }
            None => {
                outputs_i
                    .iter()
                    .zip(outputs_j.iter())
                    .map(|(oi, oj)| semantic_jaccard(oi, oj, None))
                    .sum::<f64>()
                    / outputs_i.len() as f64
            }
        };
        cg * align
    }

    /// Compute linear τ spacing across M calibration adapters.
    ///
    /// With M > 1, spaces τ linearly from `cfg.calibration_tau_spread[0]` to
    /// `cfg.calibration_tau_spread[1]`. With M == 1, returns `[calibration_tau]`.
    /// Diversifies CG measurement by removing temperature-correlation bias.
    fn tau_spread(m: usize, cfg: &H2AIConfig) -> Vec<TauValue> {
        if m <= 1 {
            return vec![TauValue::new(cfg.calibration_tau).expect("calibration_tau valid")];
        }
        let (tau_min, tau_max) = (cfg.calibration_tau_spread[0], cfg.calibration_tau_spread[1]);
        (0..m)
            .map(|i| {
                let t = tau_min + (tau_max - tau_min) * (i as f64 / (m - 1) as f64);
                TauValue::new(t).expect("tau spread must be in [0,1]")
            })
            .collect()
    }

    /// Run a slice of adapters concurrently on all prompts.
    ///
    /// Each adapter runs at `taus[i]`; if `taus` is shorter than `adapters`,
    /// the last value in `taus` is reused. Returns (per-adapter (outputs, elapsed_secs), wall_clock_secs).
    async fn run_adapters_parallel(
        adapters: &[&dyn IComputeAdapter],
        prompts: &[String],
        taus: &[TauValue],
        cfg: &H2AIConfig,
    ) -> Result<(Vec<(Vec<String>, f64)>, f64), CalibrationError> {
        let tau_fallback = taus
            .last()
            .copied()
            .unwrap_or_else(|| TauValue::new(cfg.calibration_tau).expect("calibration_tau valid"));
        let t_wall_start = Instant::now();
        let futures: Vec<_> = adapters
            .iter()
            .enumerate()
            .map(|(i, adapter)| {
                let tau = taus.get(i).copied().unwrap_or(tau_fallback);
                async move {
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
                }
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

/// Derive β₀ from a set of merge phase timings.
///
/// `spans`: each tuple is `(merge_elapsed_secs, n_proposals)` from a
/// `SemilatticeCompiledEvent`. `n_proposals` is `n_input_proposals`.
/// `t1_secs`: serial T₁ from `CalibrationHarness` (the API call time proxy).
///
/// Formula: β₀ = mean(elapsed_i / pairs_i) / T₁
/// where pairs_i = max(1, n_i × (n_i − 1) / 2).
///
/// **Note:** This denominator models O(n²) pairwise work and is accurate for
/// `OutlierResistant`/`MultiOutlierResistant` merge strategies. For `ScoreOrdered` (O(n log n)) and
/// `ConsensusMedian`, the derived β₀ will be inflated. Prefer collecting spans
/// from OutlierResistant-strategy merges when using this function for USL fitting.
///
/// Returns `None` when `spans` is empty or `t1_secs` ≤ 0. Clamps to [1e-9, 0.1].
pub fn beta_from_merge_spans(spans: &[(f64, usize)], t1_secs: f64) -> Option<f64> {
    // Guard: t1_secs rounds to sub-nanosecond on mock/in-process adapters in fast CI runs.
    // In that case pairwise_beta is undefined (any β / 0 → ∞); return None conservatively.
    if spans.is_empty() || t1_secs < 1e-9 {
        return None;
    }
    let sum: f64 = spans
        .iter()
        .map(|&(elapsed, n)| {
            let pairs = (n * n.saturating_sub(1) / 2).max(1) as f64;
            elapsed / pairs
        })
        .sum();
    let mean_per_pair = sum / spans.len() as f64;
    Some((mean_per_pair / t1_secs).clamp(1e-9, 0.1))
}

/// Derive β₀ from token costs of merge phase executions.
///
/// `spans`: each tuple is `(merge_tokens_consumed, n_proposals)` from a merge phase.
/// `t1_tokens`: mean tokens per single adapter response (baseline cost unit).
///
/// Formula: β₀ = mean(tokens_i / pairs_i) / t1_tokens
/// where pairs_i = max(1, n_i × (n_i − 1) / 2).
///
/// Token cost is the correct USL β₀ analog per Gunther: it measures coherency overhead
/// as a fraction of the baseline processing cost, independent of network I/O latency.
///
/// Returns `None` when `spans` is empty or `t1_tokens` is 0. Clamps to [1e-9, 0.1].
pub fn beta_from_token_spans(spans: &[(u64, usize)], t1_tokens: u64) -> Option<f64> {
    if spans.is_empty() || t1_tokens == 0 {
        return None;
    }
    let sum: f64 = spans
        .iter()
        .map(|&(tokens, n)| {
            let pairs = (n * n.saturating_sub(1) / 2).max(1) as f64;
            tokens as f64 / pairs
        })
        .sum();
    let mean_per_pair = sum / spans.len() as f64;
    Some((mean_per_pair / t1_tokens as f64).clamp(1e-9, 0.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_context::embedding::EmbeddingModel;

    // ── StubEmbeddingModel for CG agreement rate tests ──────────────────────

    /// Returns a fixed L2-normalised vector per text cluster.
    /// Texts starting with 'A' → auth cluster, 'B' → redis cluster.
    struct StubEmbeddingModel;
    impl EmbeddingModel for StubEmbeddingModel {
        fn embed(&self, text: &str) -> Vec<f32> {
            if text.starts_with('A') {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            }
        }
    }

    #[test]
    fn cg_embed_paraphrase_agreement_is_high() {
        // Two adapters producing semantically identical outputs (same cluster) on every prompt
        // → each prompt passes the agreement threshold → CG ≈ 1.0
        let outputs_i = vec![
            "A auth stateless jwt".into(),
            "A token rotation".into(),
            "A bearer ADR-001".into(),
        ];
        let outputs_j = vec![
            "A jwt auth bearer".into(),
            "A stateless token".into(),
            "A ADR-001 auth".into(),
        ];
        let model = StubEmbeddingModel;
        let cg =
            CalibrationHarness::adapter_pair_cg(&outputs_i, &outputs_j, Some(&model), 1.0, 0.85);
        assert!(
            cg > 0.9,
            "paraphrase adapters must score CG_embed ≈ 1.0, got {cg:.3}"
        );
    }

    #[test]
    fn cg_embed_divergent_adapters_is_low() {
        // Two adapters systematically producing outputs in different clusters
        // → cosine=0 on every prompt → CG = 0
        let outputs_i = vec!["A auth stateless jwt".into(), "A token rotation".into()];
        let outputs_j = vec!["B redis cache store".into(), "B key-value expiry".into()];
        let model = StubEmbeddingModel;
        let cg =
            CalibrationHarness::adapter_pair_cg(&outputs_i, &outputs_j, Some(&model), 1.0, 0.85);
        assert!(
            cg < 0.2,
            "divergent adapters must score CG_embed ≈ 0, got {cg:.3}"
        );
    }

    #[test]
    fn cg_embed_no_model_falls_back_to_jaccard() {
        // Without model, CG is mean per-prompt Jaccard; identical outputs → CG = 1.0
        let text = "stateless jwt auth token ADR-001";
        let outputs = vec![text.to_string(); 3];
        let cg = CalibrationHarness::adapter_pair_cg(&outputs, &outputs, None, 1.0, 0.85);
        assert!(
            (cg - 1.0).abs() < 1e-9,
            "identical outputs with no model must score CG=1.0, got {cg:.6}"
        );
    }

    #[test]
    fn cg_embed_align_scales_result() {
        // CG_embed × align: with align=0.5, output is half of raw CG
        let outputs_i = vec!["A auth stateless jwt".into()];
        let outputs_j = vec!["A jwt auth bearer".into()];
        let model = StubEmbeddingModel;
        let cg_full =
            CalibrationHarness::adapter_pair_cg(&outputs_i, &outputs_j, Some(&model), 1.0, 0.85);
        let cg_half =
            CalibrationHarness::adapter_pair_cg(&outputs_i, &outputs_j, Some(&model), 0.5, 0.85);
        assert!(
            (cg_half - cg_full * 0.5).abs() < 1e-9,
            "align=0.5 must halve CG: {cg_half:.3} vs {cg_full:.3}/2"
        );
    }

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
        assert!(
            beta0 >= 1e-6 && beta0 <= 0.1,
            "beta0 out of clamped range: {beta0}"
        );
        assert!(
            (beta0 - 0.01).abs() > 1e-6,
            "beta0 must not be the fallback value — clamp path must be taken"
        );
    }

    // ── beta_from_token_spans tests ──────────────────────────────────────────

    #[test]
    fn beta_from_token_spans_basic() {
        // 1 span: 100 tokens consumed for 5 proposals → 10 pairs → per_pair = 10
        // t1_tokens = 500 → β₀ = 10/500 = 0.02
        let spans = vec![(100u64, 5usize)];
        let beta = beta_from_token_spans(&spans, 500).unwrap();
        assert!(
            (beta - 0.02).abs() < 1e-9,
            "expected β₀=0.02, got {beta:.8}"
        );
    }

    #[test]
    fn beta_from_token_spans_clamps_to_max() {
        // Pathological: many tokens for 2 proposals → 1 pair → enormous per_pair → clamps to 0.1
        let spans = vec![(1_000_000u64, 2usize)];
        let beta = beta_from_token_spans(&spans, 1).unwrap();
        assert_eq!(beta, 0.1, "must clamp to max 0.1");
    }

    #[test]
    fn beta_from_token_spans_none_on_empty() {
        assert!(beta_from_token_spans(&[], 100).is_none());
    }

    #[test]
    fn beta_from_token_spans_none_on_zero_t1() {
        assert!(beta_from_token_spans(&[(50, 3)], 0).is_none());
    }

    #[test]
    fn beta_from_token_spans_multi_span_is_mean() {
        // Two spans with the same per-pair ratio → mean equals that ratio
        // span A: 200 tokens, 3 proposals → 3 pairs → per_pair = 200/3
        // span B: 600 tokens, 3 proposals → 3 pairs → per_pair = 600/3 = 200
        // Actually: pairs = 3*(3-1)/2 = 3
        // span A: per_pair = 200/3; span B: per_pair = 600/3 = 200
        // mean_per_pair = (200/3 + 200)/2 = (200/3 + 600/3)/2 = 800/6 = 133.33
        // β₀ = 133.33 / t1_tokens; use t1_tokens = 1333 → β₀ ≈ 0.1, then it'll clamp
        // Use t1_tokens=10000 so result is within range
        // mean_per_pair = (200/3 + 200)/2 ≈ 133.33
        // β₀ = 133.33/10000 ≈ 0.01333
        let spans = vec![(200u64, 3usize), (600u64, 3usize)];
        let beta = beta_from_token_spans(&spans, 10_000).unwrap();
        let expected = (200.0_f64 / 3.0 + 600.0 / 3.0) / 2.0 / 10_000.0;
        assert!(
            (beta - expected).abs() < 1e-9,
            "multi-span mean: expected {expected:.8}, got {beta:.8}"
        );
    }

    // ── tau_spread tests ─────────────────────────────────────────────────────

    #[test]
    fn tau_spread_m1_returns_calibration_tau() {
        let cfg = H2AIConfig::default();
        let taus = CalibrationHarness::tau_spread(1, &cfg);
        assert_eq!(taus.len(), 1);
        assert!((taus[0].value() - cfg.calibration_tau).abs() < 1e-9);
    }

    #[test]
    fn tau_spread_m3_produces_distinct_values() {
        let cfg = H2AIConfig::default();
        let taus = CalibrationHarness::tau_spread(3, &cfg);
        assert_eq!(taus.len(), 3);
        // First and last must be spread endpoints
        assert!((taus[0].value() - cfg.calibration_tau_spread[0]).abs() < 1e-9);
        assert!((taus[2].value() - cfg.calibration_tau_spread[1]).abs() < 1e-9);
        // Middle must differ from both endpoints
        assert!(taus[1].value() > taus[0].value());
        assert!(taus[1].value() < taus[2].value());
    }

    #[test]
    fn tau_spread_all_in_range() {
        let cfg = H2AIConfig::default();
        for m in 2..=6 {
            let taus = CalibrationHarness::tau_spread(m, &cfg);
            assert_eq!(taus.len(), m);
            for tau in &taus {
                assert!(
                    tau.value() >= cfg.calibration_tau_spread[0]
                        && tau.value() <= cfg.calibration_tau_spread[1],
                    "τ={} out of spread [{}, {}]",
                    tau.value(),
                    cfg.calibration_tau_spread[0],
                    cfg.calibration_tau_spread[1]
                );
            }
        }
    }
}
