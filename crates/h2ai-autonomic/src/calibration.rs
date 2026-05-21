#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines
)]
use chrono::Utc;
use futures::future::join_all;
use h2ai_config::H2AIConfig;
use h2ai_constraints::eval::eval_sync;
use h2ai_constraints::types::{ConstraintDoc, ConstraintSeverity};
use h2ai_context::embedding::{cosine_similarity, semantic_jaccard, EmbeddingModel};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::events::{
    CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode,
};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{
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

/// Inputs required to run a single calibration pass over an adapter ensemble.
///
/// All fields are borrowed for the lifetime of the pass; the harness does not
/// own the adapters or configuration, so callers can reuse them across retries.
pub struct CalibrationInput<'a> {
    /// Stable identifier that links this calibration to downstream events (e.g. `CalibrationCompletedEvent`).
    pub calibration_id: TaskId,
    /// Representative prompts sent to every adapter during both Phase A and Phase B runs.
    ///
    /// Diversity here improves CG measurement quality; a single prompt can produce
    /// artificially high agreement through topic correlation alone.
    pub task_prompts: Vec<String>,
    /// Ensemble of adapters under calibration; must be non-empty.
    pub adapters: Vec<&'a dyn IComputeAdapter>,
    /// Runtime configuration supplying USL fallback parameters, τ spread, and CG thresholds.
    pub cfg: &'a H2AIConfig,
    /// Constraint corpus for Hamming-distance CG measurement.
    /// Empty slice → CG falls back to `cfg.calibration_cg_fallback`.
    pub constraint_corpus: &'a [ConstraintDoc],
    /// Optional embedding model for semantic CG measurement.
    ///
    /// When `Some`, CG(i,j) is computed as the fraction of calibration prompts where
    /// `cosine(embed(output_i), embed(output_j)) > cfg.cg_agreement_threshold` — the
    /// theoretically correct formula that is paraphrase-insensitive.
    /// When `None`, falls back to Hamming distance on constraint fingerprints.
    pub embedding_model: Option<&'a dyn EmbeddingModel>,
}

/// Stateless entry-point for running USL-based adapter calibration.
///
/// A single `run` call exercises all adapters in two parallel phases, derives
/// α and β₀ from wall-clock timings, computes pairwise CG scores, and packages
/// the result into a `CalibrationCompletedEvent` ready for NATS publication.
pub struct CalibrationHarness;

impl CalibrationHarness {
    /// Run a full USL calibration pass and return a `CalibrationCompletedEvent`.
    ///
    /// Phase A runs the first two adapters in parallel to obtain T₂ and a T₁ proxy.
    /// Phase B runs all M adapters in parallel to obtain `T_M` and per-adapter outputs
    /// for pairwise CG computation.  α and β₀ are derived from `usl_fit`; when the
    /// fit is degenerate (M < 3 or non-positive timings), `cfg.alpha_contention` and
    /// `cfg.beta_base_default` are used as fallback values.
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

        // CG_mean: use embedding cosine when a model is available; fall back to
        // constraint-profile Hamming otherwise.  Both paths run at calibration time
        // (once per calibration call) — not per synthesis call.
        let calibration_ts = Utc::now().timestamp() as u64;
        let cg_mode = if input.embedding_model.is_some() {
            CgMode::EmbeddingCosine
        } else {
            CgMode::ConstraintProfile
        };
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
                        input.constraint_corpus,
                        input.embedding_model,
                        input.cfg,
                        align,
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
                    input.cfg.calibration_max_ensemble_size,
                )
            } else {
                EnsembleCalibration::from_cg_mean(
                    cg_mean_val,
                    input.cfg.calibration_max_ensemble_size,
                )
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
                        input.constraint_corpus,
                        input.embedding_model,
                        input.cfg,
                        align,
                    );
                    sigma[(i, j)] = cg_ij;
                    sigma[(j, i)] = cg_ij;
                }
            }
            let sigma_clone = sigma.clone();
            let eigen_delta = input.cfg.eigen_n_eff_delta;
            let ec = tokio::task::spawn_blocking(move || {
                EigenCalibration::from_cg_matrix(&sigma_clone, eigen_delta)
            })
            .await
            .expect("CG eigenvalue computation panicked");
            Some(ec)
        } else {
            None
        };

        // ── Cosine N_eff prior — semantic independence of the adapter pool ────
        // Same adapter_outputs collected above; one extra embedding pass, no extra LLM calls.
        let n_eff_cosine_prior: f64 = if let Some(model) = input.embedding_model {
            let n = adapter_outputs.len();
            let k_prompts = if n > 0 { adapter_outputs[0].len() } else { 0 };
            if n >= 2 && k_prompts > 0 {
                use h2ai_types::sizing::EigenCalibration;
                use nalgebra::DMatrix;
                // Accumulate pairwise cosine sums over K prompts.
                let mut c = DMatrix::<f64>::zeros(n, n);
                // ki is a column index used across multiple rows (adapter_outputs[i][ki],
                // adapter_outputs[j][ki]) — a range loop is the correct pattern here.
                #[allow(clippy::needless_range_loop)]
                for ki in 0..k_prompts {
                    for i in 0..n {
                        c[(i, i)] += 1.0;
                        for j in (i + 1)..n {
                            let sim = cosine_similarity(
                                &model.embed(&adapter_outputs[i][ki]),
                                &model.embed(&adapter_outputs[j][ki]),
                            )
                            .max(0.0);
                            c[(i, j)] += sim;
                            c[(j, i)] += sim;
                        }
                    }
                }
                // C[i][j] = mean over K prompts; normalise K_norm = C_avg / N so trace = 1
                let c_avg = c / k_prompts as f64;
                let k_norm = c_avg / n as f64;
                let k_clone = k_norm.clone();
                let cosine_delta = input.cfg.eigen_n_eff_delta;
                tokio::task::spawn_blocking(move || {
                    EigenCalibration::from_cosine_matrix(&k_clone, cosine_delta)
                })
                .await
                .expect("cosine eigenvalue computation panicked")
                .n_effective
            } else {
                1.0
            }
        } else {
            // No embedding model: fallback formula
            let n = adapter_outputs.len().max(1) as f64;
            input
                .cfg
                .calibration_cg_fallback
                .mul_add(n - 1.0, 1.0)
                .min(n)
        };
        // ─────────────────────────────────────────────────────────────────────

        // ── Epistemic β₀ override ──────────────────────────────────────────────
        // When an embedding model is available and we have ≥ 3 adapters, replace
        // the timing-derived β₀ with the USL constraint-inversion estimate.
        // This produces physically grounded friction consistent with observed N_eff.
        let beta_base = if input.embedding_model.is_some() && m >= 3 {
            let cg_mean = if cg_samples.is_empty() {
                input.cfg.calibration_cg_fallback
            } else {
                cg_samples.iter().copied().sum::<f64>() / cg_samples.len() as f64
            };
            let k = input.cfg.calibration_probe.neff_cg_exponent;
            beta_from_n_eff_adj(n_eff_cosine_prior, cg_mean, m, k)
        } else {
            beta_base
        };
        // ─────────────────────────────────────────────────────────────────────

        let usl_from_fallback = m < 3;
        let cg_from_fallback = adapter_outputs.len() < 2;
        let calibration_source = match (usl_from_fallback, cg_from_fallback) {
            (false, false) => CalibrationSource::Measured,
            (true, true) => CalibrationSource::SyntheticPriors,
            _ => CalibrationSource::PartialFit,
        };

        let cc = CoherencyCoefficients::new_with_timestamps(
            alpha,
            beta_base,
            cg_samples,
            cg_timestamps,
        )?;
        let coordination_threshold =
            CoordinationThreshold::from_calibration(&cc, input.cfg.coordination_threshold_max);
        let (n_max_lo, n_max_hi) = cc.n_max_ci();

        // Compute beta_quality from pairwise constraint conflict rate (Phase B outputs).
        // Uses first prompt output per adapter; calibration typically uses one prompt.
        let beta_quality = if adapter_outputs.len() >= 2 && !input.constraint_corpus.is_empty() {
            let first_outputs: Vec<&str> = adapter_outputs
                .iter()
                .filter_map(|outputs| outputs.first().map(std::string::String::as_str))
                .collect();
            compute_conflict_rate(&first_outputs, input.constraint_corpus)
        } else {
            None
        };

        Ok(CalibrationCompletedEvent {
            calibration_id: input.calibration_id,
            coefficients: cc,
            coordination_threshold,
            ensemble,
            eigen,
            timestamp: Utc::now(),
            pairwise_beta,
            cg_mode,
            adapter_families: {
                input
                    .adapters
                    .iter()
                    .map(|a| a.kind().family().to_string())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect()
            },
            explorer_verification_family_match: {
                let families: std::collections::HashSet<_> =
                    input.adapters.iter().map(|a| a.kind().family()).collect();
                families.len() > 1
            },
            single_family_warning: {
                let families: std::collections::HashSet<_> =
                    input.adapters.iter().map(|a| a.kind().family()).collect();
                families.len() == 1
            },
            n_max_lo,
            n_max_hi,
            n_eff_cosine_prior,
            calibration_quality: CalibrationQuality::default(),
            calibration_source,
            beta_quality,
        })
    }

    /// Derive USL parameters α and β₀ analytically from two parallel timing measurements.
    ///
    /// Uses the linearisation z(N) = `N·T_parallel(N)/T₁` − 1 = α(N−1) + β₀·N(N−1).
    /// With two data points at N=2 (Phase A) and N=M (Phase B):
    ///   β₀ = (`z_M` − z₂·(M−1)) / ((M−1)(M−2))
    ///   α  = z₂ − 2·β₀
    ///
    /// Falls back to (`alpha_fallback`, `beta_fallback`) when:
    /// - M < 3 (denominator (M−1)(M−2) is zero at M=2)
    /// - any timing is degenerate (≤ 0)
    /// - derived α or β₀ are negative (super-linear speedup or measurement noise)
    #[must_use]
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
        let alpha = 2.0f64.mul_add(-beta0, z2);

        // Negative params indicate degenerate measurement (e.g. super-linear speedup).
        // Must check before clamping — clamping would mask the degenerate case.
        if beta0 < 0.0 || alpha < 0.0 {
            return (alpha_fallback, beta_fallback);
        }

        (alpha.clamp(0.05, 0.5), beta0.clamp(1e-6, 0.1))
    }

    fn adapter_pair_cg(
        outputs_i: &[String],
        outputs_j: &[String],
        corpus: &[ConstraintDoc],
        embedding_model: Option<&dyn EmbeddingModel>,
        cfg: &H2AIConfig,
        align: f64,
    ) -> f64 {
        if outputs_i.is_empty() || outputs_i.len() != outputs_j.len() {
            return cfg.calibration_cg_fallback * align;
        }
        let scores: Vec<f64> = outputs_i
            .iter()
            .zip(outputs_j.iter())
            .map(|(oi, oj)| {
                if let Some(model) = embedding_model {
                    // Embedding cosine: fraction of prompts where cosine > θ_agree.
                    let sim = semantic_jaccard(oi, oj, Some(model));
                    if sim >= cfg.cg_agreement_threshold {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    if corpus.is_empty() {
                        return cfg.calibration_cg_fallback;
                    }
                    let fp_i = constraint_fingerprint(oi, corpus);
                    let fp_j = constraint_fingerprint(oj, corpus);
                    hamming_distance(&fp_i, &fp_j)
                }
            })
            .collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        mean * align
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
                let t = (tau_max - tau_min).mul_add(i as f64 / (m - 1) as f64, tau_min);
                TauValue::new(t).expect("tau spread must be in [0,1]")
            })
            .collect()
    }

    /// Run a slice of adapters concurrently on all prompts.
    ///
    /// Each adapter runs at `taus[i]`; if `taus` is shorter than `adapters`,
    /// the last value in `taus` is reused. Returns (per-adapter (outputs, `elapsed_secs`), `wall_clock_secs`).
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
                    for prompt in prompts {
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

/// Compute the mean pairwise Hamming distance over M constraint fingerprints.
///
/// Each fingerprint is a boolean vector where `true` = proposal passed the hard gate
/// for that constraint. Returns `None` when fewer than 2 outputs are given or the
/// corpus is empty. Result is clamped to `[1e-6, 1.0]`.
#[must_use]
pub fn compute_conflict_rate(outputs: &[&str], corpus: &[ConstraintDoc]) -> Option<f64> {
    let m = outputs.len();
    if m < 2 || corpus.is_empty() {
        return None;
    }
    let fingerprints: Vec<Vec<bool>> = outputs
        .iter()
        .map(|o| constraint_fingerprint(o, corpus))
        .collect();
    let mut total = 0.0f64;
    let mut count = 0usize;
    for i in 0..m {
        for j in (i + 1)..m {
            total += hamming_distance(&fingerprints[i], &fingerprints[j]);
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    Some((total / count as f64).clamp(1e-6, 1.0))
}

fn constraint_fingerprint(output: &str, corpus: &[ConstraintDoc]) -> Vec<bool> {
    corpus
        .iter()
        .map(|doc| {
            let score = eval_sync(&doc.predicate, output);
            match &doc.severity {
                ConstraintSeverity::Hard { threshold } => score >= *threshold,
                _ => true,
            }
        })
        .collect()
}

fn hamming_distance(a: &[bool], b: &[bool]) -> f64 {
    // 1.0 on degenerate input: fail open toward diversity rather than collapsing CG.
    // Callers guard corpus consistency; this branch only fires on bugs, not normal operation.
    if a.is_empty() || a.len() != b.len() {
        return 1.0;
    }
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as f64 / a.len() as f64
}

/// Derive β₀ from a set of merge phase timings.
///
/// `spans`: each tuple is `(merge_elapsed_secs, n_proposals)` from a
/// `SelectionResolvedEvent`. `n_proposals` is `n_input_proposals`.
/// `t1_secs`: serial T₁ from `CalibrationHarness` (the API call time proxy).
///
/// Formula: β₀ = `mean(elapsed_i` / `pairs_i`) / T₁
/// where `pairs_i` = max(1, `n_i` × (`n_i` − 1) / 2).
///
/// **Note:** This denominator models O(n²) pairwise work and is accurate for
/// `OutlierResistant`/`MultiOutlierResistant` merge strategies. For `ScoreOrdered` (O(n log n)) and
/// `ConsensusMedian`, the derived β₀ will be inflated. Prefer collecting spans
/// from OutlierResistant-strategy merges when using this function for USL fitting.
///
/// Returns `None` when `spans` is empty or `t1_secs` ≤ 0. Clamps to [1e-9, 0.1].
#[must_use]
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

/// Additive-increase step: decay α toward the measured yield.
///
/// On a successful iteration, α moves toward `alpha_measured` at rate `decay_rate`:
/// `α_next = max(α_current × decay_rate, alpha_measured)`
///
/// `decay_rate` ∈ (0, 1] — typical value 0.95.
/// Returns `alpha_measured` when `α_current × decay_rate < alpha_measured`.
#[must_use]
pub fn aimd_decay(alpha_current: f64, alpha_measured: f64, decay_rate: f64) -> f64 {
    (alpha_current * decay_rate).max(alpha_measured)
}

/// Multiplicative-decrease step: reset α when yield falls below threshold.
///
/// On a poor iteration (yield < `reset_threshold`), α jumps back toward `seed_alpha`:
/// `α_next = min(α_current × reset_multiplier, seed_alpha)`
///
/// `reset_multiplier` > 1 — typical value 3.0.
/// Returns `seed_alpha` when `α_current × reset_multiplier > seed_alpha`.
#[must_use]
pub fn aimd_reset(alpha_current: f64, seed_alpha: f64, reset_multiplier: f64) -> f64 {
    (alpha_current * reset_multiplier).min(seed_alpha)
}

/// Compute the mean yield from a `n_useful_history` ring buffer.
///
/// Each entry is `(n_useful: u8, n_max: u8, _unix_minutes: u32)`.
/// Yield = `n_useful` / `n_max` for each entry. Returns `None` when `history` is empty
/// or all entries have `n_max == 0`.
#[must_use]
pub fn yield_from_history(history: &[(u8, u8, u32)]) -> Option<f64> {
    if history.is_empty() {
        return None;
    }
    let (sum, count) = history
        .iter()
        .fold((0.0_f64, 0_usize), |(s, c), &(n_useful, n_max, _)| {
            if n_max == 0 {
                (s, c)
            } else {
                (s + f64::from(n_useful) / f64::from(n_max), c + 1)
            }
        });
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

/// Compute epistemic β₀ from adjusted `N_eff` using the USL constraint inversion formula.
///
/// Returns the Kontrollier friction parameter β₀ such that when `N_eff` agents collaborate
/// with group-coherence CG, `N_max = sqrt((1 - α) / β_eff)` is physically consistent.
///
/// # Formula
/// `N_eff_adj = clamp(N_eff × CG^k, 1.0, N_cal)`
/// `β₀ = max((1/N_eff_adj − 1/N_cal) / (N_cal − 1), 1e-6)`
///
/// Returns `1e-6` when `n_cal < 2` (degenerate: can't fit USL with fewer than 2 calibration samples).
#[must_use]
pub fn beta_from_n_eff_adj(n_eff: f64, cg_mean: f64, n_cal: usize, k: f64) -> f64 {
    if n_cal < 2 {
        return 1e-6;
    }
    let n_cal_f = n_cal as f64;
    let n_eff_adj = (n_eff * cg_mean.powf(k)).clamp(1.0, n_cal_f);
    ((1.0 / n_eff_adj - 1.0 / n_cal_f) / (n_cal_f - 1.0)).max(1e-6)
}

/// Derive β₀ from token costs of merge phase executions.
///
/// `spans`: each tuple is `(merge_tokens_consumed, n_proposals)` from a merge phase.
/// `t1_tokens`: mean tokens per single adapter response (baseline cost unit).
///
/// Formula: β₀ = `mean(tokens_i` / `pairs_i`) / `t1_tokens`
/// where `pairs_i` = max(1, `n_i` × (`n_i` − 1) / 2).
///
/// Token cost is the correct USL β₀ analog per Gunther: it measures coherency overhead
/// as a fraction of the baseline processing cost, independent of network I/O latency.
///
/// Returns `None` when `spans` is empty or `t1_tokens` is 0. Clamps to [1e-9, 0.1].
#[must_use]
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
