use chrono::Utc;
use futures::future::join_all;
use h2ai_config::prompts::{PROBE_SYSTEM_PREFIX, PROBE_TASK};
use h2ai_config::TaskComplexityConfig;
use h2ai_constraints::complexity::compute_corpus_complexity_with_coefficients;
use h2ai_constraints::eval::eval_sync;
use h2ai_constraints::types::{ConstraintDoc, ConstraintTier};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::events::{
    CalibrationCompletedEvent, CalibrationQuality, TaskComplexityAssessedEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{ProbeSkipReason, TaskQuadrant};

/// Assess the complexity of the task corpus and classify it into a routing quadrant.
///
/// Phase 1.5 of the MAPE-K pipeline. Called once per task, before topology provisioning.
///
/// # Routing paths
/// - **Path A (Precision)**: `TCC_structural` ≤ `tcc_precision_threshold` → skip probe, Precision
/// - **Path B (Coverage)**: `TCC_structural` ≥ `tcc_coverage_threshold` → skip probe, Coverage
/// - **Path C (Ambiguous)**: between thresholds → if `probe` is `Some`, dispatches N-probe
///   completions via `run_probe` to compute `TCC_empirical`; falls back to Coverage when `None`
/// - **Heavy-dominant bypass**: `static_coverage` < `min_static_coverage_for_probe` →
///   apply heavy amplification to `TCC_structural`, skip probe
/// - **Bootstrap guard**: `CalibrationQuality::Bootstrap` → skip probe, route Coverage
///
/// In `shadow_mode` (default: true) the quadrant is computed and emitted but does not
/// influence topology selection — all tasks route as Coverage downstream.
pub async fn assess_task_complexity(
    corpus: &[ConstraintDoc],
    calibration: &CalibrationCompletedEvent,
    cfg: &TaskComplexityConfig,
    task_id: TaskId,
    probe: Option<(&dyn IComputeAdapter, &str)>,
) -> TaskComplexityAssessedEvent {
    // Bootstrap guard: synthetic priors cannot be used to calibrate the probe.
    if calibration.calibration_quality == CalibrationQuality::Bootstrap {
        return TaskComplexityAssessedEvent {
            task_id,
            tcc_structural: 1.0,
            tcc_empirical: None,
            tcc_effective: cfg.tcc_coverage_threshold,
            n_eff_pool: calibration.eigen.as_ref().map(|e| e.n_effective),
            task_quadrant: TaskQuadrant::Coverage,
            probe_skipped: true,
            probe_skip_reason: ProbeSkipReason::BootstrapCalibration,
            heavy_fraction: 0.0,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: 0,
            timestamp: Utc::now(),
        };
    }

    let meta =
        compute_corpus_complexity_with_coefficients(corpus, cfg.k_soft, cfg.k_type, cfg.k_cross);

    let n_eff_pool = calibration.eigen.as_ref().map(|e| e.n_effective);

    // Heavy-dominant bypass: satisfaction matrix would be near-empty.
    if meta.static_coverage < cfg.min_static_coverage_for_probe {
        let tcc_eff = meta.tcc_structural * cfg.k_heavy.mul_add(meta.heavy_fraction, 1.0);
        let quadrant = classify_quadrant(tcc_eff, n_eff_pool, cfg);
        return TaskComplexityAssessedEvent {
            task_id,
            tcc_structural: meta.tcc_structural,
            tcc_empirical: None,
            tcc_effective: tcc_eff,
            n_eff_pool,
            task_quadrant: quadrant,
            probe_skipped: true,
            probe_skip_reason: ProbeSkipReason::HeavyDominantCorpus,
            heavy_fraction: meta.heavy_fraction,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: 0,
            timestamp: Utc::now(),
        };
    }

    // Path A: unambiguously Precision — probe cannot change routing.
    if meta.tcc_structural <= cfg.tcc_precision_threshold {
        let quadrant = classify_quadrant(meta.tcc_structural, n_eff_pool, cfg);
        return TaskComplexityAssessedEvent {
            task_id,
            tcc_structural: meta.tcc_structural,
            tcc_empirical: None,
            tcc_effective: meta.tcc_structural,
            n_eff_pool,
            task_quadrant: quadrant,
            probe_skipped: true,
            probe_skip_reason: ProbeSkipReason::UnambiguousPrecision,
            heavy_fraction: meta.heavy_fraction,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: meta.n_constraints,
            timestamp: Utc::now(),
        };
    }

    // Path B: unambiguously Coverage — probe cannot change routing.
    if meta.tcc_structural >= cfg.tcc_coverage_threshold {
        let quadrant = classify_quadrant(meta.tcc_structural, n_eff_pool, cfg);
        return TaskComplexityAssessedEvent {
            task_id,
            tcc_structural: meta.tcc_structural,
            tcc_empirical: None,
            tcc_effective: meta.tcc_structural,
            n_eff_pool,
            task_quadrant: quadrant,
            probe_skipped: true,
            probe_skip_reason: ProbeSkipReason::UnambiguousCoverage,
            heavy_fraction: meta.heavy_fraction,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: meta.n_constraints,
            timestamp: Utc::now(),
        };
    }

    // Ambiguous band: TCC_structural is between precision and coverage thresholds.
    // When a probe adapter is available, run N-probe empirical estimation via
    // run_probe to compute TCC_empirical. Without an adapter, fall back to Coverage.
    if let Some((adapter, system_context)) = probe {
        let static_corpus: Vec<&ConstraintDoc> = corpus
            .iter()
            .filter(|d| d.tier() == ConstraintTier::Static)
            .collect();
        return run_probe(ProbeInput {
            meta_tcc_structural: meta.tcc_structural,
            meta_heavy_fraction: meta.heavy_fraction,
            static_corpus: &static_corpus,
            n_eff_pool,
            cfg,
            task_id,
            adapter,
            system_context,
        })
        .await;
    }

    // No adapter supplied — defer probe, route conservatively to Coverage.
    let tcc_eff = meta.tcc_structural;
    let quadrant = classify_quadrant(tcc_eff, n_eff_pool, cfg);
    TaskComplexityAssessedEvent {
        task_id,
        tcc_structural: meta.tcc_structural,
        tcc_empirical: None,
        tcc_effective: tcc_eff,
        n_eff_pool,
        task_quadrant: quadrant,
        probe_skipped: true,
        probe_skip_reason: ProbeSkipReason::AmbiguousBandProbeDeferred,
        heavy_fraction: meta.heavy_fraction,
        tcc_mismatch: false,
        probe_cost_tokens: 0,
        n_informative_static: meta.n_constraints,
        timestamp: Utc::now(),
    }
}

/// Classify routing quadrant from `TCC_effective` and pool `N_eff`.
///
/// High TCC → Coverage or Complex depending on pool diversity.
/// Low TCC → Precision or Degenerate depending on pool diversity.
#[must_use]
pub fn classify_quadrant(
    tcc_effective: f64,
    n_eff_pool: Option<f64>,
    cfg: &TaskComplexityConfig,
) -> TaskQuadrant {
    let pool_ok = n_eff_pool.is_none_or(|n| n >= cfg.n_eff_complex_threshold); // no eigen calibration → assume adequate diversity

    if tcc_effective >= cfg.tcc_coverage_threshold {
        if pool_ok {
            TaskQuadrant::Coverage
        } else {
            TaskQuadrant::Complex
        }
    } else if tcc_effective <= cfg.tcc_precision_threshold {
        if pool_ok {
            TaskQuadrant::Precision
        } else {
            TaskQuadrant::Degenerate
        }
    } else {
        // Ambiguous band defaults to Coverage (conservative).
        if pool_ok {
            TaskQuadrant::Coverage
        } else {
            TaskQuadrant::Complex
        }
    }
}

// ── N-probe path: empirical TCC estimation ────────────────────────────────────

/// Participation ratio PR = (`Σλ_i)²` / `Σλ_i²` for the covariance matrix of the
/// constraint satisfaction vectors across probe outputs.
///
/// Uses the trace identity: PR = tr(C)² / tr(C²) = tr(C)² / ‖C‖_F²
/// (valid for any symmetric PSD matrix; no eigendecomposition required).
///
/// `matrix` is row-major: matrix[`probe_idx`][constraint_idx] ∈ {0, 1}.
/// Returns 1.0 for empty or all-zero inputs (degenerate fallback).
#[must_use]
pub fn participation_ratio(matrix: &[Vec<f64>]) -> f64 {
    if matrix.is_empty() {
        return 1.0;
    }
    let n_probes = matrix.len();
    let n_cols = matrix[0].len();
    if n_cols == 0 {
        return 1.0;
    }

    // Column means
    let means: Vec<f64> = (0..n_cols)
        .map(|j| matrix.iter().map(|row| row[j]).sum::<f64>() / n_probes as f64)
        .collect();

    // Centered matrix M_c[i][j] = x[i][j] - mean[j]
    let centered: Vec<Vec<f64>> = matrix
        .iter()
        .map(|row| row.iter().enumerate().map(|(j, &v)| v - means[j]).collect())
        .collect();

    // Covariance C = M_c^T × M_c / (n_probes - 1)
    // tr(C) = Σ_j C_jj = Σ_j (Σ_i M_c[i][j]²) / (n_probes - 1)
    // ‖C‖_F² = Σ_jk C_jk² = Σ_jk (Σ_i M_c[i][j]·M_c[i][k])² / (n_probes-1)²
    let denom = (n_probes.saturating_sub(1)) as f64;
    if denom < f64::EPSILON {
        return 1.0;
    }

    // trace(C) = (1/denom) × Σ_j Σ_i M_c[i][j]²
    let trace_c: f64 = (0..n_cols)
        .map(|j| centered.iter().map(|row| row[j] * row[j]).sum::<f64>())
        .sum::<f64>()
        / denom;

    if trace_c < f64::EPSILON {
        return 1.0; // all probes agree on every constraint → no useful diversity
    }

    // ‖C‖_F² = tr(C²) = (1/denom²) × Σ_jk (Σ_i M_c[i][j]·M_c[i][k])²
    let mut frob_sq: f64 = 0.0;
    for j in 0..n_cols {
        for k in 0..n_cols {
            let dot: f64 = centered.iter().map(|row| row[j] * row[k]).sum();
            frob_sq += (dot / denom) * (dot / denom);
        }
    }

    if frob_sq < f64::EPSILON {
        return 1.0;
    }

    (trace_c * trace_c / frob_sq).clamp(1.0, n_cols as f64)
}

/// Input bundle for [`run_probe`].
pub struct ProbeInput<'a> {
    pub meta_tcc_structural: f64,
    pub meta_heavy_fraction: f64,
    pub static_corpus: &'a [&'a ConstraintDoc],
    pub n_eff_pool: Option<f64>,
    pub cfg: &'a TaskComplexityConfig,
    pub task_id: TaskId,
    pub adapter: &'a dyn IComputeAdapter,
    pub system_context: &'a str,
}

/// Assess task complexity using N-probe empirical TCC estimation (ambiguous band).
///
/// Dispatches `cfg.n_probe` lightweight completions, evaluates Static-tier constraints
/// via `eval_sync`, computes the constraint satisfaction covariance matrix, and derives
/// `TCC_empirical` as its participation ratio.  Falls back to `TCC_structural` when the
/// probe outputs are degenerate (all constraints unanimously pass or fail).
///
/// # Routing decision
/// - `TCC_effective` = `max(TCC_structural`, `TCC_empirical`) + `mismatch_penalty` × [mismatch]
/// - Then re-classifies the quadrant using the same `classify_quadrant` function.
pub async fn run_probe(input: ProbeInput<'_>) -> TaskComplexityAssessedEvent {
    let ProbeInput {
        meta_tcc_structural,
        meta_heavy_fraction,
        static_corpus,
        n_eff_pool,
        cfg,
        task_id,
        adapter,
        system_context,
    } = input;
    let probe_outputs: Vec<String> = join_all((0..cfg.n_probe).map(|_| {
        adapter.execute(ComputeRequest {
            system_context: format!("{PROBE_SYSTEM_PREFIX}\n{system_context}"),
            task: PROBE_TASK.as_str().into(),
            tau: h2ai_types::sizing::TauValue::new(cfg.probe_tau.clamp(0.05, 0.95))
                .unwrap_or_else(|_| h2ai_types::sizing::TauValue::new(0.5).unwrap()),
            max_tokens: cfg.probe_max_tokens,
        })
    }))
    .await
    .into_iter()
    .filter_map(std::result::Result::ok)
    .map(|r| r.output)
    .collect();

    let probe_cost_tokens = probe_outputs.len() as u64 * cfg.probe_max_tokens;

    // Fall back to structural TCC if all probes failed
    if probe_outputs.is_empty() {
        let quadrant = classify_quadrant(meta_tcc_structural, n_eff_pool, cfg);
        return TaskComplexityAssessedEvent {
            task_id,
            tcc_structural: meta_tcc_structural,
            tcc_empirical: None,
            tcc_effective: meta_tcc_structural,
            n_eff_pool,
            task_quadrant: quadrant,
            probe_skipped: false,
            probe_skip_reason: ProbeSkipReason::None,
            heavy_fraction: meta_heavy_fraction,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: 0,
            timestamp: Utc::now(),
        };
    }

    // Evaluate each probe output against every Static-tier constraint
    let satisfaction: Vec<Vec<f64>> = probe_outputs
        .iter()
        .map(|text| {
            static_corpus
                .iter()
                .map(|c| {
                    if eval_sync(&c.predicate, text) >= 0.5 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    // Filter to informative columns (at least 1 pass AND 1 fail across probes)
    let n_static = static_corpus.len();
    let informative_cols: Vec<usize> = (0..n_static)
        .filter(|&j| {
            let passes = satisfaction.iter().filter(|row| row[j] > 0.5).count();
            passes > 0 && passes < satisfaction.len()
        })
        .collect();

    let n_informative_static = informative_cols.len();

    if n_informative_static < cfg.tcc_min_informative_constraints {
        // All probes agree on everything — no constraint discrimination signal.
        // Amplify structural TCC (same as heavy-dominant formula) and classify.
        let tcc_eff = meta_tcc_structural * cfg.k_heavy.mul_add(meta_heavy_fraction, 1.0);
        let quadrant = classify_quadrant(tcc_eff, n_eff_pool, cfg);
        return TaskComplexityAssessedEvent {
            task_id,
            tcc_structural: meta_tcc_structural,
            tcc_empirical: None,
            tcc_effective: tcc_eff,
            n_eff_pool,
            task_quadrant: quadrant,
            probe_skipped: false,
            probe_skip_reason: ProbeSkipReason::None,
            heavy_fraction: meta_heavy_fraction,
            tcc_mismatch: false,
            probe_cost_tokens,
            n_informative_static: 0,
            timestamp: Utc::now(),
        };
    }

    // Build the informative sub-matrix for covariance computation
    let informative_matrix: Vec<Vec<f64>> = satisfaction
        .iter()
        .map(|row| informative_cols.iter().map(|&j| row[j]).collect())
        .collect();

    let tcc_empirical = participation_ratio(&informative_matrix);

    // Composite TCC_effective: take max, add mismatch penalty when structural >> empirical
    let tcc_mismatch = meta_tcc_structural > tcc_empirical + 1.0;
    let penalty = if tcc_mismatch {
        cfg.tcc_mismatch_penalty
    } else {
        0.0
    };
    let tcc_effective = meta_tcc_structural.max(tcc_empirical) + penalty;

    let quadrant = classify_quadrant(tcc_effective, n_eff_pool, cfg);
    TaskComplexityAssessedEvent {
        task_id,
        tcc_structural: meta_tcc_structural,
        tcc_empirical: Some(tcc_empirical),
        tcc_effective,
        n_eff_pool,
        task_quadrant: quadrant,
        probe_skipped: false,
        probe_skip_reason: ProbeSkipReason::None,
        heavy_fraction: meta_heavy_fraction,
        tcc_mismatch,
        probe_cost_tokens,
        n_informative_static,
        timestamp: Utc::now(),
    }
}
