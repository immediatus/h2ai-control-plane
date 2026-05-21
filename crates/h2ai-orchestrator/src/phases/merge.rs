use crate::attribution::{bootstrap_interval, AttributionInput, HarnessAttribution};
use crate::diagnostics::TalagrandDiagnostic;
use crate::engine::EngineInput;
use crate::mape_k::MergeOutput;
use crate::phases::{ExitReason, StepResult};
use crate::self_optimizer::{OptimizerParams, QualityMeasurement, SelfOptimizer, SuggestInput};
use h2ai_autonomic::merger::{MergeEngine, MergeOutcome};
use h2ai_autonomic::retry_accumulator::RetryAccumulator;
use h2ai_state::semilattice::ProposalSet;
use h2ai_types::config::VerificationConfig;
use h2ai_types::events::{
    AppliedOptimization, BranchPrunedEvent, ConstraintFrontierEvent, ConstraintViolation,
    OptimizationKind, VerificationScoredEvent,
};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::{OspConfig, PredictionBasis, TaskQuadrant};

/// Parameters needed by the merge phase that come from per-wave computation.
pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub task_id: &'a TaskId,
    pub retry_count: u32,
    pub explorer_count: u32,
    pub filter_ratio: f64,
    pub p_mean: f64,
    pub rho_mean: f64,
    pub tao_turns_mean: f64,
    pub attribution_basis: PredictionBasis,
    pub tau_values: Vec<f64>,
    pub all_raw_texts_this_wave: Vec<String>,
    /// Surviving proposal texts (`synthesis_candidates`) for epistemic yield.
    pub surviving_texts: Vec<String>,
    pub iteration_verification_events: &'a [VerificationScoredEvent],
    pub frontier_event: &'a Option<ConstraintFrontierEvent>,
    pub adapter_correctness: Vec<(ExplorerId, bool)>,
    pub oracle_gate_passed: Option<bool>,
    pub wave_coherence: &'a crate::coherence::CoherenceState,
    pub quality_history: &'a [QualityMeasurement],
    pub n_max_ceiling: u32,
    pub cg_mean: f64,
    pub current_params: &'a OptimizerParams,
    pub verification_config: VerificationConfig,
    pub assessed_quadrant: TaskQuadrant,
    pub all_pruned: &'a [BranchPrunedEvent],
    pub synthesis_candidates_len: usize,
    pub provisioned_merge_strategy: h2ai_types::sizing::MergeStrategy,
    /// Flat violations from all failed proposals this wave. Passed to OSP Zone 3 builder.
    pub wave_violations: Vec<ConstraintViolation>,
    /// Task-scoped retry accumulator. `None` when OSP is disabled.
    pub retry_accumulator: Option<&'a mut RetryAccumulator>,
    /// OSP configuration. `None` disables OSP (uses legacy strategy dispatch).
    pub osp_config: Option<&'a OspConfig>,
}

/// Run Phase 5: Merge.
///
/// Calls `MergeEngine::resolve` with the provided `proposal_set` and `pruned`.
/// On success (`MergeResolved`): constructs `crate::mape_k::MergeOutput` and
/// returns `(Done(merge_output), tau_expansion_hint)`.
/// On `ZeroSurvival`: classifies the failure mode and returns
/// `(EarlyExit(ExitReason::ZeroSurvival { ... }), tau_expansion_hint)` so engine.rs can
/// perform MAPE-K state mutations (topology forcing, tombstone, `mode_collapse_count`, etc.).
///
/// The second return value is the Talagrand τ expansion factor suggestion for this wave,
/// computed regardless of outcome so engine.rs can always update `tau_spread_factor`.
pub async fn run(
    proposal_set: ProposalSet,
    pruned: Vec<BranchPrunedEvent>,
    synthesis_gain: f64,
    synthesis_comparison_events: Vec<h2ai_types::events::VerifierComparisonEvent>,
    input: Input<'_>,
) -> (StepResult<MergeOutput>, Option<f64>) {
    let engine_input = input.engine_input;
    let task_id = input.task_id;
    let retry_count = input.retry_count;
    let filter_ratio = input.filter_ratio;
    let p_mean = input.p_mean;
    let rho_mean = input.rho_mean;

    // ── Talagrand τ feedback ─────────────────────────────────────────────────
    // Computed before the merge outcome so the tau_spread_factor can be updated
    // regardless of success or ZeroSurvival.
    let tau_expansion_next: Option<f64> = {
        let iter_scores: Vec<f64> = input
            .iteration_verification_events
            .iter()
            .map(|e| e.score)
            .collect();
        TalagrandDiagnostic::from_verification_scores(&[iter_scores]).map(|diag| {
            // Return the raw factor; caller applies to tau_spread_factor.
            diag.tau_expansion_factor(1.0, engine_input.cfg.tau_spread_max_factor)
        })
    };

    // ── Attribution ──────────────────────────────────────────────────────────
    let (mut attribution, attribution_interval) = {
        let iter_talagrand_state = {
            let scores: Vec<f64> = input
                .iteration_verification_events
                .iter()
                .map(|e| e.score)
                .collect();
            TalagrandDiagnostic::from_verification_scores(&[scores]).map(|d| d.calibration_state)
        };
        let attr_input = AttributionInput {
            p_mean,
            rho_mean,
            n_agents: input.explorer_count,
            verification_filter_ratio: filter_ratio,
            tao_turns_mean: input.tao_turns_mean,
            tao_per_turn_factor: engine_input.tao_multiplier,
            prediction_basis: input.attribution_basis,
            talagrand_state: iter_talagrand_state,
            eigen_calibration: engine_input.calibration.eigen.clone(),
        };
        let attr = HarnessAttribution::compute(&attr_input);
        let interval = {
            let cg_samples = &engine_input.calibration.coefficients.cg_samples;
            if cg_samples.len() >= 2 {
                Some(bootstrap_interval(&attr_input, cg_samples, 1000))
            } else {
                None
            }
        };
        (attr, interval)
    };

    // ── Adapter correctness ─────────────────────────────────────────────────
    let adapter_correctness = input.adapter_correctness;

    // ── Merge ────────────────────────────────────────────────────────────────
    let outcome = MergeEngine::resolve(
        task_id.clone(),
        proposal_set,
        pruned,
        input.provisioned_merge_strategy.clone(),
        retry_count,
        engine_input.embedding_model,
        if input.wave_violations.is_empty() {
            None
        } else {
            Some(&input.wave_violations)
        },
        input.retry_accumulator,
        input.osp_config,
    )
    .await;

    match outcome {
        MergeOutcome::Resolved {
            selection_resolved,
            resolved,
        } => {
            attribution.synthesis_gain = synthesis_gain;
            let mut quality_history = input.quality_history.to_vec();
            quality_history.push(QualityMeasurement {
                params: input.current_params.clone(),
                q_confidence: attribution.q_confidence,
            });
            let suggested_next = SelfOptimizer::suggest(SuggestInput {
                current: input.current_params,
                history: &quality_history,
                n_max_ceiling: input.n_max_ceiling,
                n_optimal: engine_input
                    .calibration
                    .ensemble
                    .as_ref()
                    .map(|ec| ec.n_optimal as u32),
                p_mean,
                rho_mean,
                filter_ratio,
                cfg: engine_input.cfg,
            });

            let waste_ratio = filter_ratio;
            let applied_optimizations: Vec<AppliedOptimization> = if waste_ratio
                < engine_input.cfg.optimizer_waste_threshold
            {
                let mut opts = Vec::new();
                if (suggested_next.verify_threshold - input.current_params.verify_threshold).abs()
                    > 1e-9
                {
                    opts.push(AppliedOptimization {
                        kind: OptimizationKind::TauSpreadAdjusted,
                        reason: format!(
                            "waste_ratio={:.2} < threshold {:.2}; \
                                 tighten verify_threshold to reduce pruned proposals",
                            waste_ratio, engine_input.cfg.optimizer_waste_threshold
                        ),
                        before: format!("{:.3}", input.current_params.verify_threshold),
                        after: format!("{:.3}", suggested_next.verify_threshold),
                    });
                }
                opts
            } else {
                vec![]
            };

            // Epistemic yield computation.
            let epistemic_yield: Option<f64> = {
                let surviving_texts = &input.surviving_texts;
                let n_requested = input.all_raw_texts_this_wave.len().max(1);
                if let Some(model) = engine_input.embedding_model {
                    let n_eff = h2ai_autonomic::epistemic::compute_n_eff_cosine(
                        surviving_texts,
                        model,
                        engine_input.cfg.eigen_n_eff_delta,
                    );
                    let yield_ratio = n_eff / n_requested as f64;
                    tracing::debug!(
                        n_eff_cosine_actual = n_eff,
                        yield_ratio,
                        "EpistemicYield computed (cosine)"
                    );
                    Some(yield_ratio.clamp(0.0, 1.0))
                } else if surviving_texts.len() >= 2 {
                    let refs: Vec<&str> = surviving_texts
                        .iter()
                        .map(std::string::String::as_str)
                        .collect();
                    let mean_jaccard = crate::correlated_hallucination::compute_cv(&refs)
                        .map_or(0.0, |s| s.mean_jaccard_distance);
                    let survival_rate = surviving_texts.len() as f64 / n_requested as f64;
                    let yield_approx = mean_jaccard * survival_rate;
                    tracing::debug!(
                        mean_jaccard,
                        survival_rate,
                        yield_approx,
                        "EpistemicYield computed (jaccard fallback)"
                    );
                    Some(yield_approx.clamp(0.0, 1.0))
                } else {
                    None
                }
            };

            // Talagrand full diagnostic for this wave.
            let talagrand = {
                let run_scores: Vec<f64> = input
                    .iteration_verification_events
                    .iter()
                    .map(|e| e.score)
                    .collect();
                TalagrandDiagnostic::from_verification_scores(&[run_scores])
            };

            let comparison_events = synthesis_comparison_events;

            let merge_out = MergeOutput {
                task_id: task_id.clone(),
                resolved_output: resolved.resolved_output,
                selection_resolved: true,
                selection_resolved_event: selection_resolved,
                attribution,
                attribution_interval,
                talagrand,
                suggested_next_params: Some(suggested_next),
                waste_ratio,
                applied_optimizations,
                epistemic_yield,
                frontier_event: input.frontier_event.clone(),
                adapter_correctness,
                coherence_state: input.wave_coherence.clone(),
                comparison_events,
                oracle_gate_passed: input.oracle_gate_passed,
                tau_values: input.tau_values,
                iteration_verification_events: input.iteration_verification_events.to_vec(),
            };

            (StepResult::Done(merge_out), tau_expansion_next)
        }

        MergeOutcome::ZeroSurvival(mut zero_event) => {
            // GAP-A4 #4: coherence-closed early exit — handled in engine.rs after we return.

            // Compute epistemic diagnostics for failure mode classification.
            let detected_failure_mode = if let Some(model) = engine_input.embedding_model {
                let n_eff = h2ai_autonomic::epistemic::compute_n_eff_cosine(
                    &input.all_raw_texts_this_wave,
                    model,
                    engine_input.cfg.eigen_n_eff_delta,
                );
                let failure = h2ai_autonomic::epistemic::classify_failure_mode(
                    n_eff,
                    input.all_raw_texts_this_wave.len().max(1),
                    engine_input.cfg.safety.diversity_threshold,
                );
                zero_event.n_eff_cosine_actual = Some(n_eff);
                zero_event.failure_mode = Some(failure.clone());
                Some(failure)
            } else {
                None
            };

            let n_eff_cosine = zero_event.n_eff_cosine_actual;

            (
                StepResult::EarlyExit(ExitReason::ZeroSurvival {
                    failure_mode: detected_failure_mode,
                    coherence: input.wave_coherence.clone(),
                    n_eff_cosine,
                    filter_ratio,
                    tau_values: input.tau_values,
                }),
                tau_expansion_next,
            )
        }
    }
}
