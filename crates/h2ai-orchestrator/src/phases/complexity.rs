use crate::complexity::assess_task_complexity;
use crate::engine::{EngineError, EngineInput};
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::events::TaskComplexityAssessedEvent;
use h2ai_types::sizing::{MultiplicationConditionFailure, TaskQuadrant};

/// BFT/Krum/SRANI quorum minimum — N must be at least 3 for voting algorithms to work.
const QUORUM_FLOOR: u32 = 3;

pub struct Output {
    pub assessed_quadrant: TaskQuadrant,
    pub complexity_event: TaskComplexityAssessedEvent,
    pub cg_mean: f64,
    pub n_max_ceiling: u32,
}

pub async fn run(input: &EngineInput<'_>, system_context: &str) -> Result<Output, EngineError> {
    let task_id = input.task_id.clone();

    let probe_adapter = input.explorer_adapters.first().copied();
    let complexity_assessment = assess_task_complexity(
        &input.constraint_corpus,
        &input.calibration,
        &input.cfg.task_complexity,
        task_id.clone(),
        probe_adapter.map(|a| (a as &dyn IComputeAdapter, system_context)),
    )
    .await;
    let assessed_quadrant = complexity_assessment.task_quadrant;
    let complexity_event = complexity_assessment;

    // Degenerate guard (non-shadow mode only): both TCC and pool N_eff are below
    // their thresholds. The pool cannot explore the solution space for this task;
    // fail immediately rather than wasting MAPE-K retries.
    if !input.cfg.task_complexity.shadow_mode && assessed_quadrant == TaskQuadrant::Degenerate {
        input.store.mark_failed(&task_id);
        return Err(EngineError::MultiplicationConditionFailed(
            MultiplicationConditionFailure::InsufficientPoolDiversity {
                n_eff: input
                    .calibration
                    .eigen
                    .as_ref()
                    .map_or(0.0, |e| e.n_effective),
                threshold: input.cfg.task_complexity.n_eff_complex_threshold,
            }
            .to_string(),
        ));
    }

    let cg_mean = input.calibration.coefficients.cg_mean();
    let cc = &input.calibration.coefficients;

    // Quorum degradation guard (non-shadow mode only): if N_max < 3, the adapter is too
    // degraded to support BFT/Krum/SRANI. Fail immediately rather than silently using a
    // sub-quorum pool, which would disable the very safety mechanisms that make the failure
    // detectable.
    if !input.cfg.task_complexity.shadow_mode && cc.n_max_degraded() {
        let unclamped = cc.n_max();
        tracing::warn!(
            target: "h2ai.engine",
            unclamped_n_max = unclamped,
            "adapter N_max below quorum floor — adapter should be marked Offline"
        );
        input.store.mark_failed(&task_id);
        return Err(EngineError::MultiplicationConditionFailed(
            MultiplicationConditionFailure::QuorumDegradedBelowMinimum {
                unclamped_n_max: unclamped,
            }
            .to_string(),
        ));
    }

    let n_max_ceiling = cc.n_max().floor() as u32;
    // Hard floor: preserve quorum even after shadow-mode tasks or CI override paths.
    let n_max_ceiling = n_max_ceiling.max(QUORUM_FLOOR);

    Ok(Output {
        assessed_quadrant,
        complexity_event,
        cg_mean,
        n_max_ceiling,
    })
}
