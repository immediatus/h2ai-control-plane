use crate::complexity::assess_task_complexity;
use crate::engine::{EngineError, EngineInput};
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::events::TaskComplexityAssessedEvent;
use h2ai_types::sizing::{MultiplicationConditionFailure, TaskQuadrant};

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
    let complexity_event = complexity_assessment.clone();

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
                    .map(|e| e.n_effective)
                    .unwrap_or(0.0),
                threshold: input.cfg.task_complexity.n_eff_complex_threshold,
            }
            .to_string(),
        ));
    }

    let cg_mean = input.calibration.coefficients.cg_mean();
    let n_max_ceiling = input.calibration.coefficients.n_max().floor() as u32;

    Ok(Output {
        assessed_quadrant,
        complexity_event,
        cg_mean,
        n_max_ceiling,
    })
}
