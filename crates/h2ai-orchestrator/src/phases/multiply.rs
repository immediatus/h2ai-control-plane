use crate::engine::{EngineError, EngineInput};
use crate::phases::{ExitReason, StepResult};
use h2ai_autonomic::checker::MultiplicationChecker;
use h2ai_types::events::TopologyProvisionedEvent;

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub provisioned: &'a TopologyProvisionedEvent,
    pub baseline_competence: f64,
    pub error_correlation: f64,
    pub retry_count: u32,
}

pub struct Output {
    pub tau_values: Vec<f64>,
}

/// Run Phase 2.5: Multiplication Condition Gate.
///
/// Calls `MultiplicationChecker::check` with the pre-derived `baseline_competence`
/// and `error_correlation` values (caller derives these from `EnsembleCalibration` or
/// `cg_mean` heuristic — they are needed elsewhere in engine.rs so we take them as input
/// rather than re-deriving here).
///
/// Returns `StepResult::Done(Output { tau_values })` on success.
/// Returns `StepResult::EarlyExit(ExitReason::MultiplicationFailed { ... })` when the
/// multiplication condition fails; the caller is responsible for updating
/// `last_multiplication_failure` / `tau_values_tried` and invoking `RetryPolicy::decide`.
///
/// Never returns `StepResult::Fatal`.
#[must_use]
pub fn run(input: Input<'_>) -> StepResult<Output> {
    let engine_input = input.engine_input;
    let provisioned = input.provisioned;
    let retry_count = input.retry_count;

    let tau_values: Vec<f64> = provisioned
        .explorer_configs
        .iter()
        .map(|ec| ec.tau.value())
        .collect();

    if let Err(mc_event) = MultiplicationChecker::check(
        &engine_input.task_id,
        &engine_input.calibration.coefficients,
        &engine_input.calibration.coordination_threshold,
        input.baseline_competence,
        input.error_correlation,
        retry_count,
        engine_input.cfg,
    ) {
        let msg = format!("{:?}", mc_event.failure);
        return StepResult::EarlyExit(ExitReason::MultiplicationFailed {
            msg,
            tau_values,
            failure: mc_event.failure,
        });
    }

    StepResult::Done(Output { tau_values })
}

// Suppress unused-import lint — EngineError is part of StepResult::Fatal which this
// phase never emits but must remain importable.
const _: fn() = || {
    let _: Option<EngineError> = None;
};
