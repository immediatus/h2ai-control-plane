use crate::engine::{EngineError, EngineInput};
use crate::phases::{ExitReason, StepResult};
use h2ai_types::events::TopologyProvisionedEvent;

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub provisioned: &'a TopologyProvisionedEvent,
}

/// Run Phase 2.6: Pool Diversity Guard.
///
/// Checks whether the explorer pool's prior `n_eff_cosine` meets the configured
/// diversity threshold. A low `n_eff_prior` (< 1.0 + `diversity_threshold`) signals
/// that the adapter pool is too homogeneous for meaningful ensemble diversity.
///
/// When `diversity_threshold` is zero (disabled), or when `n_eff_prior` is zero
/// (not measured), the gate is skipped and `Done(())` is returned.
///
/// Returns `StepResult::Done(())` when the gate passes (or is disabled).
/// Returns `StepResult::EarlyExit(ExitReason::DiversityFailed { n_eff, tau_values })`
/// when the pool diversity is insufficient; the caller is responsible for updating
/// `last_multiplication_failure` / `tau_values_tried` and invoking `RetryPolicy::decide`.
///
/// Never returns `StepResult::Fatal`.
#[must_use]
pub fn run(input: Input<'_>) -> StepResult<()> {
    let engine_input = input.engine_input;
    let provisioned = input.provisioned;

    if engine_input.cfg.safety.diversity_threshold <= 0.0 {
        return StepResult::Done(());
    }

    let n_eff_prior = engine_input.calibration.n_eff_cosine_prior;
    let threshold = 1.0 + engine_input.cfg.safety.diversity_threshold;

    if n_eff_prior > 0.0 && n_eff_prior < threshold {
        let tau_values: Vec<f64> = provisioned
            .explorer_configs
            .iter()
            .map(|ec| ec.tau.value())
            .collect();
        return StepResult::EarlyExit(ExitReason::DiversityFailed {
            n_eff: n_eff_prior,
            tau_values,
        });
    }

    StepResult::Done(())
}

// Suppress unused-import lint — EngineError is part of StepResult::Fatal which this
// phase never emits but must remain importable.
const _: fn() = || {
    let _: Option<EngineError> = None;
};
