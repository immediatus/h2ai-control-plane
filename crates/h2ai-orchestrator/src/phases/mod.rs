use crate::coherence::CoherenceState;
use crate::engine::EngineError;
use h2ai_types::events::FailureMode;
use h2ai_types::sizing::MultiplicationConditionFailure;

pub enum StepResult<T> {
    Done(T),
    EarlyExit(ExitReason),
    Fatal(EngineError),
}

#[derive(Debug)]
pub enum ExitReason {
    MultiplicationFailed {
        msg: String,
        tau_values: Vec<f64>,
        failure: MultiplicationConditionFailure,
    },
    DiversityFailed {
        n_eff: f64,
        tau_values: Vec<f64>,
    },
    ZeroSurvival {
        failure_mode: Option<FailureMode>,
        coherence: CoherenceState,
        n_eff_cosine: Option<f64>,
        filter_ratio: f64,
        /// τ values from the current generation wave; pushed to `tau_values_tried` before retry.
        tau_values: Vec<f64>,
    },
    OracleBlocked,
    /// GAP-C1 correlated hallucination detected — clustered ensemble; retry with grounding.
    HallucinationDetected {
        /// Formatted retry_context hint to set before `continue`.
        retry_context_hint: String,
        tau_values: Vec<f64>,
        warning: h2ai_types::events::CorrelatedEnsembleWarning,
        researcher_grounding_events: Vec<h2ai_types::events::ResearcherGroundingEvent>,
    },
}

pub mod audit;
pub mod bootstrap;
pub mod complexity;
pub mod diversity;
pub mod domain_coverage;
pub mod frontier;
pub mod generation;
pub mod hallucination;
pub mod merge;
pub mod multiply;
pub mod oracle;
pub mod srani;
pub mod synthesis;
pub mod topology;
pub mod verify;
