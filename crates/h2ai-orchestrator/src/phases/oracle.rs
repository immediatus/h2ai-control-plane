use crate::engine::{EngineError, EngineInput};
use crate::phases::{ExitReason, StepResult};
use h2ai_types::events::OracleGateResultEvent;

#[derive(Debug, PartialEq, Eq)]
pub enum PostSelectionDecision {
    Accept,
    Evict,
    Clarify,
}

pub fn apply_on_fail_policy(gate_passed: Option<bool>, on_fail: &str) -> PostSelectionDecision {
    match gate_passed {
        Some(false) => match on_fail {
            "clarify" => PostSelectionDecision::Clarify,
            "pass" => PostSelectionDecision::Accept,
            _ => PostSelectionDecision::Evict,
        },
        _ => PostSelectionDecision::Accept,
    }
}

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
}

/// Run the oracle NATS gate check (Phase 3→4 transition).
///
/// Returns `StepResult::Done(flag)` where `flag` is:
/// - `None`        — gate disabled or no NATS client; caller continues normally
/// - `Some(true)`  — gate passed
/// - `Some(false)` — gate explicitly failed; caller should abort
///
/// Returns `StepResult::EarlyExit(ExitReason::OracleBlocked)` when a hard block
/// is signalled (currently mapped from the `Some(false)` branch by the caller
/// in engine.rs — the engine returns `Err(MaxRetriesExhausted)` in that case).
///
/// This function never returns `StepResult::Fatal`.
pub async fn run(input: Input<'_>) -> StepResult<Option<bool>> {
    let engine_input = input.engine_input;

    if !engine_input.cfg.oracle_gate.enabled {
        return StepResult::Done(None);
    }

    let Some(nats) = &engine_input.nats_raw else {
        return StepResult::Done(None);
    };

    let gate_payload = serde_json::json!({
        "task_id": &engine_input.task_id,
        "phase": 3,
    });
    let payload_bytes = serde_json::to_vec(&gate_payload).unwrap_or_default();
    let timeout = std::time::Duration::from_secs(engine_input.cfg.oracle_gate.timeout_secs);

    let gate_result = match tokio::time::timeout(
        timeout,
        nats.request(
            engine_input.cfg.oracle_gate.subject.clone(),
            payload_bytes.into(),
        ),
    )
    .await
    {
        Ok(Ok(response)) => {
            match serde_json::from_slice::<OracleGateResultEvent>(&response.payload) {
                Ok(result) => Some(result.gate_passed),
                Err(_) => Some(engine_input.cfg.oracle_gate.on_timeout == "pass"),
            }
        }
        _ => Some(engine_input.cfg.oracle_gate.on_timeout == "pass"),
    };

    if gate_result == Some(false) {
        return StepResult::EarlyExit(ExitReason::OracleBlocked);
    }

    StepResult::Done(gate_result)
}

pub struct PostSelectionInput<'a> {
    pub task_id: String,
    pub winner_text: &'a str,
    pub oracle_config: &'a h2ai_config::OracleGateConfig,
    pub nats: Option<&'a async_nats::Client>,
}

pub async fn run_post_selection(input: PostSelectionInput<'_>) -> PostSelectionDecision {
    if !input.oracle_config.enabled {
        return PostSelectionDecision::Accept;
    }
    let Some(nats) = input.nats else {
        return PostSelectionDecision::Accept;
    };

    let payload = serde_json::json!({
        "task_id": &input.task_id,
        "phase": "post_selection",
        "winner_text": input.winner_text,
    });
    let payload_bytes = serde_json::to_vec(&payload).unwrap_or_default();
    let timeout = std::time::Duration::from_secs(input.oracle_config.timeout_secs);

    let gate_result: Option<bool> = match tokio::time::timeout(
        timeout,
        nats.request(input.oracle_config.subject.clone(), payload_bytes.into()),
    )
    .await
    {
        Ok(Ok(response)) => {
            match serde_json::from_slice::<OracleGateResultEvent>(&response.payload) {
                Ok(result) => Some(result.gate_passed),
                Err(_) => Some(input.oracle_config.on_timeout == "pass"),
            }
        }
        _ => Some(input.oracle_config.on_timeout == "pass"),
    };

    apply_on_fail_policy(gate_result, &input.oracle_config.on_fail)
}

// Suppress unused-import lint — EngineError is part of the StepResult::Fatal variant
// and must remain importable from this module even though this phase never emits it.
const _: fn() = || {
    let _: Option<EngineError> = None;
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evict_policy_on_failed_gate() {
        assert_eq!(apply_on_fail_policy(Some(false), "evict"), PostSelectionDecision::Evict);
    }

    #[test]
    fn pass_policy_ignores_failure() {
        assert_eq!(apply_on_fail_policy(Some(false), "pass"), PostSelectionDecision::Accept);
    }

    #[test]
    fn accept_when_gate_passed() {
        assert_eq!(apply_on_fail_policy(Some(true), "evict"), PostSelectionDecision::Accept);
    }

    #[test]
    fn accept_when_gate_not_run() {
        assert_eq!(apply_on_fail_policy(None, "evict"), PostSelectionDecision::Accept);
    }

    #[test]
    fn clarify_policy_on_failure() {
        assert_eq!(apply_on_fail_policy(Some(false), "clarify"), PostSelectionDecision::Clarify);
    }
}
