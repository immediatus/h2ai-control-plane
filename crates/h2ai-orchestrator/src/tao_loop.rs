use crate::engine::EngineError;
use crate::output_schema::{validate_output, SchemaValidationResult};
use chrono::Utc;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::{OutputSchemaConfig, TaoConfig};
use h2ai_types::events::{ProposalEvent, TaoIterationEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use regex::Regex;
use std::time::Duration;
use tokio::time::timeout;

/// Accumulated per-turn memory for a single TAO loop execution.
struct TaoMemoryEntry {
    turn: u8,
    observation: String,
    passed: bool,
}

fn format_memory(entries: &[TaoMemoryEntry]) -> String {
    if entries.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = entries
        .iter()
        .map(|e| {
            let status = if e.passed { "PASS" } else { "FAIL" };
            format!("[TAO turn {} — {}]: {}", e.turn, status, e.observation)
        })
        .collect();
    format!("\n\n[TAO MEMORY]\n{}", lines.join("\n"))
}

pub struct TaoInput<'a> {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub adapter: &'a dyn IComputeAdapter,
    pub initial_request: ComputeRequest,
    pub config: TaoConfig,
    pub schema_config: Option<OutputSchemaConfig>,
}

pub struct TaoProposal {
    pub event: ProposalEvent,
    pub tao_turns: u8,
    pub iterations: Vec<TaoIterationEvent>,
}

pub struct TaoLoop;

impl TaoLoop {
    pub async fn run(input: TaoInput<'_>) -> Result<TaoProposal, EngineError> {
        // Fix I-1: guard against max_turns == 0 before entering the loop
        if input.config.max_turns == 0 {
            return Err(EngineError::Adapter("TAO max_turns must be >= 1".into()));
        }

        // Fix I-2: propagate invalid regex as a hard error
        let pattern = input
            .config
            .verify_pattern
            .as_deref()
            .map(Regex::new)
            .transpose()
            .map_err(|e| EngineError::Parse(format!("invalid TAO verify_pattern: {e}")))?;

        let mut req = input.initial_request.clone();
        // Fix I-3: collect per-turn iteration events
        let mut iterations: Vec<TaoIterationEvent> = Vec::new();
        let mut memory: Vec<TaoMemoryEntry> = Vec::new();

        for turn in 1..=input.config.max_turns {
            let resp = timeout(Duration::from_secs(30), input.adapter.execute(req.clone()))
                .await
                .map_err(|_| EngineError::Adapter("TAO timeout".into()))?
                .map_err(|e| EngineError::Adapter(e.to_string()))?;

            let pattern_passed = pattern
                .as_ref()
                .map(|re| re.is_match(&resp.output))
                .unwrap_or(true);

            let schema_result = validate_output(&resp.output, input.schema_config.as_ref());
            let schema_passed = !matches!(schema_result, SchemaValidationResult::Invalid(_));

            let passed = pattern_passed && schema_passed;

            let observation = if passed {
                input.config.observation_pass.clone()
            } else if !pattern_passed {
                input
                    .config
                    .observation_fail_pattern
                    .replace("{turn}", &turn.to_string())
            } else {
                input
                    .config
                    .observation_fail_schema
                    .replace("{turn}", &turn.to_string())
                    .replace(
                        "{error}",
                        schema_result.as_invalid_msg().unwrap_or("unknown schema error"),
                    )
            };

            let iter_event = TaoIterationEvent {
                task_id: input.task_id.clone(),
                explorer_id: input.explorer_id.clone(),
                turn,
                observation: observation.clone(),
                passed,
                timestamp: Utc::now(),
            };
            iterations.push(iter_event);

            if passed || turn == input.config.max_turns {
                return Ok(TaoProposal {
                    event: ProposalEvent {
                        task_id: input.task_id.clone(),
                        explorer_id: input.explorer_id.clone(),
                        tau: req.tau,
                        raw_output: resp.output,
                        token_cost: resp.token_cost,
                        adapter_kind: resp.adapter_kind,
                        timestamp: Utc::now(),
                    },
                    tao_turns: turn,
                    iterations,
                });
            }

            memory.push(TaoMemoryEntry { turn, observation: observation.clone(), passed });
            req.system_context = format!(
                "{}{}",
                input.initial_request.system_context,
                format_memory(&memory)
            );
            req.task = format!(
                "{}\n\n{}",
                req.task,
                input.config.retry_instruction.replace("{turn}", &turn.to_string())
            );
        }

        // Fix I-1: explicit error instead of unreachable!()
        Err(EngineError::Adapter("TAO max_turns must be >= 1".into()))
    }
}
