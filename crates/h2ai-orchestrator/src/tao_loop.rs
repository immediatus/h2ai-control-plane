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
    /// MAPE-K retry-loop generation counter (0-based). Threaded into `ProposalEvent::generation`
    /// so `ProposalSet` can apply generation-first LUB semantics.
    pub generation: u64,
}

pub struct TaoProposal {
    pub event: ProposalEvent,
    pub tao_turns: u8,
    pub iterations: Vec<TaoIterationEvent>,
}

impl std::fmt::Debug for TaoProposal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaoProposal")
            .field("tao_turns", &self.tao_turns)
            .field("iterations_len", &self.iterations.len())
            .finish()
    }
}

/// Online estimator for the TAO loop per-turn quality improvement factor.
///
/// Records `q_after / q_before` from Tier 1 verified multi-turn tasks and
/// converges on an empirical multiplier. Falls back to the heuristic prior (0.6)
/// until 20 samples are accumulated.
#[derive(Debug, Clone, Default)]
pub struct TaoMultiplierEstimator {
    samples: Vec<f64>,
}

impl TaoMultiplierEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a TAO turn's quality change from a Tier 1 verified task.
    /// `q_before` and `q_after` must both be in [0, 1].
    pub fn update(&mut self, q_before: f64, q_after: f64) {
        if q_before > 0.0 {
            self.samples.push((q_after / q_before).clamp(0.0, 2.0));
        }
    }

    /// Current estimate of the per-turn decay factor.
    /// Returns the heuristic prior (0.6) until 20 samples are available.
    pub fn multiplier(&self) -> f64 {
        if self.samples.len() < 20 {
            0.6
        } else {
            self.samples.iter().sum::<f64>() / self.samples.len() as f64
        }
    }

    /// Number of samples collected so far.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
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
        let mut last_output: Option<String> = None;

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
                        schema_result
                            .as_invalid_msg()
                            .unwrap_or("unknown schema error"),
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
                        generation: input.generation,
                        raw_output: resp.output,
                        token_cost: resp.token_cost,
                        adapter_kind: resp.adapter_kind,
                        timestamp: Utc::now(),
                    },
                    tao_turns: turn,
                    iterations,
                });
            }

            // Turn failed and more turns remain. Check for stuck repetition loop.
            if let Some(ref prev) = last_output {
                let sim = crate::repetition::similarity(prev, &resp.output);
                if sim >= input.config.repetition_threshold {
                    return Err(EngineError::Adapter(format!(
                        "TAO repetition detected at turn {turn}: similarity {sim:.2} \
                         >= threshold {:.2}",
                        input.config.repetition_threshold
                    )));
                }
            }
            last_output = Some(resp.output.clone());

            memory.push(TaoMemoryEntry {
                turn,
                observation: observation.clone(),
                passed,
            });
            req.system_context = format!(
                "{}{}",
                input.initial_request.system_context,
                format_memory(&memory)
            );
            req.task = format!(
                "{}\n\n{}",
                req.task,
                input
                    .config
                    .retry_instruction
                    .replace("{turn}", &turn.to_string())
            );
        }

        // Fix I-1: explicit error instead of unreachable!()
        Err(EngineError::Adapter("TAO max_turns must be >= 1".into()))
    }
}
