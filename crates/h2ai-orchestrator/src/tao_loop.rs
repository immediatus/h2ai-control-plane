use crate::engine::EngineError;
use crate::output_schema::{validate_output, SchemaValidationResult};
use chrono::Utc;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::{OutputSchemaConfig, TaoConfig};
use h2ai_types::events::{ProposalEvent, TaoIterationEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use regex::Regex;
use serde::{Deserialize, Serialize};
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

/// All inputs required to drive a single TAO (Think–Act–Observe) loop execution for one explorer.
pub struct TaoInput<'a> {
    /// Identifier of the parent task, threaded into every emitted event.
    pub task_id: TaskId,
    /// Identifier of the explorer running this TAO loop.
    pub explorer_id: ExplorerId,
    /// Compute adapter the loop calls on each turn.
    pub adapter: &'a dyn IComputeAdapter,
    /// The initial compute request; the system context and task prompt are extended with TAO memory on each retry turn.
    pub initial_request: ComputeRequest,
    /// TAO loop configuration: `max_turns`, patterns, observation templates, and repetition threshold.
    pub config: TaoConfig,
    /// Optional JSON schema used to validate adapter output on each turn.
    pub schema_config: Option<OutputSchemaConfig>,
    /// MAPE-K retry-loop generation counter (0-based). Threaded into `ProposalEvent::generation`
    /// so `ProposalSet` can apply generation-first LUB semantics.
    pub generation: u64,
}

/// Result produced by a completed TAO loop for one explorer.
pub struct TaoProposal {
    /// The proposal event to be published and inserted into the `ProposalSet`.
    pub event: ProposalEvent,
    /// Number of TAO turns that were executed (1 = passed on the first turn).
    pub tao_turns: u8,
    /// Per-turn iteration events recording observations and pass/fail status.
    pub iterations: Vec<TaoIterationEvent>,
    /// Output from TAO loop turn 1. `Some` only when `tao_turns > 1`.
    /// Retained so the engine can score this intermediate output via a second
    /// verification pass and feed `TaoMultiplierEstimator` with `(score_turn1,
    /// score_final)` pairs — enabling online estimation of per-turn quality gain.
    pub turn1_output: Option<String>,
}

impl std::fmt::Debug for TaoProposal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaoProposal")
            .field("tao_turns", &self.tao_turns)
            .field("iterations_len", &self.iterations.len())
            .field("has_turn1_output", &self.turn1_output.is_some())
            .finish()
    }
}

/// Online estimator for the TAO loop per-turn quality improvement factor.
///
/// Tracks `q_after / q_before` ratios from multi-turn proposals using an EMA
/// after a 20-sample warm-up. Persists to NATS KV (`H2AI_ESTIMATOR`) so the
/// estimate survives restarts and tracks drift. Falls back to 0.6 prior until
/// 20 samples are collected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaoMultiplierEstimator {
    /// Current EMA value. Only valid (non-zero) once `count` reaches 20;
    /// callers must use `multiplier()` which returns the 0.6 prior below that threshold.
    ema: f64,
    /// Number of valid samples recorded so far.
    pub count: usize,
    /// EMA smoothing factor α. Not persisted — zeroed on deserialisation and restored
    /// via `with_alpha()` from config before use.
    #[serde(skip)]
    alpha: f64,
    /// Accumulator for the arithmetic mean over the first 20 warm-up samples.
    /// Not persisted — zeroed on deserialisation. When `count < 20` at restart
    /// the warm-up mean is recomputed from only post-restart samples; the NATS put
    /// path avoids this by skipping persistence until warm-up completes (see `persist_state`).
    /// Zeroed automatically once warm-up finishes and `ema` is initialised.
    #[serde(skip)]
    warmup_sum: f64,
}

impl TaoMultiplierEstimator {
    pub fn new_with_alpha(alpha: f64) -> Self {
        Self {
            ema: 0.0,
            count: 0,
            alpha,
            warmup_sum: 0.0,
        }
    }

    /// Restore alpha after deserializing from NATS.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Record a quality change from a multi-turn proposal.
    /// `q_before` is the turn-1 verification score; `q_after` is the final score.
    /// No-op if `q_before <= 0.0`.
    pub fn update(&mut self, q_before: f64, q_after: f64) {
        if q_before <= 0.0 {
            return;
        }
        let ratio = (q_after / q_before).clamp(0.0, 2.0);
        self.count += 1;
        if self.count < 20 {
            self.warmup_sum += ratio;
        } else if self.count == 20 {
            self.warmup_sum += ratio;
            self.ema = self.warmup_sum / 20.0;
        } else {
            self.ema += self.alpha * (ratio - self.ema);
        }
    }

    /// Current estimate of the per-turn quality factor.
    /// Returns the heuristic prior (0.6) until 20 samples are available.
    pub fn multiplier(&self) -> f64 {
        if self.count < 20 {
            0.6
        } else {
            self.ema
        }
    }

    /// Number of valid samples collected so far.
    pub fn sample_count(&self) -> usize {
        self.count
    }

    /// Returns `(ema, count)` for NATS persistence, or `None` when warm-up is incomplete
    /// (count < 20). Callers must not persist partial warm-up state.
    pub fn persist_state(&self) -> Option<(f64, usize)> {
        if self.count >= 20 {
            Some((self.ema, self.count))
        } else {
            None
        }
    }
}

/// Stateless driver for the Think–Act–Observe (TAO) iterative refinement loop.
///
/// Repeatedly calls the compute adapter, evaluates the output against a regex pattern
/// and optional JSON schema, and injects TAO memory (prior observations) into the prompt
/// on each failed turn until the output passes or `max_turns` is exhausted.
pub struct TaoLoop;

impl TaoLoop {
    /// Execute the TAO loop for one explorer and return its proposal.
    ///
    /// Returns `Err(EngineError::Adapter)` on timeout, invalid configuration, or a
    /// repetition-loop detection (consecutive outputs too similar).
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
        let mut turn1_output: Option<String> = None;

        for turn in 1..=input.config.max_turns {
            let resp = timeout(Duration::from_secs(30), input.adapter.execute(req.clone()))
                .await
                .map_err(|_| EngineError::Adapter("TAO timeout".into()))?
                .map_err(|e| EngineError::Adapter(e.to_string()))?;

            if turn == 1 {
                turn1_output = Some(resp.output.clone());
            }

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
                    turn1_output: if turn > 1 { turn1_output } else { None },
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
