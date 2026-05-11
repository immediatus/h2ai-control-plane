use dashmap::DashMap;
use futures::future::join_all;
use h2ai_config::prompts::VERIFICATION_TASK;
use h2ai_constraints::eval::eval_sync;
use h2ai_constraints::types::{
    aggregate_compliance_score, ComplianceResult, CompositeOp, ConstraintDoc, ConstraintPredicate,
    ConstraintSeverity,
};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::VerificationConfig;
use h2ai_types::events::{ConstraintViolation, ProposalEvent};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

/// Per-task evaluation cache: maps constraint_id → Vec<(proposal_text, score)>.
/// Shared across concurrent explorer evaluations via DashMap (no blocking mutex).
/// Created fresh per task; dropped when the task's verification phase completes.
pub type EvalCache = Arc<DashMap<String, Vec<(String, f64)>>>;

pub fn new_eval_cache() -> EvalCache {
    Arc::new(DashMap::new())
}

const CACHE_SIMILARITY_THRESHOLD: f64 = 0.85;

/// One `bool` per constraint in the corpus: `true` = hard gate passed.
/// Derived from `Vec<ComplianceResult>` via `results.iter().map(|r| r.hard_passes()).collect()`.
pub type SatisfactionFingerprint = Vec<bool>;

pub struct VerificationInput<'a> {
    pub proposals: Vec<ProposalEvent>,
    pub constraint_corpus: &'a [ConstraintDoc],
    pub evaluator: &'a dyn IComputeAdapter,
    pub config: VerificationConfig,
    /// Per-task eval cache. Pass the same `Arc` across retry rounds to share hits within a task.
    pub eval_cache: EvalCache,
    /// Number of LLM judge passes for Hard LlmJudge constraints. Averaged. Default 1.
    pub consensus_passes: u8,
}

pub struct VerificationOutput {
    /// (proposal, per_constraint_results, any_cache_hit)
    pub passed: Vec<(ProposalEvent, Vec<ComplianceResult>, bool)>,
    /// (proposal, per_constraint_results, violations, any_cache_hit)
    pub failed: Vec<(
        ProposalEvent,
        Vec<ComplianceResult>,
        Vec<ConstraintViolation>,
        bool,
    )>,
    /// Populated only when `config.record_adversarial_comparison == true`.
    pub comparison_events: Vec<h2ai_types::events::VerifierComparisonEvent>,
}

#[derive(Deserialize)]
struct ScoreResponse {
    score: f64,
    reason: String,
}

pub struct VerificationPhase;

impl VerificationPhase {
    pub async fn run(input: VerificationInput<'_>) -> VerificationOutput {
        let evaluator = input.evaluator;
        let corpus = input.constraint_corpus;
        let threshold = input.config.threshold;
        let rubric = input.config.rubric.clone();
        let sp = input.config.evaluator_system_prompt.clone();
        let tau = input.config.evaluator_tau;
        let max_tokens = input.config.evaluator_max_tokens;
        let record_adversarial_comparison = input.config.record_adversarial_comparison;
        let input_config = input.config.clone();
        let consensus_passes = input.consensus_passes;
        // Fresh cache for the adversarial pass: sharing the standard cache would pollute
        // standard-score entries with adversarial scores, causing incorrect cache hits
        // on retry waves when the same proposals are re-verified.
        let eval_cache_for_adv = new_eval_cache();
        let eval_cache = input.eval_cache;

        let futures = input.proposals.into_iter().map(|proposal| {
            let rubric = rubric.clone();
            let sp = sp.clone();
            let cache = Arc::clone(&eval_cache);
            async move {
                let (results, any_cache_hit) = Self::eval_all(
                    corpus,
                    &proposal.raw_output,
                    evaluator,
                    &rubric,
                    &sp,
                    tau,
                    max_tokens,
                    &cache,
                    consensus_passes,
                )
                .await;
                (proposal, results, any_cache_hit)
            }
        });

        let all = join_all(futures).await;
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        for (proposal, results, any_cache_hit) in all {
            let hard_gate = results.iter().all(|r| r.hard_passes());
            let soft_score = aggregate_compliance_score(&results);
            let overall = if hard_gate { soft_score } else { 0.0 };

            if overall >= threshold {
                passed.push((proposal, results, any_cache_hit));
            } else {
                let violations: Vec<ConstraintViolation> = results
                    .iter()
                    .filter(|r| !r.hard_passes() || r.score < threshold)
                    .map(|r| ConstraintViolation {
                        constraint_id: r.constraint_id.clone(),
                        score: r.score,
                        severity_label: severity_label(&r.severity),
                        remediation_hint: r.remediation_hint.clone(),
                    })
                    .collect();
                failed.push((proposal, results, violations, any_cache_hit));
            }
        }

        let output = VerificationOutput {
            passed,
            failed,
            comparison_events: vec![],
        };

        let comparison_events = if record_adversarial_comparison {
            use h2ai_types::prompts::ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT;

            let mut adv_config = input_config;
            adv_config.evaluator_system_prompt = ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT.to_string();
            // Disable comparison in adversarial pass to prevent recursion
            adv_config.record_adversarial_comparison = false;

            // Collect all proposals (passed + failed from normal run)
            let all_proposals: Vec<ProposalEvent> = output
                .passed
                .iter()
                .map(|(p, _, _)| p.clone())
                .chain(output.failed.iter().map(|(p, _, _, _)| p.clone()))
                .collect();

            let adv_output = Box::pin(VerificationPhase::run(VerificationInput {
                proposals: all_proposals.clone(),
                constraint_corpus: corpus,
                evaluator,
                config: adv_config,
                eval_cache: eval_cache_for_adv,
                consensus_passes,
            }))
            .await;

            // Build score lookup: explorer_id → (score, passed)
            let mut adv_map: std::collections::HashMap<
                h2ai_types::identity::ExplorerId,
                (f64, bool),
            > = std::collections::HashMap::new();
            for (p, results, _) in &adv_output.passed {
                adv_map.insert(
                    p.explorer_id.clone(),
                    (aggregate_compliance_score(results), true),
                );
            }
            for (p, results, _, _) in &adv_output.failed {
                adv_map.insert(
                    p.explorer_id.clone(),
                    (aggregate_compliance_score(results), false),
                );
            }

            let mut std_map: std::collections::HashMap<
                h2ai_types::identity::ExplorerId,
                (f64, bool),
            > = std::collections::HashMap::new();
            for (p, results, _) in &output.passed {
                std_map.insert(
                    p.explorer_id.clone(),
                    (aggregate_compliance_score(results), true),
                );
            }
            for (p, results, _, _) in &output.failed {
                std_map.insert(
                    p.explorer_id.clone(),
                    (aggregate_compliance_score(results), false),
                );
            }

            all_proposals
                .iter()
                .filter_map(|p| {
                    let (std_score, std_passed) = std_map.get(&p.explorer_id)?;
                    let (adv_score, adv_passed) = adv_map.get(&p.explorer_id)?;
                    Some(h2ai_types::events::VerifierComparisonEvent {
                        task_id: p.task_id.clone(),
                        explorer_id: p.explorer_id.clone(),
                        standard_score: *std_score,
                        adversarial_score: *adv_score,
                        standard_passed: *std_passed,
                        adversarial_passed: *adv_passed,
                        verifier_kind: "llmjudge".to_string(),
                        timestamp: chrono::Utc::now(),
                    })
                })
                .collect()
        } else {
            vec![]
        };

        VerificationOutput {
            passed: output.passed,
            failed: output.failed,
            comparison_events,
        }
    }

    /// Score proposals numerically without pass/fail gating.
    /// Returns `(proposal, aggregate_compliance_score)` for each input, in order.
    /// Used by the engine to score turn-1 outputs and feed `TaoMultiplierEstimator`.
    pub async fn score_proposals(
        proposals: Vec<ProposalEvent>,
        evaluator: &dyn IComputeAdapter,
        config: &VerificationConfig,
        corpus: &[ConstraintDoc],
    ) -> Vec<(ProposalEvent, f64)> {
        let rubric = config.rubric.clone();
        let sp = config.evaluator_system_prompt.clone();
        let tau = config.evaluator_tau;
        let max_tokens = config.evaluator_max_tokens;

        let scoring_cache = new_eval_cache();
        let futures = proposals.into_iter().map(|proposal| {
            let rubric = rubric.clone();
            let sp = sp.clone();
            let cache = Arc::clone(&scoring_cache);
            async move {
                let (results, _) = Self::eval_all(
                    corpus,
                    &proposal.raw_output,
                    evaluator,
                    &rubric,
                    &sp,
                    tau,
                    max_tokens,
                    &cache,
                    1, // score_proposals uses single-pass scoring (used for TAO estimator)
                )
                .await;
                let score = aggregate_compliance_score(&results);
                (proposal, score)
            }
        });
        join_all(futures).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn eval_all(
        corpus: &[ConstraintDoc],
        output: &str,
        evaluator: &dyn IComputeAdapter,
        rubric: &str,
        sp: &str,
        tau: h2ai_types::sizing::TauValue,
        max_tokens: u64,
        cache: &EvalCache,
        consensus_passes: u8,
    ) -> (Vec<ComplianceResult>, bool) {
        // If corpus is empty, fall back to the CoT rubric (G-Eval, arxiv 2303.16634).
        // The default rubric (h2ai_config::prompts::COT_RUBRIC) is criteria-first to reduce
        // verbosity bias. Operators may override via VerificationConfig::rubric.
        // llm_score_raw appends "\n\nProposal:\n{output}", so we pass only the criteria here.
        if corpus.is_empty() {
            let effective_rubric: &str = if rubric.is_empty() {
                h2ai_config::prompts::COT_RUBRIC
            } else {
                rubric
            };
            let score =
                Self::llm_score_raw(effective_rubric, output, evaluator, sp, tau, max_tokens).await;
            return (
                vec![ComplianceResult {
                    constraint_id: "__rubric__".into(),
                    score,
                    severity: ConstraintSeverity::Hard { threshold: 0.45 },
                    remediation_hint: None,
                }],
                false,
            );
        }

        let futs = corpus.iter().map(|doc| {
            let constraint_id = doc.id.clone();
            let severity = doc.severity.clone();
            let remediation_hint = doc.remediation_hint.clone();
            let predicate = doc.predicate.clone();
            let output = output.to_owned();
            let cache = Arc::clone(cache);
            let sp = sp.to_owned();
            async move {
                // Check if a sufficiently similar proposal was already scored for this constraint.
                let cached_score = cache.get(&constraint_id).and_then(|entries| {
                    entries
                        .iter()
                        .find(|(prev, _)| {
                            crate::repetition::similarity(prev, &output)
                                >= CACHE_SIMILARITY_THRESHOLD
                        })
                        .map(|(_, score)| *score)
                });

                // For Hard constraints, apply multi-pass consensus when consensus_passes > 1.
                let effective_passes = match &severity {
                    ConstraintSeverity::Hard { .. } => consensus_passes.max(1),
                    _ => 1,
                };

                let (score, hit) = if let Some(score) = cached_score {
                    tracing::debug!(
                        target: "h2ai.verification.cache",
                        constraint_id = %constraint_id,
                        score,
                        "eval cache hit — reusing score for similar proposal"
                    );
                    (score, true)
                } else {
                    let score = Self::eval_predicate_async(
                        &predicate,
                        &output,
                        evaluator,
                        &sp,
                        tau,
                        max_tokens,
                        effective_passes,
                    )
                    .await;
                    cache
                        .entry(constraint_id.clone())
                        .or_default()
                        .push((output.clone(), score));
                    (score, false)
                };

                (
                    ComplianceResult {
                        constraint_id,
                        score,
                        severity,
                        remediation_hint,
                    },
                    hit,
                )
            }
        });

        let results: Vec<(ComplianceResult, bool)> = join_all(futs).await;
        let hit_flag = results.iter().any(|(_, h)| *h);
        (results.into_iter().map(|(r, _)| r).collect(), hit_flag)
    }

    /// Evaluate any predicate, including Composite trees that contain LlmJudge children.
    /// Returns a score in [0.0, 1.0]. Uses Box::pin for recursive async support.
    ///
    /// For `Composite { And, children }`, static children are evaluated first. If any
    /// returns 0.0 (hard failure — e.g. NegativeKeyword found a prohibited term), heavy
    /// children (LlmJudge, Oracle) are skipped entirely. This avoids spurious LLM calls
    /// when a proposal already fails on fast deterministic checks.
    fn eval_predicate_async<'a>(
        pred: &'a ConstraintPredicate,
        output: &'a str,
        evaluator: &'a dyn IComputeAdapter,
        sp: &'a str,
        tau: h2ai_types::sizing::TauValue,
        max_tokens: u64,
        consensus_passes: u8,
    ) -> Pin<Box<dyn Future<Output = f64> + Send + 'a>> {
        Box::pin(async move {
            match pred {
                ConstraintPredicate::LlmJudge { rubric } => {
                    let passes = consensus_passes.max(1) as usize;
                    let mut scores = Vec::with_capacity(passes);
                    for _ in 0..passes {
                        let s = match tokio::time::timeout(
                            std::time::Duration::from_secs(600),
                            Self::llm_score_raw(rubric, output, evaluator, sp, tau, max_tokens),
                        )
                        .await
                        {
                            Ok(score) => score,
                            Err(_) => {
                                tracing::warn!(
                                    target: "h2ai.verification",
                                    "LlmJudge timed out (600s); skipping — score defaults to 0.5"
                                );
                                0.5
                            }
                        };
                        scores.push(s);
                    }
                    scores.iter().sum::<f64>() / scores.len() as f64
                }
                ConstraintPredicate::OracleExecution {
                    test_runner_uri,
                    test_suite,
                    timeout_secs,
                } => Self::eval_oracle(test_runner_uri, test_suite, *timeout_secs, output).await,
                ConstraintPredicate::Composite { op, children } => {
                    match op {
                        CompositeOp::And => {
                            // Evaluate static children first; short-circuit if any hits 0.0.
                            let mut min_score = 1.0_f64;
                            let mut deferred = Vec::new();
                            for child in children {
                                match child {
                                    ConstraintPredicate::LlmJudge { .. }
                                    | ConstraintPredicate::OracleExecution { .. } => {
                                        deferred.push(child);
                                    }
                                    other => {
                                        let s = eval_sync(other, output);
                                        min_score = min_score.min(s);
                                        if min_score <= 0.0 {
                                            return 0.0; // hard failure on static check
                                        }
                                    }
                                }
                            }
                            // Only call LlmJudge if static predicates all passed.
                            for child in deferred {
                                let s = Self::eval_predicate_async(
                                    child,
                                    output,
                                    evaluator,
                                    sp,
                                    tau,
                                    max_tokens,
                                    consensus_passes,
                                )
                                .await;
                                min_score = min_score.min(s);
                                if min_score <= 0.0 {
                                    return 0.0;
                                }
                            }
                            min_score
                        }
                        CompositeOp::Or => {
                            let mut max_score = 0.0_f64;
                            for child in children {
                                let s = Self::eval_predicate_async(
                                    child,
                                    output,
                                    evaluator,
                                    sp,
                                    tau,
                                    max_tokens,
                                    consensus_passes,
                                )
                                .await;
                                max_score = max_score.max(s);
                                if max_score >= 1.0 {
                                    return 1.0;
                                }
                            }
                            max_score
                        }
                        CompositeOp::Not => {
                            let s = if let Some(child) = children.first() {
                                Self::eval_predicate_async(
                                    child,
                                    output,
                                    evaluator,
                                    sp,
                                    tau,
                                    max_tokens,
                                    consensus_passes,
                                )
                                .await
                            } else {
                                0.0
                            };
                            1.0 - s
                        }
                    }
                }
                other => eval_sync(other, output),
            }
        })
    }

    async fn eval_oracle(
        test_runner_uri: &str,
        test_suite: &str,
        timeout_secs: u64,
        output: &str,
    ) -> f64 {
        #[derive(Serialize)]
        struct OracleRequest<'a> {
            output: &'a str,
            test_suite: &'a str,
        }

        #[derive(Deserialize)]
        struct OracleResponse {
            passed: bool,
            #[allow(dead_code)]
            failure_count: u32,
            #[allow(dead_code)]
            output_text: String,
            #[allow(dead_code)]
            duration_ms: u64,
        }

        let client = reqwest::Client::new();
        let body = OracleRequest { output, test_suite };
        match client
            .post(test_runner_uri)
            .json(&body)
            .timeout(Duration::from_secs(timeout_secs))
            .send()
            .await
        {
            Ok(resp) => match resp.json::<OracleResponse>().await {
                Ok(or) => {
                    if !or.passed {
                        tracing::debug!(
                            target: "h2ai.verification.oracle",
                            failure_count = or.failure_count,
                            "oracle execution failed"
                        );
                    }
                    if or.passed {
                        1.0
                    } else {
                        0.0
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target: "h2ai.verification.oracle",
                        error = %e,
                        "oracle response parse error"
                    );
                    0.0
                }
            },
            Err(e) => {
                if e.is_timeout() {
                    tracing::warn!(
                        target: "h2ai.verification.oracle",
                        uri = test_runner_uri,
                        "oracle_timeout"
                    );
                } else {
                    tracing::warn!(
                        target: "h2ai.verification.oracle",
                        error = %e,
                        uri = test_runner_uri,
                        "oracle request failed"
                    );
                }
                0.0
            }
        }
    }

    async fn llm_score_raw(
        rubric: &str,
        output: &str,
        evaluator: &dyn IComputeAdapter,
        sp: &str,
        tau: h2ai_types::sizing::TauValue,
        max_tokens: u64,
    ) -> f64 {
        // Separate criterion (what to check) from the proposal (what to score).
        // The JSON response format is owned by EVALUATOR_SYSTEM_PROMPT — rubrics must
        // not repeat it; they contain only behavioral pass/fail criteria.
        let prompt = VERIFICATION_TASK.render(&[("rubric", rubric), ("output", output)]);
        let req = ComputeRequest {
            system_context: sp.to_owned(),
            task: prompt,
            tau,
            max_tokens,
        };
        match evaluator.execute(req).await {
            Ok(resp) => match extract_json_object::<ScoreResponse>(&resp.output) {
                Some(s) => {
                    tracing::info!(
                        target: "h2ai.verification",
                        score = s.score,
                        reason = %s.reason,
                        "LlmJudge scored"
                    );
                    s.score.clamp(0.0, 1.0)
                }
                // JSON parse failure: model did not emit a score object.
                // Fall back to neutral (0.7) so static predicates remain the actual gate.
                None => {
                    tracing::info!(
                        target: "h2ai.verification",
                        raw = %resp.output,
                        "LlmJudge response did not contain JSON score object; using neutral 0.7"
                    );
                    0.7
                }
            },
            Err(e) => {
                tracing::warn!(target: "h2ai.verification", error = %e, "LlmJudge execute error; using neutral 0.7");
                0.7
            }
        }
    }
}

/// Extract the last valid JSON object `{...}` from a string that may contain
/// surrounding prose or markdown code fences (e.g. ```json ... ```).
///
/// Returns the LAST valid match rather than the first. Reasoning models (e.g.
/// DeepSeek-R1 / Qwen3 thinking mode) embed intermediate JSON objects in their
/// chain-of-thought before writing the final answer. Returning the last object
/// ensures we read the model's conclusion, not an intermediate consideration.
pub(crate) fn extract_json_object<T: serde::de::DeserializeOwned>(text: &str) -> Option<T> {
    // Fast path: whole string is valid JSON.
    if let Ok(v) = serde_json::from_str::<T>(text) {
        return Some(v);
    }
    // Scan every `{...}` span; keep the LAST one that parses successfully.
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut last_valid: Option<T> = None;
    for start in 0..n {
        if chars[start] != '{' {
            continue;
        }
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        for end in start..n {
            let c = chars[end];
            if escaped {
                escaped = false;
            } else if c == '\\' && in_string {
                escaped = true;
            } else if c == '"' {
                in_string = !in_string;
            } else if !in_string {
                if c == '{' {
                    depth += 1;
                } else if c == '}' {
                    depth -= 1;
                    if depth == 0 {
                        let slice: String = chars[start..=end].iter().collect();
                        if let Ok(v) = serde_json::from_str::<T>(&slice) {
                            last_valid = Some(v);
                        }
                        break;
                    }
                }
            }
        }
    }
    last_valid
}

fn severity_label(s: &ConstraintSeverity) -> String {
    match s {
        ConstraintSeverity::Hard { .. } => "Hard".into(),
        ConstraintSeverity::Soft { .. } => "Soft".into(),
        ConstraintSeverity::Advisory => "Advisory".into(),
    }
}
