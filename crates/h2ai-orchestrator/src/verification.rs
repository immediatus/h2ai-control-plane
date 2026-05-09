use futures::future::join_all;
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
use std::time::Duration;

/// One `bool` per constraint in the corpus: `true` = hard gate passed.
/// Derived from `Vec<ComplianceResult>` via `results.iter().map(|r| r.hard_passes()).collect()`.
pub type SatisfactionFingerprint = Vec<bool>;

pub struct VerificationInput<'a> {
    pub proposals: Vec<ProposalEvent>,
    pub constraint_corpus: &'a [ConstraintDoc],
    pub evaluator: &'a dyn IComputeAdapter,
    pub config: VerificationConfig,
}

pub struct VerificationOutput {
    /// (proposal, per_constraint_results)
    pub passed: Vec<(ProposalEvent, Vec<ComplianceResult>)>,
    /// (proposal, per_constraint_results, violations)
    pub failed: Vec<(
        ProposalEvent,
        Vec<ComplianceResult>,
        Vec<ConstraintViolation>,
    )>,
}

#[derive(Deserialize)]
struct ScoreResponse {
    score: f64,
    #[allow(dead_code)]
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

        let futures = input.proposals.into_iter().map(|proposal| {
            let rubric = rubric.clone();
            let sp = sp.clone();
            async move {
                let results = Self::eval_all(
                    corpus,
                    &proposal.raw_output,
                    evaluator,
                    &rubric,
                    &sp,
                    tau,
                    max_tokens,
                )
                .await;
                (proposal, results)
            }
        });

        let all = join_all(futures).await;
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        for (proposal, results) in all {
            let hard_gate = results.iter().all(|r| r.hard_passes());
            let soft_score = aggregate_compliance_score(&results);
            let overall = if hard_gate { soft_score } else { 0.0 };

            if overall >= threshold {
                passed.push((proposal, results));
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
                failed.push((proposal, results, violations));
            }
        }

        VerificationOutput { passed, failed }
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

        let futures = proposals.into_iter().map(|proposal| {
            let rubric = rubric.clone();
            let sp = sp.clone();
            async move {
                let results = Self::eval_all(
                    corpus,
                    &proposal.raw_output,
                    evaluator,
                    &rubric,
                    &sp,
                    tau,
                    max_tokens,
                )
                .await;
                let score = aggregate_compliance_score(&results);
                (proposal, score)
            }
        });
        join_all(futures).await
    }

    async fn eval_all(
        corpus: &[ConstraintDoc],
        output: &str,
        evaluator: &dyn IComputeAdapter,
        rubric: &str,
        sp: &str,
        tau: h2ai_types::sizing::TauValue,
        max_tokens: u64,
    ) -> Vec<ComplianceResult> {
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
            return vec![ComplianceResult {
                constraint_id: "__rubric__".into(),
                score,
                severity: ConstraintSeverity::Hard { threshold: 0.45 },
                remediation_hint: None,
            }];
        }

        let futs = corpus.iter().map(|doc| async move {
            let score =
                Self::eval_predicate_async(&doc.predicate, output, evaluator, sp, tau, max_tokens)
                    .await;
            ComplianceResult {
                constraint_id: doc.id.clone(),
                score,
                severity: doc.severity.clone(),
                remediation_hint: doc.remediation_hint.clone(),
            }
        });
        join_all(futs).await
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
    ) -> Pin<Box<dyn Future<Output = f64> + Send + 'a>> {
        Box::pin(async move {
            match pred {
                ConstraintPredicate::LlmJudge { rubric } => {
                    // Apply a per-call timeout so slow local models don't stall verification.
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(90),
                        Self::llm_score_raw(rubric, output, evaluator, sp, tau, max_tokens),
                    )
                    .await
                    {
                        Ok(score) => score,
                        Err(_) => {
                            tracing::warn!(
                                target: "h2ai.verification",
                                "LlmJudge timed out (90s); skipping — score defaults to 0.5"
                            );
                            0.5
                        }
                    }
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
                                    child, output, evaluator, sp, tau, max_tokens,
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
                                    child, output, evaluator, sp, tau, max_tokens,
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
                                    child, output, evaluator, sp, tau, max_tokens,
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
        let prompt = format!("Criterion:\n{rubric}\n\nProposal:\n{output}");
        let req = ComputeRequest {
            system_context: sp.to_owned(),
            task: prompt,
            tau,
            max_tokens,
        };
        match evaluator.execute(req).await {
            Ok(resp) => match extract_json_object::<ScoreResponse>(&resp.output) {
                Some(s) => s.score.clamp(0.0, 1.0),
                // JSON parse failure: model did not emit a score object.
                // Fall back to neutral (0.7) so static predicates remain the actual gate.
                None => {
                    tracing::debug!(
                        target: "h2ai.verification",
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

/// Extract the first valid JSON object `{...}` from a string that may contain
/// surrounding prose or markdown code fences (e.g. ```json ... ```).
pub(crate) fn extract_json_object<T: serde::de::DeserializeOwned>(text: &str) -> Option<T> {
    // Fast path: whole string is valid JSON.
    if let Ok(v) = serde_json::from_str::<T>(text) {
        return Some(v);
    }
    // Try every `{...}` span: find each `{`, pair with the matching `}`, parse.
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
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
                            return Some(v);
                        }
                        break;
                    }
                }
            }
        }
    }
    None
}

fn severity_label(s: &ConstraintSeverity) -> String {
    match s {
        ConstraintSeverity::Hard { .. } => "Hard".into(),
        ConstraintSeverity::Soft { .. } => "Soft".into(),
        ConstraintSeverity::Advisory => "Advisory".into(),
    }
}
