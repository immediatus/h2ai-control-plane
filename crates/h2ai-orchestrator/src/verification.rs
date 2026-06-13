use dashmap::DashMap;
use futures::future::join_all;
use h2ai_config::prompts::VERIFICATION_TASK;
use h2ai_config::JudgePanelConfig;
use h2ai_constraints::eval::eval_sync;
use h2ai_constraints::types::{
    aggregate_compliance_score, ComplianceResult, CompositeOp, ConstraintDoc, ConstraintPredicate,
    ConstraintSeverity,
};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::VerificationConfig;
use h2ai_types::events::{ConstraintViolation, ProposalEvent};
use h2ai_types::identity::ExplorerId;
use h2ai_types::prompts::BINARY_CLASSIFIER_SYSTEM_PROMPT;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::judge_panel::{aggregate_votes, ConstraintVerdict, JudgePanel};

/// Derive a deterministic compliance score from binary check verdicts.
///
/// Returns `present_checks / n_checks` when verdicts are non-empty and `n_checks > 0`.
/// Falls back to `llm_float` for constraints with no binary checks (empty verdicts
/// or n_checks == 0) — those constraints continue using the LLM's holistic score.
pub fn score_from_verdicts(verdicts: &[bool], n_checks: usize, llm_float: f64) -> f64 {
    if n_checks == 0 || verdicts.is_empty() {
        return llm_float;
    }
    let present = verdicts.iter().filter(|&&v| v).count();
    present as f64 / n_checks as f64
}

/// Per-task evaluation cache: maps `constraint_id` → Vec<(`proposal_text`, score)>.
///
/// Shared across concurrent explorer evaluations via `DashMap` (no blocking mutex).
/// Created fresh per task; dropped when the task's verification phase completes.
pub type EvalCache = Arc<DashMap<String, Vec<(String, f64)>>>;

#[must_use]
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
    /// Number of LLM judge passes for Hard `LlmJudge` constraints. Averaged. Default 1.
    pub consensus_passes: u8,
}

pub struct VerificationOutput {
    /// (proposal, `per_constraint_results`, `any_cache_hit`)
    pub passed: Vec<(ProposalEvent, Vec<ComplianceResult>, bool)>,
    /// (proposal, `per_constraint_results`, violations, `any_cache_hit`)
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
        let ct_scale = input.config.constraint_threshold_scale;
        let rubric = input.config.rubric.clone();
        let sp = input.config.evaluator_system_prompt.clone();
        let tau = input.config.evaluator_tau;
        let max_tokens = input.config.evaluator_max_tokens;
        let evaluator_timeout_secs = input.config.evaluator_timeout_secs;
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
                    threshold,
                    evaluator_timeout_secs,
                )
                .await;
                (proposal, results, any_cache_hit)
            }
        });

        let all = join_all(futures).await;
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        for (proposal, results, any_cache_hit) in all {
            let hard_gate = results.iter().all(|r| r.hard_passes_scaled(ct_scale));
            let soft_score = aggregate_compliance_score(&results);
            let overall = if hard_gate { soft_score } else { 0.0 };

            if overall >= threshold {
                passed.push((proposal, results, any_cache_hit));
            } else {
                let violations: Vec<ConstraintViolation> = results
                    .iter()
                    .filter(|r| !r.hard_passes_scaled(ct_scale) || r.score < threshold)
                    .map(|r| ConstraintViolation {
                        constraint_id: r.constraint_id.clone(),
                        score: r.score,
                        severity_label: severity_label(&r.severity),
                        remediation_hint: r.remediation_hint.clone(),
                        constraint_description: r.constraint_description.clone(),
                        verifier_reason: r.verifier_reason.clone(),
                        check_verdicts: r.check_verdicts.clone(),
                        criteria_pass: r.criteria_pass.clone(),
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

            let adv_output = Box::pin(Self::run(VerificationInput {
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

    /// Multi-variant panel evaluation. Fires all panel variants in parallel per constraint,
    /// aggregates votes into `ConstraintVerdict`. Returns standard `VerificationOutput`
    /// plus a map of `ExplorerId → Vec<ConstraintId>` for uncertain constraints.
    pub async fn run_with_panel(
        input: VerificationInput<'_>,
        panel: &JudgePanel<'_>,
        cfg: &JudgePanelConfig,
    ) -> (
        VerificationOutput,
        std::collections::HashMap<ExplorerId, Vec<String>>,
    ) {
        // Single-variant shortcut: delegate to run() with empty uncertain map.
        if panel.variants.len() == 1 {
            let out = Self::run(input).await;
            return (out, std::collections::HashMap::new());
        }

        // --- Multi-variant path ---
        let corpus = input.constraint_corpus;
        let threshold = input.config.threshold;
        let ct_scale = input.config.constraint_threshold_scale;
        let rubric = input.config.rubric.clone();
        let base_sp = input.config.evaluator_system_prompt.clone();
        let tau = input.config.evaluator_tau;
        let max_tokens = input.config.evaluator_max_tokens;
        let evaluator_timeout_secs = input.config.evaluator_timeout_secs;
        let consensus_passes = input.consensus_passes;
        let uncertainty_weight = cfg.uncertainty_weight;

        let mut passed = Vec::new();
        let mut failed = Vec::new();
        let mut uncertain_map: std::collections::HashMap<ExplorerId, Vec<String>> =
            std::collections::HashMap::new();

        // Evaluate each proposal independently.
        for proposal in input.proposals {
            // Pre-compute per-variant system prompts and caches.
            let variant_contexts: Vec<(String, EvalCache)> = panel
                .variants
                .iter()
                .map(|variant| {
                    let persona_prefix = variant.persona.system_prompt_prefix();
                    let sp = if persona_prefix.is_empty() {
                        base_sp.clone()
                    } else {
                        format!("{persona_prefix}\n\n{base_sp}")
                    };
                    (sp, new_eval_cache())
                })
                .collect();

            // Fire all variants in parallel per proposal.
            let variant_results: Vec<Vec<ComplianceResult>> =
                join_all(panel.variants.iter().zip(variant_contexts.iter()).map(
                    |(variant, (sp, cache))| {
                        Self::eval_all(
                            corpus,
                            &proposal.raw_output,
                            variant.adapter,
                            &rubric,
                            sp,
                            tau,
                            max_tokens,
                            cache,
                            consensus_passes,
                            threshold,
                            evaluator_timeout_secs,
                        )
                    },
                ))
                .await
                .into_iter()
                .map(|(results, _)| results)
                .collect();

            // For each constraint index, aggregate votes across variants.
            let n_constraints = variant_results.first().map_or(0, std::vec::Vec::len);
            let mut final_results: Vec<ComplianceResult> = Vec::with_capacity(n_constraints);
            let mut uncertain_ids: Vec<String> = Vec::new();
            let mut hard_fail = false;

            for ci in 0..n_constraints {
                // Compute per-variant pass/fail vote using hard_passes() for Hard constraints.
                let mut votes_pass = 0usize;
                let mut votes_fail = 0usize;
                let mut score_sum = 0.0f64;

                for vr in &variant_results {
                    let r = &vr[ci];
                    score_sum += r.score;
                    if r.hard_passes_scaled(ct_scale) {
                        votes_pass += 1;
                    } else {
                        votes_fail += 1;
                    }
                }

                let avg_score = score_sum / variant_results.len() as f64;

                // Use severity + remediation_hint from the first variant (consistent across all).
                let ref_result = &variant_results[0][ci];
                let verdict = aggregate_votes(
                    votes_pass,
                    votes_fail,
                    &panel.diversity_kind,
                    cfg.quorum_fraction,
                );

                let final_score = match &verdict {
                    ConstraintVerdict::Pass => avg_score,
                    ConstraintVerdict::Fail => {
                        // Hard Fail: score set below Hard threshold to guarantee hard_passes() = false.
                        match &ref_result.severity {
                            ConstraintSeverity::Hard { threshold: ht } => (ht - 0.01).max(0.0),
                            _ => 0.0,
                        }
                    }
                    ConstraintVerdict::Uncertain { .. } => {
                        // Apply uncertainty weight; track as uncertain.
                        uncertain_ids.push(ref_result.constraint_id.clone());
                        avg_score * uncertainty_weight
                    }
                };

                // Track whether this constraint is a hard non-uncertain fail.
                let is_hard_fail = matches!(verdict, ConstraintVerdict::Fail)
                    && matches!(ref_result.severity, ConstraintSeverity::Hard { .. });
                if is_hard_fail {
                    hard_fail = true;
                }

                final_results.push(ComplianceResult {
                    constraint_id: ref_result.constraint_id.clone(),
                    score: final_score,
                    severity: ref_result.severity.clone(),
                    remediation_hint: ref_result.remediation_hint.clone(),
                    constraint_description: ref_result.constraint_description.clone(),
                    verifier_reason: ref_result.verifier_reason.clone(),
                    check_verdicts: ref_result.check_verdicts.clone(),
                    criteria_pass: ref_result.criteria_pass.clone(),
                });
            }

            if !uncertain_ids.is_empty() {
                uncertain_map.insert(proposal.explorer_id.clone(), uncertain_ids);
            }

            // Route proposal: hard non-uncertain Fail → failed; otherwise apply threshold check.
            let soft_score = aggregate_compliance_score(&final_results);
            let hard_gate =
                !hard_fail && final_results.iter().all(|r| r.hard_passes_scaled(ct_scale));
            let overall = if hard_gate { soft_score } else { 0.0 };

            if overall >= threshold {
                passed.push((proposal, final_results, false));
            } else {
                let violations: Vec<ConstraintViolation> = final_results
                    .iter()
                    .filter(|r| !r.hard_passes_scaled(ct_scale) || r.score < threshold)
                    .map(|r| ConstraintViolation {
                        constraint_id: r.constraint_id.clone(),
                        score: r.score,
                        severity_label: severity_label(&r.severity),
                        remediation_hint: r.remediation_hint.clone(),
                        constraint_description: r.constraint_description.clone(),
                        verifier_reason: r.verifier_reason.clone(),
                        check_verdicts: r.check_verdicts.clone(),
                        criteria_pass: r.criteria_pass.clone(),
                    })
                    .collect();
                failed.push((proposal, final_results, violations, false));
            }
        }

        let output = VerificationOutput {
            passed,
            failed,
            comparison_events: vec![],
        };
        (output, uncertain_map)
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
        let threshold = config.threshold;
        let evaluator_timeout_secs = config.evaluator_timeout_secs;

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
                    threshold,
                    evaluator_timeout_secs,
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
        // Used as the hard pass threshold for the rubric fallback (empty corpus).
        // Respects the caller's verify_threshold rather than a hardcoded constant.
        rubric_threshold: f64,
        timeout_secs: u64,
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
            let (score, reason) =
                Self::llm_score_raw(effective_rubric, output, evaluator, sp, tau, max_tokens).await;
            return (
                vec![ComplianceResult {
                    constraint_id: "__rubric__".into(),
                    score,
                    severity: ConstraintSeverity::Hard {
                        threshold: rubric_threshold,
                    },
                    remediation_hint: None,
                    constraint_description: String::new(),
                    verifier_reason: if reason.is_empty() {
                        None
                    } else {
                        Some(reason)
                    },
                    check_verdicts: vec![],
                    criteria_pass: None,
                }],
                false,
            );
        }

        let futs = corpus.iter().map(|doc| {
            let constraint_id = doc.id.clone();
            let constraint_description = doc.description.clone();
            let severity = doc.severity.clone();
            let remediation_hint = doc.remediation_hint.clone();
            let criteria_pass = doc.pass_criteria.clone();
            let predicate = doc.predicate.clone();
            let n_checks = doc.binary_checks.len();
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

                // For constraints with binary checks, use tau=0.0 (greedy/deterministic decoding)
                // to reduce LLM stochasticity — the holistic score is overridden by
                // score_from_verdicts anyway, but the reason text (check verdicts) benefits
                // from deterministic output.
                let effective_tau = if n_checks > 0 {
                    h2ai_types::sizing::TauValue::new(0.0).unwrap_or(tau)
                } else {
                    tau
                };

                let (score, verifier_reason, hit) = if let Some(score) = cached_score {
                    tracing::debug!(
                        target: "h2ai.verification.cache",
                        constraint_id = %constraint_id,
                        score,
                        "eval cache hit — reusing score for similar proposal"
                    );
                    (score, None, true)
                } else {
                    let (score, reason) = Self::eval_predicate_async(
                        &predicate,
                        &output,
                        evaluator,
                        &sp,
                        effective_tau,
                        max_tokens,
                        effective_passes,
                        timeout_secs,
                    )
                    .await;
                    cache
                        .entry(constraint_id.clone())
                        .or_default()
                        .push((output.clone(), score));
                    (score, reason, false)
                };

                let check_verdicts = verifier_reason
                    .as_deref()
                    .map(|r| parse_check_verdicts(r, n_checks))
                    .unwrap_or_default();
                let effective_score = score_from_verdicts(&check_verdicts, n_checks, score);
                (
                    ComplianceResult {
                        constraint_id,
                        score: effective_score,
                        severity,
                        remediation_hint,
                        constraint_description,
                        verifier_reason,
                        check_verdicts,
                        criteria_pass,
                    },
                    hit,
                )
            }
        });

        let results: Vec<(ComplianceResult, bool)> = join_all(futs).await;
        let hit_flag = results.iter().any(|(_, h)| *h);
        (results.into_iter().map(|(r, _)| r).collect(), hit_flag)
    }

    /// Evaluate any predicate, including Composite trees that contain `LlmJudge` children.
    /// Returns `(score, reason)` where reason is `Some` only for `LlmJudge` arms.
    /// Uses `Box::pin` for recursive async support.
    ///
    /// For `Composite { And, children }`, static children are evaluated first. If any
    /// returns 0.0 (hard failure — e.g. `NegativeKeyword` found a prohibited term), heavy
    /// children (`LlmJudge`, Oracle) are skipped entirely. This avoids spurious LLM calls
    /// when a proposal already fails on fast deterministic checks.
    #[allow(clippy::too_many_arguments)]
    fn eval_predicate_async<'a>(
        pred: &'a ConstraintPredicate,
        output: &'a str,
        evaluator: &'a dyn IComputeAdapter,
        sp: &'a str,
        tau: h2ai_types::sizing::TauValue,
        max_tokens: u64,
        consensus_passes: u8,
        timeout_secs: u64,
    ) -> Pin<Box<dyn Future<Output = (f64, Option<String>)> + Send + 'a>> {
        Box::pin(async move {
            match pred {
                ConstraintPredicate::LlmJudge { rubric } => {
                    let passes = consensus_passes.max(1) as usize;
                    let mut pairs: Vec<(f64, String)> = Vec::with_capacity(passes);
                    for _ in 0..passes {
                        let pair = if let Ok(pair) = tokio::time::timeout(
                            std::time::Duration::from_secs(timeout_secs),
                            Self::llm_score_raw(rubric, output, evaluator, sp, tau, max_tokens),
                        )
                        .await
                        {
                            pair
                        } else {
                            tracing::warn!(
                                target: "h2ai.verification",
                                timeout_secs,
                                "LlmJudge timed out; skipping — score defaults to 0.5"
                            );
                            (0.5, String::new())
                        };
                        pairs.push(pair);
                    }
                    let avg = pairs.iter().map(|(s, _)| s).sum::<f64>() / pairs.len() as f64;
                    // Use the reason from the lowest-scoring pass — most specific failure diagnosis.
                    let reason = pairs
                        .into_iter()
                        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(_, r)| r)
                        .filter(|r| !r.is_empty());
                    (avg, reason)
                }
                ConstraintPredicate::OracleExecution {
                    test_runner_uri,
                    test_suite,
                    timeout_secs,
                } => (
                    Self::eval_oracle(test_runner_uri, test_suite, *timeout_secs, output).await,
                    None,
                ),
                ConstraintPredicate::SemanticOrdering {
                    first,
                    then,
                    passes,
                } => (
                    Self::majority_binary_check(
                        &format!(
                            "Does the following response demonstrate that '{first}' occurs \
                             BEFORE '{then}'? Answer with exactly one word: YES or NO.\n\n{output}"
                        ),
                        evaluator,
                        sp,
                        tau,
                        *passes,
                        false,
                    )
                    .await,
                    None,
                ),
                ConstraintPredicate::SemanticPresence { concept, passes } => (
                    Self::majority_binary_check(
                        &format!(
                            "Does the following response include or demonstrate '{concept}'? \
                             Answer with exactly one word: YES or NO.\n\n{output}"
                        ),
                        evaluator,
                        sp,
                        tau,
                        *passes,
                        false,
                    )
                    .await,
                    None,
                ),
                ConstraintPredicate::SemanticExclusion { pattern, passes } => (
                    Self::majority_binary_check(
                        &format!(
                            "Does the following response contain '{pattern}'? \
                             Answer with exactly one word: YES or NO.\n\n{output}"
                        ),
                        evaluator,
                        sp,
                        tau,
                        *passes,
                        true,
                    )
                    .await,
                    None,
                ),
                ConstraintPredicate::Composite { op, children } => {
                    match op {
                        CompositeOp::And => {
                            // Evaluate static children first; short-circuit if any hits 0.0.
                            let mut min_score = 1.0_f64;
                            let mut min_reason: Option<String> = None;
                            let mut deferred = Vec::new();
                            for child in children {
                                match child {
                                    ConstraintPredicate::LlmJudge { .. }
                                    | ConstraintPredicate::OracleExecution { .. }
                                    | ConstraintPredicate::SemanticPresence { .. }
                                    | ConstraintPredicate::SemanticOrdering { .. }
                                    | ConstraintPredicate::SemanticExclusion { .. } => {
                                        deferred.push(child);
                                    }
                                    other => {
                                        let s = eval_sync(other, output);
                                        if s < min_score {
                                            min_score = s;
                                            min_reason = None; // static predicates produce no reason
                                        }
                                        if min_score <= 0.0 {
                                            return (0.0, None); // hard failure on static check
                                        }
                                    }
                                }
                            }
                            // Only call LlmJudge if static predicates all passed.
                            for child in deferred {
                                let (s, r) = Self::eval_predicate_async(
                                    child,
                                    output,
                                    evaluator,
                                    sp,
                                    tau,
                                    max_tokens,
                                    consensus_passes,
                                    timeout_secs,
                                )
                                .await;
                                if s < min_score {
                                    min_score = s;
                                    min_reason = r;
                                }
                                if min_score <= 0.0 {
                                    return (0.0, min_reason);
                                }
                            }
                            (min_score, min_reason)
                        }
                        CompositeOp::Or => {
                            let mut max_score = 0.0_f64;
                            for child in children {
                                let (s, _) = Self::eval_predicate_async(
                                    child,
                                    output,
                                    evaluator,
                                    sp,
                                    tau,
                                    max_tokens,
                                    consensus_passes,
                                    timeout_secs,
                                )
                                .await;
                                max_score = max_score.max(s);
                                if max_score >= 1.0 {
                                    return (1.0, None);
                                }
                            }
                            (max_score, None)
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
                                    timeout_secs,
                                )
                                .await
                                .0
                            } else {
                                0.0
                            };
                            (1.0 - s, None)
                        }
                    }
                }
                other => (eval_sync(other, output), None),
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
    ) -> (f64, String) {
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
                    (s.score.clamp(0.0, 1.0), s.reason)
                }
                // JSON parse failure: model did not emit a score object.
                // Fall back to neutral (0.7) so static predicates remain the actual gate.
                None => {
                    tracing::info!(
                        target: "h2ai.verification",
                        raw = %resp.output,
                        "LlmJudge response did not contain JSON score object; using neutral 0.7"
                    );
                    (0.7, String::new())
                }
            },
            Err(e) => {
                tracing::warn!(target: "h2ai.verification", error = %e, "LlmJudge execute error; using neutral 0.7");
                (0.7, String::new())
            }
        }
    }

    /// Majority-vote binary check. Calls the evaluator `passes` times with a YES/NO prompt.
    /// Returns 1.0 if strictly more than half answer YES, 0.0 otherwise (conservative: tie → fail).
    /// If `invert` is true, YES means the pattern was FOUND → returns 0.0 (used for exclusion gates).
    ///
    /// Uses a neutral binary-classifier system prompt instead of the adversarial evaluator prompt.
    /// The adversarial framing (hostile reviewer → find failures) is wrong for factual presence
    /// checks and causes the model to answer NO regardless of content.
    async fn majority_binary_check(
        prompt: &str,
        evaluator: &dyn IComputeAdapter,
        _sp: &str,
        tau: h2ai_types::sizing::TauValue,
        passes: u8,
        invert: bool,
    ) -> f64 {
        let passes = passes.max(1) as usize;
        let mut yes_count = 0usize;
        for _ in 0..passes {
            let req = ComputeRequest {
                system_context: BINARY_CLASSIFIER_SYSTEM_PROMPT.to_owned(),
                task: prompt.to_owned(),
                tau,
                max_tokens: 16,
            };
            let is_yes = match evaluator.execute(req).await {
                Ok(resp) => resp.output.trim().to_uppercase().starts_with("YES"),
                Err(e) => {
                    tracing::warn!(
                        target: "h2ai.verification",
                        error = %e,
                        "binary predicate call failed; counting as NO (conservative)"
                    );
                    false
                }
            };
            if is_yes {
                yes_count += 1;
            }
        }
        // Strict majority: tie → fail (conservative for structural constraints)
        let raw = if yes_count * 2 > passes {
            1.0_f64
        } else {
            0.0_f64
        };
        if invert {
            1.0 - raw
        } else {
            raw
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
    if let Ok(v) = serde_json::from_str::<T>(text.trim()) {
        return Some(v);
    }
    // Walk every `{...}` span; keep the LAST one that deserialises successfully.
    // Reasoning models (DeepSeek-R1, Qwen3) embed intermediate JSON objects in their
    // chain-of-thought — we want the final conclusion, not an intermediate step.
    let mut last_valid: Option<T> = None;
    let mut search = text;
    while let Some(rel) = search.find('{') {
        let tail = &search[rel..];
        let mut stream = serde_json::Deserializer::from_str(tail).into_iter::<serde_json::Value>();
        match stream.next() {
            Some(Ok(_)) => {
                let end = stream.byte_offset();
                if let Ok(v) = serde_json::from_str::<T>(&tail[..end]) {
                    last_valid = Some(v);
                }
                search = &tail[end..];
            }
            _ => search = &tail[1..],
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

/// Parse per-check PRESENT/MISSING verdicts from a LlmJudge CoT reason string.
///
/// The EVALUATOR_SYSTEM_PROMPT instructs the model to emit lines of the form:
///   `CHECK N: <text> → PRESENT`  or  `CHECK N: <text> → MISSING`
///
/// Returns a `Vec<bool>` of length `n_checks` where index `i` corresponds to CHECK `i+1`.
/// - `true`  = CHECK was PRESENT (passed)
/// - `false` = CHECK was MISSING (failed) or not found in the reason (conservative default)
///
/// When `n_checks == 0`, returns an empty vec.
#[must_use]
pub fn parse_check_verdicts(reason: &str, n_checks: usize) -> Vec<bool> {
    if n_checks == 0 {
        return vec![];
    }
    let mut verdicts = vec![false; n_checks];
    // Models emit check verdicts in two formats:
    //   A) "CHECK N: PRESENT (explanation)"  or  "CHECK N: MISSING (explanation)"
    //   B) "CHECK N: explanation → PRESENT"  or  "CHECK N: explanation → MISSING"
    // Both may appear comma-separated on a single line or as separate lines.
    // Split on "CHECK " to handle both layouts uniformly.
    for segment in reason.split("CHECK ").skip(1) {
        let segment = segment.trim();
        let colon_pos = match segment.find(':') {
            Some(p) => p,
            None => continue,
        };
        let num_str = segment[..colon_pos].trim();
        let check_num: usize = match num_str.parse() {
            Ok(n) if n >= 1 => n,
            _ => continue,
        };
        let idx = check_num - 1;
        if idx >= n_checks {
            continue;
        }
        let after_colon = segment[colon_pos + 1..].trim();
        // Format B: look for → PRESENT / → MISSING
        let verdict_str = if let Some(arrow_pos) = after_colon.rfind('→') {
            after_colon[arrow_pos + '→'.len_utf8()..].trim()
        } else {
            // Format A: PRESENT or MISSING appears at the start of after_colon
            after_colon
        };
        let upper = verdict_str.to_ascii_uppercase();
        if upper.starts_with("PRESENT") {
            verdicts[idx] = true;
        }
        // MISSING (or anything else) keeps the default false — no else needed.
    }
    verdicts
}

#[cfg(test)]
mod check_verdicts_tests {
    use super::{parse_check_verdicts, score_from_verdicts};

    #[test]
    fn score_from_verdicts_computes_fraction() {
        assert_eq!(score_from_verdicts(&[true, false, true, true], 4, 0.5), 0.75);
        assert_eq!(score_from_verdicts(&[false, false], 2, 0.9), 0.0);
        assert_eq!(score_from_verdicts(&[true, true, true], 3, 0.1), 1.0);
    }

    #[test]
    fn score_from_verdicts_falls_back_when_no_checks() {
        assert_eq!(score_from_verdicts(&[], 0, 0.42), 0.42);
        assert_eq!(score_from_verdicts(&[], 3, 0.7), 0.7);
    }

    #[test]
    fn score_from_verdicts_falls_back_on_empty_verdicts_with_nonzero_n() {
        // verdicts empty but n_checks > 0: fallback (parse yielded nothing)
        assert_eq!(score_from_verdicts(&[], 4, 0.6), 0.6);
    }

    #[test]
    fn format_a_present_missing() {
        // Format A: "CHECK N: PRESENT (reason)" / "CHECK N: MISSING (reason)"
        let reason = "CHECK 1: PRESENT (Lua script found)\nCHECK 2: MISSING (no audit log)\nCHECK 3: PRESENT (JWT used)";
        let v = parse_check_verdicts(reason, 3);
        assert_eq!(v, vec![true, false, true]);
    }

    #[test]
    fn format_b_arrow() {
        // Format B: "CHECK N: explanation → PRESENT" / "CHECK N: explanation → MISSING"
        let reason = "CHECK 1: some reason → PRESENT\nCHECK 2: another → MISSING";
        let v = parse_check_verdicts(reason, 2);
        assert_eq!(v, vec![true, false]);
    }

    #[test]
    fn zero_checks_returns_empty() {
        assert!(parse_check_verdicts("CHECK 1: PRESENT", 0).is_empty());
    }

    #[test]
    fn out_of_range_check_number_ignored() {
        let v = parse_check_verdicts("CHECK 5: PRESENT", 3);
        assert_eq!(v, vec![false, false, false]);
    }

    #[test]
    fn missing_defaults_to_false() {
        let v = parse_check_verdicts("CHECK 1: MISSING (not implemented)", 2);
        assert_eq!(v, vec![false, false]);
    }

    #[test]
    fn mixed_formats_same_reason() {
        let reason = "CHECK 1: PRESENT (ok), CHECK 2: description → PRESENT";
        let v = parse_check_verdicts(reason, 2);
        assert_eq!(v, vec![true, true]);
    }
}
