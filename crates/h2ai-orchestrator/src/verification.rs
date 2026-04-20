use futures::future::join_all;
use h2ai_constraints::eval::eval_sync;
use h2ai_constraints::types::{
    aggregate_compliance_score, ComplianceResult, ConstraintDoc, ConstraintPredicate,
    ConstraintSeverity,
};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::VerificationConfig;
use h2ai_types::events::{ConstraintViolation, ProposalEvent};
use serde::Deserialize;

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
    pub failed: Vec<(ProposalEvent, Vec<ComplianceResult>, Vec<ConstraintViolation>)>,
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
                let results =
                    Self::eval_all(corpus, &proposal.raw_output, evaluator, &rubric, &sp, tau, max_tokens)
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

    async fn eval_all(
        corpus: &[ConstraintDoc],
        output: &str,
        evaluator: &dyn IComputeAdapter,
        rubric: &str,
        sp: &str,
        tau: h2ai_types::physics::TauValue,
        max_tokens: u64,
    ) -> Vec<ComplianceResult> {
        // If corpus is empty, fall back to a single LLM-scored result using the rubric.
        if corpus.is_empty() {
            let score = Self::llm_score_raw(rubric, output, evaluator, sp, tau, max_tokens).await;
            return vec![ComplianceResult {
                constraint_id: "__rubric__".into(),
                score,
                severity: ConstraintSeverity::Hard { threshold: 0.45 },
                remediation_hint: None,
            }];
        }

        let futs = corpus.iter().map(|doc| async move {
            let score = if matches!(doc.predicate, ConstraintPredicate::LlmJudge { .. }) {
                let judge_rubric = match &doc.predicate {
                    ConstraintPredicate::LlmJudge { rubric: r } => r.clone(),
                    _ => unreachable!(),
                };
                Self::llm_score_raw(&judge_rubric, output, evaluator, sp, tau, max_tokens).await
            } else {
                eval_sync(&doc.predicate, output)
            };
            ComplianceResult {
                constraint_id: doc.id.clone(),
                score,
                severity: doc.severity.clone(),
                remediation_hint: doc.remediation_hint.clone(),
            }
        });
        join_all(futs).await
    }

    async fn llm_score_raw(
        rubric: &str,
        output: &str,
        evaluator: &dyn IComputeAdapter,
        sp: &str,
        tau: h2ai_types::physics::TauValue,
        max_tokens: u64,
    ) -> f64 {
        let prompt = format!("{rubric}\n\nProposal:\n{output}");
        let req = ComputeRequest {
            system_context: sp.to_owned(),
            task: prompt,
            tau,
            max_tokens,
        };
        match evaluator.execute(req).await {
            Ok(resp) => match serde_json::from_str::<ScoreResponse>(&resp.output) {
                Ok(s) => s.score.clamp(0.0, 1.0),
                Err(_) => 0.0,
            },
            Err(_) => 0.0,
        }
    }
}

fn severity_label(s: &ConstraintSeverity) -> String {
    match s {
        ConstraintSeverity::Hard { .. } => "Hard".into(),
        ConstraintSeverity::Soft { .. } => "Soft".into(),
        ConstraintSeverity::Advisory => "Advisory".into(),
    }
}
