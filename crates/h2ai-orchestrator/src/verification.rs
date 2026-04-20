use futures::future::join_all;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::VerificationConfig;
use h2ai_types::events::ProposalEvent;
use serde::Deserialize;

pub struct VerificationInput<'a> {
    pub proposals: Vec<ProposalEvent>,
    pub constraints: &'a [String],
    pub evaluator: &'a dyn IComputeAdapter,
    pub config: VerificationConfig,
}

pub struct VerificationOutput {
    pub passed: Vec<(ProposalEvent, f64)>,
    pub failed: Vec<(ProposalEvent, f64, String)>,
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
        let rubric = input.config.rubric.clone();
        let constraints_str = input.constraints.join(", ");
        let threshold = input.config.threshold;
        let evaluator_system_prompt = input.config.evaluator_system_prompt.clone();
        let evaluator_tau = input.config.evaluator_tau;
        let evaluator_max_tokens = input.config.evaluator_max_tokens;

        let futures = input.proposals.into_iter().map(|proposal| {
            let prompt = format!(
                "{}\n\nConstraints: {}\n\nProposal:\n{}",
                rubric,
                constraints_str,
                proposal.raw_output
            );
            let req = ComputeRequest {
                system_context: evaluator_system_prompt.clone(),
                task: prompt,
                tau: evaluator_tau,
                max_tokens: evaluator_max_tokens,
            };
            async move {
                let (score, reason) = match evaluator.execute(req).await {
                    Ok(resp) => match serde_json::from_str::<ScoreResponse>(&resp.output) {
                        Ok(s) => (s.score.clamp(0.0, 1.0), s.reason),
                        Err(_) => (0.5, "parse error — neutral score".into()),
                    },
                    Err(e) => (0.5, format!("evaluator error: {e}")),
                };
                let pass = score >= threshold;
                (proposal, score, reason, pass)
            }
        });

        let results = join_all(futures).await;

        let mut passed = Vec::new();
        let mut failed = Vec::new();
        for (proposal, score, reason, pass) in results {
            if pass {
                passed.push((proposal, score));
            } else {
                failed.push((proposal, score, reason));
            }
        }

        VerificationOutput { passed, failed }
    }
}
