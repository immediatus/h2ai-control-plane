use crate::engine::EngineInput;
use crate::phases::{ExitReason, StepResult};
use h2ai_types::adapter::ComputeRequest;
use h2ai_types::events::{CorrelatedEnsembleWarning, ResearcherGroundingEvent};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::TauValue;

use super::generation::Output as GenerationOutput;

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub task_id: &'a TaskId,
    pub retry_count: u32,
    pub system_context: &'a str,
    pub system_context_with_rubric: &'a str,
}

/// Output when no hallucination is detected — wraps the generation output.
pub struct Output {
    pub generation: GenerationOutput,
}

/// Run the Correlated Hallucination Detection phase.
///
/// Checks CV of pairwise Jaccard distances on raw proposal texts. When the ensemble
/// is semantically clustered (low CV + low mean Jaccard distance), a grounding hint
/// is built (via reactive researcher call if available) and
/// `StepResult::EarlyExit(ExitReason::HallucinationDetected { ... })` is returned.
///
/// The engine.rs early-exit handler for `HallucinationDetected` is responsible for:
///   - extending `all_correlated_warnings` with `reason.warning`
///   - extending `all_researcher_grounding_events` with `reason.researcher_grounding_events`
///   - pushing `reason.tau_values` to `tau_values_tried`
///   - setting `retry_context = Some(reason.retry_context_hint)`
///   - calling `apply_optimizer` and then `continue`
///
/// Returns `StepResult::Done(Output { generation })` when no hallucination is detected.
/// Never returns `StepResult::Fatal`.
pub async fn run(generation: GenerationOutput, input: Input<'_>) -> StepResult<Output> {
    let engine_input = input.engine_input;
    let task_id = input.task_id;
    let retry_count = input.retry_count;
    let system_context = input.system_context;
    let system_context_with_rubric = input.system_context_with_rubric;

    let proposals = &generation.proposals;
    let tau_values = generation.tau_values.clone();

    // ── Correlated Hallucination Detection ──────────────────
    // Check CV of pairwise Jaccard distances on raw proposal texts.
    // Low CV = proposals are semantically clustered → retry with grounding hint.
    if engine_input.cfg.correlated_hallucination_cv_threshold > 0.0 && proposals.len() >= 2 {
        let proposal_texts: Vec<&str> = proposals.iter().map(|p| p.raw_output.as_str()).collect();
        if let Some(signal) = crate::correlated_hallucination::compute_cv(&proposal_texts) {
            if signal.cv < engine_input.cfg.correlated_hallucination_cv_threshold
                && signal.mean_jaccard_distance
                    < engine_input.cfg.correlated_hallucination_min_jaccard_floor
                && retry_count < engine_input.cfg.max_autonomic_retries
            {
                let warning = CorrelatedEnsembleWarning {
                    task_id: task_id.clone(),
                    cv: signal.cv,
                    mean_jaccard_distance: signal.mean_jaccard_distance,
                    retry_count,
                };

                // Build grounding: call researcher (reactive path) if available.
                let mut researcher_grounding_events: Vec<ResearcherGroundingEvent> = Vec::new();
                let grounding_hint = if let Some(ref researcher) = engine_input.researcher_adapter {
                    let proposal_summary = proposals
                        .iter()
                        // 300-char cap: debug context only — enough for correlation diagnosis
                        .map(|p| p.raw_output[..p.raw_output.len().min(300)].to_string())
                        .collect::<Vec<_>>()
                        .join("\n---\n");
                    let research_req = ComputeRequest {
                        system_context: system_context.to_owned(),
                        task: format!(
                            "These AI proposals may share a common assumption.\
                                 \nPROPOSALS:\n{proposal_summary}\n\n\
                                 Search for current state-of-the-art evidence that \
                                 contradicts the shared assumption. Return JSON: \
                                 {{\"shared_assumption\": \"...\", \
                                   \"literature_summary\": \"...\", \
                                   \"grounding_statement\": \"...\"}}",
                        ),
                        tau: TauValue::new(0.3).unwrap(),
                        max_tokens: engine_input.cfg.hallucination_check_max_tokens,
                    };
                    match researcher.execute(research_req).await {
                        Ok(resp) => {
                            #[derive(serde::Deserialize)]
                            struct ResearchResult {
                                shared_assumption: String,
                                literature_summary: String,
                                grounding_statement: String,
                            }
                            crate::verification::extract_json_object::<ResearchResult>(&resp.output)
                                .map(|r| {
                                    researcher_grounding_events.push(ResearcherGroundingEvent {
                                        task_id: task_id.clone(),
                                        shared_assumption: r.shared_assumption,
                                        literature_summary: r.literature_summary.clone(),
                                        slot: None,
                                        source: h2ai_types::events::GroundingSource::LlmResearcher,
                                    });
                                    format!(
                                        "[EXTERNAL GROUNDING]: {}\n\
                                         Find the assumption all current proposals share \
                                         and propose a solution that contradicts it.",
                                        r.grounding_statement
                                    )
                                })
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };

                let hint = grounding_hint.unwrap_or_else(|| {
                    "Find the assumption all current proposals share that might be wrong. \
                     Propose a solution that directly contradicts it."
                        .to_string()
                });

                let retry_context_hint = format!(
                    "{system_context_with_rubric}\n\n\
                     --- CORRELATED ENSEMBLE DETECTED (iteration {retry_count}) ---\n\
                     {hint}\n\
                     ---"
                );

                return StepResult::EarlyExit(ExitReason::HallucinationDetected {
                    retry_context_hint,
                    tau_values,
                    warning,
                    researcher_grounding_events,
                });
            }
        }
    }

    StepResult::Done(Output { generation })
}
