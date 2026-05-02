use crate::parsing::extract_json;
use chrono::Utc;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::AgentRole;
use h2ai_types::identity::{SubtaskId, TaskId};
use h2ai_types::manifest::TaskManifest;
use h2ai_types::physics::TauValue;
use h2ai_types::plan::{PlanStatus, Subtask, SubtaskPlan};
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlannerError {
    #[error("adapter error: {0}")]
    Adapter(String),
    #[error("failed to parse LLM JSON response: {0}")]
    ParseError(String),
    #[error("dependency index {index} out of range (only {len} subtasks defined)")]
    InvalidDependencyIndex { index: usize, len: usize },
}

#[derive(Deserialize)]
struct LlmSubtask {
    description: String,
    #[serde(default)]
    depends_on: Vec<usize>,
    role_hint: Option<String>,
}

#[derive(Deserialize)]
struct LlmDecomposition {
    subtasks: Vec<LlmSubtask>,
}

pub struct PlanningEngine;

impl PlanningEngine {
    pub async fn decompose(
        manifest: &TaskManifest,
        adapter: &dyn IComputeAdapter,
        tau: TauValue,
    ) -> Result<SubtaskPlan, PlannerError> {
        let constraints_csv = manifest.constraints.join(", ");
        let constraints_str = if constraints_csv.is_empty() {
            "none"
        } else {
            &constraints_csv
        };
        let prompt = h2ai_config::prompts::DECOMPOSER_TASK.render(&[
            ("description", &manifest.description),
            ("constraints", constraints_str),
        ]);

        let request = ComputeRequest {
            system_context: h2ai_config::prompts::DECOMPOSER_SYSTEM.as_str().into(),
            task: prompt,
            tau,
            max_tokens: 1024,
        };

        let response = adapter
            .execute(request)
            .await
            .map_err(|e| PlannerError::Adapter(e.to_string()))?;

        let decomposition = parse_decomposition(&response.output)?;
        build_plan(decomposition)
    }
}

fn parse_decomposition(raw: &str) -> Result<LlmDecomposition, PlannerError> {
    let json = extract_json(raw);
    serde_json::from_str(json).map_err(|e| {
        PlannerError::ParseError(format!("{e}\n  extracted: {json}\n  raw:       {raw}"))
    })
}

fn parse_role_hint(s: &Option<String>) -> Option<AgentRole> {
    s.as_deref().and_then(|h| match h {
        "Executor" => Some(AgentRole::Executor),
        "Evaluator" => Some(AgentRole::Evaluator),
        "Synthesizer" => Some(AgentRole::Synthesizer),
        "Coordinator" => Some(AgentRole::Coordinator),
        // `Custom` requires runtime fields (name, tau, cost) that cannot be
        // recovered from a single string; all unrecognised values are ignored.
        _ => None,
    })
}

fn build_plan(decomp: LlmDecomposition) -> Result<SubtaskPlan, PlannerError> {
    let n = decomp.subtasks.len();
    let ids: Vec<SubtaskId> = (0..n).map(|_| SubtaskId::new()).collect();

    let subtasks = decomp
        .subtasks
        .iter()
        .enumerate()
        .map(|(i, raw)| {
            let depends_on = raw
                .depends_on
                .iter()
                .map(|&idx| {
                    if idx >= n {
                        Err(PlannerError::InvalidDependencyIndex { index: idx, len: n })
                    } else {
                        Ok(ids[idx].clone())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(Subtask {
                id: ids[i].clone(),
                description: raw.description.clone(),
                depends_on,
                role_hint: parse_role_hint(&raw.role_hint),
            })
        })
        .collect::<Result<Vec<_>, PlannerError>>()?;

    Ok(SubtaskPlan {
        plan_id: TaskId::new(),
        // Placeholder — overridden by CompoundTaskEngine::run which sets it to
        // CompoundTaskInput.task_id after decompose() returns.
        parent_task_id: TaskId::new(),
        subtasks,
        status: PlanStatus::PendingReview,
        created_at: Utc::now(),
    })
}
