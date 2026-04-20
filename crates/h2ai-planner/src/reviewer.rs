use crate::decomposer::PlannerError;
use crate::parsing::extract_json;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::identity::SubtaskId;
use h2ai_types::physics::TauValue;
use h2ai_types::plan::SubtaskPlan;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum ReviewOutcome {
    Approved,
    Rejected { reason: String },
}

pub struct PlanReviewer;

impl PlanReviewer {
    /// Evaluates a `SubtaskPlan` via LLM.
    ///
    /// Structural checks (empty plan, cyclic dependencies) run locally first.
    /// If both pass, makes one LLM call for a semantic review.
    pub async fn evaluate(
        plan: &SubtaskPlan,
        original_description: &str,
        adapter: &dyn IComputeAdapter,
        tau: TauValue,
    ) -> Result<ReviewOutcome, PlannerError> {
        if plan.subtasks.is_empty() {
            return Ok(ReviewOutcome::Rejected {
                reason: "Plan contains no subtasks.".into(),
            });
        }
        if let Some(reason) = detect_cycle(plan) {
            return Ok(ReviewOutcome::Rejected { reason });
        }

        let subtask_summary = plan
            .subtasks
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let deps = if s.depends_on.is_empty() {
                    "none".into()
                } else {
                    s.depends_on
                        .iter()
                        .map(|id| {
                            plan.subtasks
                                .iter()
                                .position(|t| &t.id == id)
                                .map_or_else(|| "<unknown>".into(), |idx| idx.to_string())
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                format!("  {i}. {desc} (depends on: {deps})", desc = s.description)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are reviewing a subtask decomposition plan.\n\
             \n\
             Original task: {original_description}\n\
             \n\
             Proposed plan:\n{subtask_summary}\n\
             \n\
             Evaluate:\n\
             1. Does this plan fully address the original task with no obvious missing steps?\n\
             2. Is the dependency order logical?\n\
             \n\
             Respond ONLY with valid JSON:\n\
             {{\"approved\": true, \"reason\": \"...\"}}"
        );

        let response = adapter
            .execute(ComputeRequest {
                system_context: "You are a critical plan reviewer. Respond only with valid JSON."
                    .into(),
                task: prompt,
                tau,
                max_tokens: 256,
            })
            .await
            .map_err(|e| PlannerError::Adapter(e.to_string()))?;

        parse_review(&response.output)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct LlmReview {
    approved: bool,
    reason: String,
}

fn parse_review(raw: &str) -> Result<ReviewOutcome, PlannerError> {
    let json = extract_json(raw);

    let review: LlmReview = serde_json::from_str(json)
        .map_err(|e| PlannerError::ParseError(format!("review JSON: {e}\n  raw: {raw}")))?;

    if review.approved {
        Ok(ReviewOutcome::Approved)
    } else {
        Ok(ReviewOutcome::Rejected {
            reason: review.reason,
        })
    }
}

/// Returns `Some(reason)` if a cycle is detected via DFS colour-marking.
fn detect_cycle(plan: &SubtaskPlan) -> Option<String> {
    let index: HashMap<&SubtaskId, usize> = plan
        .subtasks
        .iter()
        .enumerate()
        .map(|(i, s)| (&s.id, i))
        .collect();

    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Gray,
        Black,
    }

    let n = plan.subtasks.len();
    let mut color = vec![Color::White; n];

    fn dfs(
        v: usize,
        plan: &SubtaskPlan,
        index: &HashMap<&SubtaskId, usize>,
        color: &mut Vec<Color>,
    ) -> bool {
        color[v] = Color::Gray;
        for dep_id in &plan.subtasks[v].depends_on {
            if let Some(&u) = index.get(dep_id) {
                if color[u] == Color::Gray
                    || (color[u] == Color::White && dfs(u, plan, index, color))
                {
                    return true;
                }
            }
        }
        color[v] = Color::Black;
        false
    }

    for i in 0..n {
        if color[i] == Color::White && dfs(i, plan, &index, &mut color) {
            return Some("Cyclic dependency detected in subtask plan.".into());
        }
    }
    None
}
