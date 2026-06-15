use crate::sizing::TauValue;
use serde::{Deserialize, Serialize};

/// A single step in a chained request: a prompt template, creativity temperature, and token budget.
///
/// Each `ChainStep` represents one prompt-filling LLM call, parameterized by:
/// - `template`: The prompt string with fill-in-the-blank placeholders.
/// - `tau`: Creativity temperature ∈ [0, 1], controlling output diversity.
/// - `max_tokens`: Maximum tokens to generate for this step's response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStep {
    pub template: String,
    pub tau: TauValue,
    pub max_tokens: u64,
}

/// A linear chain of LLM calls.
///
/// `ChainedRequest` sequences multiple prompt-filling steps: the system context is established
/// once, then each `ChainStep` is executed in order, with the output of one step feeding
/// into the template variables of the next.
///
/// Used by `execute_chain` (Task 2) and `tournament_merge` (Task 3) to implement
/// orchestrated linear-inference pipelines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainedRequest {
    pub initial_system_context: String,
    pub steps: Vec<ChainStep>,
}
