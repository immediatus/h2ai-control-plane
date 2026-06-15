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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_step_roundtrip() {
        let step = ChainStep {
            template: "fill this".into(),
            tau: TauValue::new(0.3).unwrap(),
            max_tokens: 512,
        };
        let json = serde_json::to_string(&step).unwrap();
        let restored: ChainStep = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.template, "fill this");
        assert_eq!(restored.max_tokens, 512);
    }

    #[test]
    fn chained_request_roundtrip() {
        let req = ChainedRequest {
            initial_system_context: "sys".into(),
            steps: vec![
                ChainStep {
                    template: "t1".into(),
                    tau: TauValue::new(0.2).unwrap(),
                    max_tokens: 256,
                },
                ChainStep {
                    template: "t2".into(),
                    tau: TauValue::new(0.5).unwrap(),
                    max_tokens: 128,
                },
            ],
        };
        let json = serde_json::to_string(&req).unwrap();
        let restored: ChainedRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.steps.len(), 2);
        assert_eq!(restored.steps[0].template, "t1");
        assert_eq!(restored.steps[1].max_tokens, 128);
    }
}
