use crate::events::OracleGateResultEvent;
use crate::manifest::CotStyle;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    Fast,
    #[default]
    Standard,
    Capable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeSpec {
    pub name: String,
    pub persona: String,
    pub scope: String,
    /// Self-reported confidence 0.0–1.0; used as synthesis weight.
    pub confidence: f64,
    pub tau: f64,
    pub model_tier: ModelTier,
    pub cot_style: CotStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeOutput {
    pub archetype: ArchetypeSpec,
    pub problem_analysis: String,
    pub solution_sketch: String,
    /// Self-reported confidence from brainstorm output; defaults to archetype.confidence.
    pub confidence: f64,
    /// Oracle gate result for this archetype's output. `None` when no oracle gate ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oracle_result: Option<OracleGateResultEvent>,
}

/// Synthesis output produced each iteration; carries forward (only shared_understanding +
/// tensions forwarded to next archetype selection — Think Twice: discard intermediates).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThinkingReport {
    pub shared_understanding: String,
    pub tensions: Vec<String>,
    pub coverage_score: f64,
    pub iteration: u32,
    /// Cosine similarity to previous iteration's shared_understanding. 0.0 on first iteration.
    pub prev_similarity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_tier_default_is_standard() {
        assert_eq!(ModelTier::default(), ModelTier::Standard);
    }

    #[test]
    fn thinking_report_default_has_zero_coverage() {
        let r = ThinkingReport::default();
        assert_eq!(r.coverage_score, 0.0);
        assert!(r.tensions.is_empty());
        assert!(r.shared_understanding.is_empty());
    }
}
