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

/// Synthesis output produced each iteration; carries forward (only `shared_understanding` +
/// tensions forwarded to next archetype selection — Think Twice: discard intermediates).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThinkingReport {
    pub shared_understanding: String,
    pub tensions: Vec<String>,
    pub coverage_score: f64,
    pub iteration: u32,
    /// Cosine similarity to previous iteration's `shared_understanding`. 0.0 on first iteration.
    pub prev_similarity: f64,
    /// IDs of all KnowledgeNodes retrieved across all thinking loop iterations (deduplicated).
    /// Used by post_run to apply retrieval violation penalties.
    #[serde(default)]
    pub retrieved_node_ids: Vec<String>,
    /// Count of unique Synthetic (skill) nodes that appeared in retrieved_node_ids.
    /// Emitted in TaskAttributionEvent for contrastive offline analysis.
    #[serde(default)]
    pub skill_nodes_used: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_report_new_fields_default_to_zero_empty() {
        let report = ThinkingReport::default();
        assert!(report.retrieved_node_ids.is_empty());
        assert_eq!(report.skill_nodes_used, 0);
    }

    #[test]
    fn thinking_report_backwards_compat_deserializes_without_new_fields() {
        // Old JSON without retrieved_node_ids / skill_nodes_used
        let json = r#"{"shared_understanding":"test","tensions":[],"coverage_score":0.8,"iteration":1,"prev_similarity":0.5}"#;
        let report: ThinkingReport = serde_json::from_str(json).unwrap();
        assert!(report.retrieved_node_ids.is_empty());
        assert_eq!(report.skill_nodes_used, 0);
    }

    #[test]
    fn thinking_report_roundtrip_preserves_new_fields() {
        let mut report = ThinkingReport::default();
        report.retrieved_node_ids = vec!["node-1".to_string(), "node-2".to_string()];
        report.skill_nodes_used = 2;
        let json = serde_json::to_string(&report).unwrap();
        let restored: ThinkingReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.retrieved_node_ids, vec!["node-1", "node-2"]);
        assert_eq!(restored.skill_nodes_used, 2);
    }
}
