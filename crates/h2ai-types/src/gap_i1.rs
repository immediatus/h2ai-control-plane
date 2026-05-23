use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGapRecord {
    pub constraint_id: String,
    pub check_idx: usize,
    pub incorrect_concept: String,
    pub gap_query: String,
    pub pass_rate_across_waves: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSynthesis {
    pub check_id: (String, usize),
    pub incorrect_pattern: String,
    pub correct_pattern: String,
    pub mechanistic_reason: String,
    pub source: Option<String>,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knowledge_gap_record_fields_accessible() {
        let rec = KnowledgeGapRecord {
            constraint_id: "CONSTRAINT-008".to_string(),
            check_idx: 1,
            incorrect_concept: "SETNX as standalone idempotency primitive".to_string(),
            gap_query: "Redis atomic CAS quota update Lua EVAL without distributed locks"
                .to_string(),
            pass_rate_across_waves: 0.0,
        };
        assert_eq!(rec.constraint_id, "CONSTRAINT-008");
        assert_eq!(rec.check_idx, 1);
        assert_eq!(rec.pass_rate_across_waves, 0.0);
    }

    #[test]
    fn domain_synthesis_fields_accessible() {
        let synth = DomainSynthesis {
            check_id: ("CONSTRAINT-008".to_string(), 1),
            incorrect_pattern: "SETNX as standalone idempotency primitive".to_string(),
            correct_pattern: "SET key val NX EX ttl inside Lua EVAL".to_string(),
            mechanistic_reason: "Lua EVAL is atomic; SETNX alone does not protect against concurrent updates outside the script".to_string(),
            source: Some("https://redis.io/docs/manual/programmability/lua-api/".to_string()),
            confidence: 0.85,
        };
        assert_eq!(synth.check_id.0, "CONSTRAINT-008");
        assert!(synth.confidence > 0.7);
        assert!(synth.source.is_some());
    }

    #[test]
    fn domain_synthesis_low_confidence_detectable() {
        let synth = DomainSynthesis {
            check_id: ("CONSTRAINT-TAU-2".to_string(), 0),
            incorrect_pattern: "async cache invalidation is sufficient".to_string(),
            correct_pattern: "push-based invalidation via Redis Stream with bounded TTL"
                .to_string(),
            mechanistic_reason: "Stream ensures ≤TTL convergence; async pub/sub can be lost"
                .to_string(),
            source: None,
            confidence: 0.5,
        };
        assert!(
            synth.confidence < 0.7,
            "low confidence should be filterable"
        );
    }
}
