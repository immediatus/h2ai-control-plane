use h2ai_types::gap_i1::{DomainSynthesis, KnowledgeGapRecord};

// ── injection-tracking tests ──────────────────────────────────────────────────

#[test]
fn domain_synthesis_new_fields_default_to_none_and_empty() {
    let ds = DomainSynthesis {
        check_id: ("C-001".to_string(), 0),
        incorrect_pattern: "wrong".to_string(),
        correct_pattern: "right".to_string(),
        mechanistic_reason: "because".to_string(),
        source: None,
        confidence: 0.8,
        injected_at_wave: None,
        pre_injection_pass_rate: None,
        post_injection_pass_rates: vec![],
    };
    assert!(ds.injected_at_wave.is_none());
    assert!(ds.pre_injection_pass_rate.is_none());
    assert!(ds.post_injection_pass_rates.is_empty());
}

#[test]
fn domain_synthesis_with_injection_tracking() {
    let ds = DomainSynthesis {
        check_id: ("C-008".to_string(), 2),
        incorrect_pattern: "old".to_string(),
        correct_pattern: "new".to_string(),
        mechanistic_reason: "reason".to_string(),
        source: None,
        confidence: 0.5,
        injected_at_wave: Some(2),
        pre_injection_pass_rate: Some(0.0),
        post_injection_pass_rates: vec![0.0, 0.0],
    };
    assert_eq!(ds.injected_at_wave, Some(2));
    assert_eq!(ds.pre_injection_pass_rate, Some(0.0));
    assert_eq!(ds.post_injection_pass_rates.len(), 2);
}

#[test]
fn domain_synthesis_json_roundtrip_with_new_fields() {
    let ds = DomainSynthesis {
        check_id: ("C-008".to_string(), 1),
        incorrect_pattern: "bad".to_string(),
        correct_pattern: "good".to_string(),
        mechanistic_reason: "reason".to_string(),
        source: Some("test".to_string()),
        confidence: 0.9,
        injected_at_wave: Some(3),
        pre_injection_pass_rate: Some(0.25),
        post_injection_pass_rates: vec![0.25, 0.5],
    };
    let json = serde_json::to_string(&ds).unwrap();
    let back: DomainSynthesis = serde_json::from_str(&json).unwrap();
    assert_eq!(back.injected_at_wave, Some(3));
    assert_eq!(back.pre_injection_pass_rate, Some(0.25));
    assert_eq!(back.post_injection_pass_rates, vec![0.25, 0.5]);
}

#[test]
fn domain_synthesis_old_json_without_new_fields_deserializes_with_defaults() {
    // Existing serialized DomainSynthesis without the new fields must still parse
    let json = r#"{"check_id":["C-001",0],"incorrect_pattern":"wrong","correct_pattern":"right","mechanistic_reason":"because","source":null,"confidence":0.8}"#;
    let ds: DomainSynthesis = serde_json::from_str(json).unwrap();
    assert!(ds.injected_at_wave.is_none());
    assert!(ds.pre_injection_pass_rate.is_none());
    assert!(ds.post_injection_pass_rates.is_empty());
}

#[test]
fn knowledge_gap_record_fields_accessible() {
    let rec = KnowledgeGapRecord {
        constraint_id: "CONSTRAINT-008".to_string(),
        check_idx: 1,
        incorrect_concept: "SETNX as standalone idempotency primitive".to_string(),
        gap_query: "Redis atomic CAS quota update Lua EVAL without distributed locks".to_string(),
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
        injected_at_wave: None,
        pre_injection_pass_rate: None,
        post_injection_pass_rates: vec![],
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
        correct_pattern: "push-based invalidation via Redis Stream with bounded TTL".to_string(),
        mechanistic_reason: "Stream ensures ≤TTL convergence; async pub/sub can be lost"
            .to_string(),
        source: None,
        confidence: 0.5,
        injected_at_wave: None,
        pre_injection_pass_rate: None,
        post_injection_pass_rates: vec![],
    };
    assert!(
        synth.confidence < 0.7,
        "low confidence should be filterable"
    );
}
