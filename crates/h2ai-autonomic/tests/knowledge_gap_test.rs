use h2ai_autonomic::knowledge_gap::{build_gap_queries, detect_cold_checks};

#[test]
fn detects_cold_check_when_pass_rate_is_zero() {
    let check_rates = vec![
        (("CONSTRAINT-008".to_string(), 0usize), 0.5_f64),
        (("CONSTRAINT-008".to_string(), 1usize), 0.0_f64), // cold
        (("CONSTRAINT-TAU-2".to_string(), 0usize), 0.0_f64), // cold
    ];
    let threshold = 0.0_f64;
    let cold = detect_cold_checks(&check_rates, threshold);
    assert_eq!(cold.len(), 2);
    let ids: Vec<_> = cold
        .iter()
        .map(|r| (&r.constraint_id, r.check_idx))
        .collect();
    assert!(ids.contains(&(&"CONSTRAINT-008".to_string(), 1)));
    assert!(ids.contains(&(&"CONSTRAINT-TAU-2".to_string(), 0)));
}

#[test]
fn does_not_detect_passing_check() {
    let check_rates = vec![(("CONSTRAINT-008".to_string(), 0usize), 1.0_f64)];
    let cold = detect_cold_checks(&check_rates, 0.0);
    assert!(cold.is_empty());
}

#[test]
fn respects_max_records_cap() {
    let check_rates: Vec<_> = (0..10).map(|i| (("C".to_string(), i), 0.0_f64)).collect();
    let cold = detect_cold_checks(&check_rates, 0.0);
    // detect_cold_checks returns all; caller applies max cap
    assert_eq!(cold.len(), 10);
}

#[test]
fn build_gap_queries_returns_three_queries() {
    let queries = build_gap_queries(
        "Does the proposed design use Redis Lua EVAL for atomic quota updates?",
        "SETNX as standalone idempotency primitive outside Lua EVAL",
    );
    assert_eq!(
        queries.len(),
        3,
        "must return canonical, failure-mode, migration queries"
    );
    for q in &queries {
        assert!(!q.is_empty());
    }
}

#[test]
fn build_gap_queries_contains_incorrect_concept_context() {
    let queries = build_gap_queries(
        "Does the design use atomic CAS?",
        "WATCH/MULTI/EXEC as distributed lock",
    );
    let joined = queries.join(" ");
    assert!(
        joined.to_lowercase().contains("watch")
            || joined.to_lowercase().contains("multi")
            || joined.to_lowercase().contains("redis"),
        "queries should be domain-specific: {joined}"
    );
}
