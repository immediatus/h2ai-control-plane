use h2ai_orchestrator::mape_k::build_best_passing_pin_hint;

// build_best_passing_pin_hint(constraint_id, dynamic_reasons, corpus_hint) -> Option<String>
// Returns the dynamic reason if available, else the corpus hint.

#[test]
fn dynamic_reason_takes_priority_over_corpus_hint() {
    let mut dynamic: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    dynamic.insert(
        "C-TAU-2".to_string(),
        "verifier saw audit trail at /events endpoint with actor+timestamp".to_string(),
    );

    let result = build_best_passing_pin_hint(
        "C-TAU-2",
        &dynamic,
        Some("pass_criteria from corpus".to_string()),
    );
    assert_eq!(
        result.as_deref(),
        Some("verifier saw audit trail at /events endpoint with actor+timestamp"),
        "dynamic reason must take priority over corpus hint"
    );
}

#[test]
fn corpus_hint_used_when_dynamic_reason_absent() {
    let dynamic: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    let result = build_best_passing_pin_hint(
        "C-004",
        &dynamic,
        Some("must use idempotency key".to_string()),
    );
    assert_eq!(
        result.as_deref(),
        Some("must use idempotency key"),
        "corpus hint must be used when no dynamic reason is available"
    );
}

#[test]
fn none_returned_when_both_absent() {
    let dynamic: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    let result = build_best_passing_pin_hint("C-005", &dynamic, None);
    assert!(
        result.is_none(),
        "None expected when both dynamic reason and corpus hint are absent"
    );
}
