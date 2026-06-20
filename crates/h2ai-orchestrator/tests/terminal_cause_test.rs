use h2ai_types::events::TerminalCause;

#[test]
fn top_violated_constraints_sorted_capped_at_5() {
    let mut freq: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (id, count) in [
        ("C-005", 4u32),
        ("C-004", 2),
        ("C-008", 3),
        ("C-TAU-1", 1),
        ("C-001", 2),
        ("C-002", 5),
    ] {
        freq.insert(id.to_string(), count);
    }
    let mut sorted: Vec<(String, u32)> = freq.into_iter().collect();
    sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
    sorted.truncate(5);
    assert_eq!(sorted.len(), 5);
    assert_eq!(sorted[0].1, 5); // highest count first
    let top_ids: Vec<&str> = sorted.iter().map(|(k, _)| k.as_str()).collect();
    assert!(top_ids.contains(&"C-002")); // count=5, must be first
    assert!(!top_ids.contains(&"C-TAU-1")); // count=1, must be excluded
}

#[test]
fn severity_dominant_cause_selected() {
    let causes = [
        TerminalCause::Timeout,
        TerminalCause::LlmAdapterUnavailable,
        TerminalCause::VerificationExhaustion,
    ];
    let primary = causes
        .iter()
        .min_by_key(|c| c.severity_rank())
        .unwrap()
        .clone();
    assert_eq!(primary, TerminalCause::LlmAdapterUnavailable);
}
