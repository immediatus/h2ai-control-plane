use h2ai_orchestrator::specification_grounding::apply_implied_by_suppression;
use std::collections::HashMap;

fn clickhouse_table() -> HashMap<String, Vec<String>> {
    let mut m = HashMap::new();
    m.insert(
        "ClickHouse".to_string(),
        vec![
            "MergeTree".to_string(),
            "ReplacingMergeTree".to_string(),
            "SummingMergeTree".to_string(),
            "AggregatingMergeTree".to_string(),
            "CollapsingMergeTree".to_string(),
            "VersionedCollapsingMergeTree".to_string(),
            "GraphiteMergeTree".to_string(),
            "CoalescingMergeTree".to_string(),
            "Distributed".to_string(),
        ],
    );
    m
}

#[test]
fn suppression_removes_child_when_parent_is_grounded() {
    let nouns = vec!["MergeTree".to_string(), "BillingEvent".to_string()];
    let grounded_parents = vec!["ClickHouse".to_string()];
    let result = apply_implied_by_suppression(&nouns, &clickhouse_table(), &grounded_parents);
    assert!(
        !result.contains(&"MergeTree".to_string()),
        "MergeTree must be suppressed when ClickHouse is grounded"
    );
    assert!(
        result.contains(&"BillingEvent".to_string()),
        "BillingEvent is not a child of ClickHouse and must remain"
    );
}

#[test]
fn suppression_does_not_remove_child_when_parent_not_grounded() {
    let nouns = vec!["MergeTree".to_string()];
    let grounded_parents: Vec<String> = vec![];
    let result = apply_implied_by_suppression(&nouns, &clickhouse_table(), &grounded_parents);
    assert!(
        result.contains(&"MergeTree".to_string()),
        "MergeTree must remain when ClickHouse is not grounded"
    );
}

#[test]
fn suppression_is_idempotent() {
    let nouns = vec!["MergeTree".to_string()];
    let grounded_parents = vec!["ClickHouse".to_string()];
    let r1 = apply_implied_by_suppression(&nouns, &clickhouse_table(), &grounded_parents);
    let r2 = apply_implied_by_suppression(&r1, &clickhouse_table(), &grounded_parents);
    assert_eq!(r1, r2);
}

#[test]
fn empty_nouns_returns_empty() {
    let result =
        apply_implied_by_suppression(&[], &clickhouse_table(), &["ClickHouse".to_string()]);
    assert!(result.is_empty());
}

#[test]
fn empty_implied_by_returns_nouns_unchanged() {
    let nouns = vec!["MergeTree".to_string()];
    let result = apply_implied_by_suppression(&nouns, &HashMap::new(), &["ClickHouse".to_string()]);
    assert_eq!(result, nouns);
}

#[test]
fn srani_phase_suppresses_mergetree_when_clickhouse_grounded() {
    // Pure-fn path: apply_implied_by_suppression with a custom table.
    let nouns = vec![
        "MergeTree".to_string(),
        "BillingEvent".to_string(),
        "ClickHouse".to_string(),
    ];
    let table = clickhouse_table();
    let grounded = vec!["ClickHouse".to_string()];
    let filtered = apply_implied_by_suppression(&nouns, &table, &grounded);
    assert!(!filtered.contains(&"MergeTree".to_string()));
    assert!(filtered.contains(&"BillingEvent".to_string()));
}
