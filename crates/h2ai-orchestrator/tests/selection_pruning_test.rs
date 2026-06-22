use h2ai_orchestrator::gap_checkers::selection_pruning::extract_gaps_from_pruned;
use h2ai_orchestrator::gap_checkers::{GapKind, GapSource};

#[test]
fn empty_pruned_returns_no_gaps() {
    let gaps = extract_gaps_from_pruned(&[]);
    assert!(gaps.is_empty());
}

#[test]
fn constraint_violation_reason_produces_missing_provision_gap() {
    let pruned = vec![(
        "e1".to_string(),
        "The proposal lacks an explicit limitation-of-liability clause capping aggregate damages."
            .to_string(),
    )];
    let gaps = extract_gaps_from_pruned(&pruned);
    assert_eq!(gaps.len(), 1);
    assert!(matches!(gaps[0].kind, GapKind::MissingProvision));
    assert!(matches!(gaps[0].source, GapSource::SelectionPruning));
    assert!(!gaps[0].description.is_empty());
    assert!(gaps[0].id.starts_with('g'));
}

#[test]
fn multiple_pruned_with_duplicate_descriptions_produces_deduplicated_gaps() {
    let pruned = vec![
        ("e1".to_string(), "Missing liability cap".to_string()),
        ("e2".to_string(), "Missing liability cap".to_string()),
        ("e3".to_string(), "No IP ownership clause".to_string()),
    ];
    let gaps = extract_gaps_from_pruned(&pruned);
    // Duplicates deduplicated by description
    assert_eq!(gaps.len(), 2);
}

#[test]
fn gap_ids_are_stable_and_unique() {
    let pruned = vec![
        ("e1".to_string(), "Gap A".to_string()),
        ("e2".to_string(), "Gap B".to_string()),
    ];
    let gaps = extract_gaps_from_pruned(&pruned);
    let ids: std::collections::HashSet<&str> = gaps.iter().map(|g| g.id.as_str()).collect();
    assert_eq!(ids.len(), gaps.len(), "IDs must be unique");
}
