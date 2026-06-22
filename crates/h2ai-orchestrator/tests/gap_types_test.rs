use h2ai_orchestrator::gap_checkers::{Gap, GapKind, GapSeverity, GapSource};
use h2ai_orchestrator::gap_registry::{CycleError, GapRegistry};

#[test]
fn uncertain_domain_gap_construction() {
    let g = Gap {
        id: "g-uncertain-ctx-0".into(),
        kind: GapKind::UncertainDomain,
        severity: GapSeverity::Medium,
        description: "Task context flags domain as uncertain (keyword: \"unsettled\")".into(),
        affected_provisions: vec!["context_uncertainty_0".into()],
        depends_on: None,
        source: GapSource::TaskContextSeeding,
    };
    assert!(matches!(g.kind, GapKind::UncertainDomain));
    assert!(matches!(g.source, GapSource::TaskContextSeeding));
    assert!(g.depends_on.is_none());
}

#[test]
fn uncertain_domain_gap_is_independent_and_batched_first() {
    // UncertainDomain gaps have no dependencies — they go in the first batch.
    let gaps = vec![
        Gap {
            id: "g-uncertain-ctx-0".into(),
            kind: GapKind::UncertainDomain,
            severity: GapSeverity::Medium,
            description: "unsettled domain".into(),
            affected_provisions: vec![],
            depends_on: None,
            source: GapSource::TaskContextSeeding,
        },
        Gap {
            id: "g1".into(),
            kind: GapKind::MissingProvision,
            severity: GapSeverity::High,
            description: "missing clause".into(),
            affected_provisions: vec![],
            depends_on: None,
            source: GapSource::SelectionPruning,
        },
    ];
    let registry = GapRegistry::new(gaps);
    let batches = registry.dispatch_batches().expect("no cycle");
    assert_eq!(batches.len(), 1, "both independent — one batch");
    assert_eq!(batches[0].len(), 2);
}

#[test]
fn gap_construction_and_display() {
    let g = Gap {
        id: "g1".into(),
        kind: GapKind::MissingProvision,
        severity: GapSeverity::High,
        description: "Liability cap missing".into(),
        affected_provisions: vec!["Section 1".into()],
        depends_on: None,
        source: GapSource::SelectionPruning,
    };
    assert_eq!(g.id, "g1");
    assert!(matches!(g.kind, GapKind::MissingProvision));
    assert!(g.depends_on.is_none());
}

#[test]
fn dispatch_batches_independent_gaps_returns_single_batch() {
    let gaps = vec![
        Gap {
            id: "g1".into(),
            kind: GapKind::MissingProvision,
            severity: GapSeverity::High,
            description: "".into(),
            affected_provisions: vec!["Section 1".into()],
            depends_on: None,
            source: GapSource::SelectionPruning,
        },
        Gap {
            id: "g2".into(),
            kind: GapKind::MissingProvision,
            severity: GapSeverity::Medium,
            description: "".into(),
            affected_provisions: vec!["Section 2".into()],
            depends_on: None,
            source: GapSource::SelectionPruning,
        },
    ];
    let registry = GapRegistry::new(gaps);
    let batches = registry.dispatch_batches().expect("no cycle");
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].len(), 2);
}

#[test]
fn dispatch_batches_chain_dependency_returns_ordered_batches() {
    let gaps = vec![
        Gap {
            id: "g1".into(),
            kind: GapKind::MissingProvision,
            severity: GapSeverity::High,
            description: "".into(),
            affected_provisions: vec!["Section 1".into()],
            depends_on: None,
            source: GapSource::SelectionPruning,
        },
        Gap {
            id: "g2".into(),
            kind: GapKind::MissingProvision,
            severity: GapSeverity::Medium,
            description: "".into(),
            affected_provisions: vec!["Section 1".into()],
            depends_on: Some(vec!["g1".into()]),
            source: GapSource::SelectionPruning,
        },
    ];
    let registry = GapRegistry::new(gaps);
    let batches = registry.dispatch_batches().expect("no cycle");
    assert_eq!(batches.len(), 2);
    assert!(batches[0].contains(&"g1".to_string()));
    assert!(batches[1].contains(&"g2".to_string()));
}

#[test]
fn dispatch_batches_cycle_returns_error() {
    let gaps = vec![
        Gap {
            id: "g1".into(),
            kind: GapKind::MissingProvision,
            severity: GapSeverity::High,
            description: "".into(),
            affected_provisions: vec!["Section 1".into()],
            depends_on: Some(vec!["g2".into()]),
            source: GapSource::SelectionPruning,
        },
        Gap {
            id: "g2".into(),
            kind: GapKind::MissingProvision,
            severity: GapSeverity::Medium,
            description: "".into(),
            affected_provisions: vec!["Section 2".into()],
            depends_on: Some(vec!["g1".into()]),
            source: GapSource::SelectionPruning,
        },
    ];
    let registry = GapRegistry::new(gaps);
    let result = registry.dispatch_batches();
    assert!(matches!(result, Err(CycleError(_))));
}
