use h2ai_constraints::conflict::ConstraintConflictGraph;
use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity, NumericOp};

fn make_doc(id: &str, pred: ConstraintPredicate) -> ConstraintDoc {
    ConstraintDoc {
        id: id.to_owned(),
        source_file: format!("{id}.yaml"),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: pred,
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    }
}

#[test]
fn semantic_ordering_conflict_detected() {
    let docs = vec![
        make_doc(
            "A",
            ConstraintPredicate::SemanticOrdering {
                first: "debit".to_owned(),
                then: "publish".to_owned(),
                passes: 3,
            },
        ),
        make_doc(
            "B",
            ConstraintPredicate::SemanticOrdering {
                first: "publish".to_owned(),
                then: "debit".to_owned(),
                passes: 3,
            },
        ),
    ];
    let g = ConstraintConflictGraph::build(&docs);
    assert!(g.are_conflicting("A", "B"));
    assert!(g.are_conflicting("B", "A"));
}

#[test]
fn non_conflicting_ordering_not_flagged() {
    let docs = vec![
        make_doc(
            "A",
            ConstraintPredicate::SemanticOrdering {
                first: "debit".to_owned(),
                then: "publish".to_owned(),
                passes: 3,
            },
        ),
        make_doc(
            "B",
            ConstraintPredicate::SemanticOrdering {
                first: "debit".to_owned(),
                then: "notify".to_owned(),
                passes: 3,
            },
        ),
    ];
    let g = ConstraintConflictGraph::build(&docs);
    assert!(!g.are_conflicting("A", "B"));
}

#[test]
fn numeric_threshold_empty_feasibility_conflict() {
    // Le 50 AND Ge 60 → empty region
    let docs = vec![
        make_doc(
            "C",
            ConstraintPredicate::NumericThreshold {
                field_pattern: "timeout_ms".to_owned(),
                op: NumericOp::Le,
                value: 50.0,
            },
        ),
        make_doc(
            "D",
            ConstraintPredicate::NumericThreshold {
                field_pattern: "timeout_ms".to_owned(),
                op: NumericOp::Ge,
                value: 60.0,
            },
        ),
    ];
    let g = ConstraintConflictGraph::build(&docs);
    assert!(g.are_conflicting("C", "D"));
}

#[test]
fn numeric_threshold_compatible_range_not_flagged() {
    // Le 100 AND Ge 50 → [50,100] is valid
    let docs = vec![
        make_doc(
            "E",
            ConstraintPredicate::NumericThreshold {
                field_pattern: "timeout_ms".to_owned(),
                op: NumericOp::Le,
                value: 100.0,
            },
        ),
        make_doc(
            "F",
            ConstraintPredicate::NumericThreshold {
                field_pattern: "timeout_ms".to_owned(),
                op: NumericOp::Ge,
                value: 50.0,
            },
        ),
    ];
    let g = ConstraintConflictGraph::build(&docs);
    assert!(!g.are_conflicting("E", "F"));
}

#[test]
fn conflicts_for_returns_all_conflicting_ids() {
    let docs = vec![
        make_doc(
            "A",
            ConstraintPredicate::SemanticOrdering {
                first: "x".to_owned(),
                then: "y".to_owned(),
                passes: 3,
            },
        ),
        make_doc(
            "B",
            ConstraintPredicate::SemanticOrdering {
                first: "y".to_owned(),
                then: "x".to_owned(),
                passes: 3,
            },
        ),
        make_doc(
            "C",
            ConstraintPredicate::SemanticOrdering {
                first: "y".to_owned(),
                then: "x".to_owned(),
                passes: 3,
            },
        ),
    ];
    let g = ConstraintConflictGraph::build(&docs);
    let mut conflicts = g.conflicts_for("A");
    conflicts.sort();
    assert_eq!(conflicts, vec!["B", "C"]);
}

#[test]
fn empty_corpus_is_empty() {
    let g = ConstraintConflictGraph::build(&[]);
    assert!(g.is_empty());
}

#[test]
fn composite_and_conflict_propagates() {
    use h2ai_constraints::types::CompositeOp;
    // A contains SemanticOrdering(X→Y) inside And — should propagate conflict with B(Y→X)
    let docs = vec![
        make_doc(
            "A",
            ConstraintPredicate::Composite {
                op: CompositeOp::And,
                children: vec![ConstraintPredicate::SemanticOrdering {
                    first: "x".to_owned(),
                    then: "y".to_owned(),
                    passes: 3,
                }],
            },
        ),
        make_doc(
            "B",
            ConstraintPredicate::SemanticOrdering {
                first: "y".to_owned(),
                then: "x".to_owned(),
                passes: 3,
            },
        ),
    ];
    let g = ConstraintConflictGraph::build(&docs);
    assert!(
        g.are_conflicting("A", "B"),
        "And-composite should propagate ordering conflict"
    );
}

#[test]
fn composite_or_does_not_propagate_conflict() {
    use h2ai_constraints::types::CompositeOp;
    // A contains SemanticOrdering(X→Y) inside Or — should NOT flag conflict with B(Y→X)
    let docs = vec![
        make_doc(
            "A",
            ConstraintPredicate::Composite {
                op: CompositeOp::Or,
                children: vec![ConstraintPredicate::SemanticOrdering {
                    first: "x".to_owned(),
                    then: "y".to_owned(),
                    passes: 3,
                }],
            },
        ),
        make_doc(
            "B",
            ConstraintPredicate::SemanticOrdering {
                first: "y".to_owned(),
                then: "x".to_owned(),
                passes: 3,
            },
        ),
    ];
    let g = ConstraintConflictGraph::build(&docs);
    assert!(
        !g.are_conflicting("A", "B"),
        "Or-composite must not produce false conflict"
    );
}
