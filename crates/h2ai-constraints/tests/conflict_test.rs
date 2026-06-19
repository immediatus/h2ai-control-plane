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
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
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
    conflicts.sort_unstable();
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

// ── From conflict.rs inline tests ─────────────────────────────────────────────

#[test]
fn seeds_coupling_from_related_to() {
    let make_doc_r = |id: &str, related: &[&str]| -> ConstraintDoc {
        ConstraintDoc {
            related_to: related.iter().map(|s| s.to_string()).collect(),
            ..ConstraintDoc::new_llm_judge(id, "")
        }
    };
    let docs = vec![
        make_doc_r("C-005", &["C-TAU-2"]),
        make_doc_r("C-TAU-2", &["C-005"]),
    ];
    let graph = ConstraintConflictGraph::build(&docs);
    assert!(
        graph.are_conflicting("C-005", "C-TAU-2"),
        "related_to cross-reference must produce a coupling pair in the graph"
    );
    assert!(graph.conflicts_for("C-005").contains(&"C-TAU-2"));
}

#[test]
fn related_to_only_on_one_side_still_seeds() {
    let docs = vec![
        ConstraintDoc {
            related_to: vec!["B".to_string()],
            ..ConstraintDoc::new_llm_judge("A", "")
        },
        ConstraintDoc::new_llm_judge("B", ""),
    ];
    let graph = ConstraintConflictGraph::build(&docs);
    assert!(graph.are_conflicting("A", "B"));
}

// ── Lines 67-70: conflicts_for b == id branch and None branch ─────────────────

#[test]
fn conflicts_for_works_when_querying_second_member_of_pair() {
    // A conflicts with B. Calling conflicts_for("B") exercises the `b == id` branch
    // (canonical pair stores alphabetically: ("A", "B"), so when id="B", b matches).
    // conflicts_for("A") exercises the `a == id` arm.
    // Adding a third pair "C"↔"D" ensures the None branch (neither a nor b == id)
    // is also exercised when iterating all pairs.
    let docs = vec![
        make_doc(
            "A",
            ConstraintPredicate::SemanticOrdering {
                first: "alpha".to_owned(),
                then: "beta".to_owned(),
                passes: 3,
            },
        ),
        make_doc(
            "B",
            ConstraintPredicate::SemanticOrdering {
                first: "beta".to_owned(),
                then: "alpha".to_owned(),
                passes: 3,
            },
        ),
        make_doc(
            "C",
            ConstraintPredicate::SemanticOrdering {
                first: "x".to_owned(),
                then: "y".to_owned(),
                passes: 3,
            },
        ),
        make_doc(
            "D",
            ConstraintPredicate::SemanticOrdering {
                first: "y".to_owned(),
                then: "x".to_owned(),
                passes: 3,
            },
        ),
    ];
    let g = ConstraintConflictGraph::build(&docs);

    // conflicts_for("B") must return "A" — exercises the `b == id` arm
    // (pair ("A","B"): a="A", b="B", so b == id → Some(a))
    let conflicts_b = g.conflicts_for("B");
    assert_eq!(conflicts_b, vec!["A"], "conflicts_for(B) must return [A]");

    // conflicts_for("A") exercises the `a == id` arm
    // Also exercises the `else { None }` arm for pair ("C","D") which doesn't involve "A"
    let conflicts_a = g.conflicts_for("A");
    assert_eq!(conflicts_a, vec!["B"], "conflicts_for(A) must return [B]");
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

// ── ConstraintDoc serde defaults ──────────────────────────────────────────────

#[test]
fn constraint_doc_version_defaults_to_one_when_absent() {
    let json = serde_json::json!({
        "id": "TEST-001",
        "source_file": "test.yaml",
        "description": "Test constraint",
        "severity": {"Hard": {"threshold": 0.5}},
        "predicate": {"LlmJudge": {"rubric": "pass if correct"}},
        "remediation_hint": null
    });
    let doc: ConstraintDoc = serde_json::from_value(json).unwrap();
    assert_eq!(doc.version, 1, "missing version field must default to 1");
}

#[test]
fn oracle_execution_timeout_defaults_to_thirty_when_absent() {
    let json = serde_json::json!({
        "OracleExecution": {
            "test_runner_uri": "http://localhost:9000/run",
            "test_suite": "suite_a"
        }
    });
    let pred: ConstraintPredicate = serde_json::from_value(json).unwrap();
    match pred {
        ConstraintPredicate::OracleExecution { timeout_secs, .. } => {
            assert_eq!(timeout_secs, 30, "missing timeout_secs must default to 30");
        }
        _ => panic!("expected OracleExecution"),
    }
}
