use chrono::Utc;
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation, ContradictionAnalysis};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::RoleErrorCost;

fn pruned(reason: &str, constraint_ids: &[&str]) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: reason.to_string(),
        raw_output: String::new(),
        constraint_error_cost: RoleErrorCost::new(0.0).unwrap(),
        violated_constraints: constraint_ids
            .iter()
            .map(|id| ConstraintViolation {
                constraint_id: id.to_string(),
                score: 0.0,
                severity_label: "Hard".to_string(),
                remediation_hint: None,
                constraint_description: format!("{id} description"),
                verifier_reason: Some(format!("{id} verifier reason")),
                check_verdicts: vec![],
                criteria_pass: None,
                check_reasons: None,
            })
            .collect(),
        timestamp: Utc::now(),
        retry_count: 0,
        bypass_reason: None,
    }
}

#[test]
fn from_pruned_empty_returns_no_contradictions() {
    let analysis = ContradictionAnalysis::from_pruned(&[], 5, 4, String::new());
    assert_eq!(analysis.n_valid, 4);
    assert_eq!(analysis.n_total, 5);
    assert!(analysis.contradictions.is_empty());
}

#[test]
fn from_pruned_single_entry_with_violated_constraints() {
    let pruned_events = vec![pruned("proof flaw detected", &["CONSTRAINT-HLE-1"])];
    let analysis = ContradictionAnalysis::from_pruned(&pruned_events, 5, 4, String::new());

    assert_eq!(analysis.n_valid, 4);
    assert_eq!(analysis.n_total, 5);
    assert_eq!(analysis.contradictions.len(), 1);

    let entry = &analysis.contradictions[0];
    assert_eq!(entry.reason, "proof flaw detected");
    assert_eq!(entry.violated_constraints.len(), 1);
    assert_eq!(
        entry.violated_constraints[0].constraint_id,
        "CONSTRAINT-HLE-1"
    );
}

#[test]
fn from_pruned_multiple_entries_preserve_order() {
    let pruned_events = vec![
        pruned("flaw A", &["CONSTRAINT-HLE-1"]),
        pruned("flaw B", &["CONSTRAINT-HLE-1", "CONSTRAINT-HLE-2"]),
    ];
    let analysis = ContradictionAnalysis::from_pruned(&pruned_events, 5, 3, String::new());

    assert_eq!(analysis.n_valid, 3);
    assert_eq!(analysis.n_total, 5);
    assert_eq!(analysis.contradictions.len(), 2);
    assert_eq!(analysis.contradictions[0].reason, "flaw A");
    assert_eq!(analysis.contradictions[1].violated_constraints.len(), 2);
}

#[test]
fn from_pruned_no_violated_constraints_stores_reason() {
    let pruned_events = vec![BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: "semilattice rejection".to_string(),
        raw_output: String::new(),
        constraint_error_cost: RoleErrorCost::new(0.0).unwrap(),
        violated_constraints: vec![],
        timestamp: Utc::now(),
        retry_count: 0,
        bypass_reason: None,
    }];
    let analysis = ContradictionAnalysis::from_pruned(&pruned_events, 3, 2, String::new());

    let entry = &analysis.contradictions[0];
    assert_eq!(entry.reason, "semilattice rejection");
    assert!(entry.violated_constraints.is_empty());
}

#[test]
fn from_pruned_stores_rendered_string_verbatim() {
    let pruned_events = vec![pruned("flaw", &["C-1"])];
    let rendered = "pre-formatted template text".to_string();
    let analysis = ContradictionAnalysis::from_pruned(&pruned_events, 5, 4, rendered.clone());
    assert_eq!(analysis.rendered, rendered);
}

#[test]
fn contradiction_analysis_serializes_and_deserializes() {
    let pruned_events = vec![pruned("round-trip check", &["CONSTRAINT-HLE-1"])];
    let rendered = "rendered template".to_string();
    let analysis = ContradictionAnalysis::from_pruned(&pruned_events, 5, 4, rendered.clone());

    let json = serde_json::to_string(&analysis).expect("serialize");
    let back: ContradictionAnalysis = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(back.n_valid, analysis.n_valid);
    assert_eq!(back.n_total, analysis.n_total);
    assert_eq!(back.rendered, rendered);
    assert_eq!(back.contradictions.len(), 1);
    assert_eq!(back.contradictions[0].reason, "round-trip check");
}
