use chrono::Utc;
use h2ai_autonomic::merger::render_contradiction;
use h2ai_config::prompts::{
    CONTRADICTION_DETECTED_HEADER, CONTRADICTION_NOTE_HEADER, CONTRADICTION_SECTION_HEADER,
};
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation};
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
fn render_uses_all_three_template_section_headers() {
    let events = vec![pruned("flaw", &["C-1"])];
    let out = render_contradiction(&events, 5, 4);
    assert!(out.contains(CONTRADICTION_SECTION_HEADER.trim()));
    assert!(out.contains(CONTRADICTION_DETECTED_HEADER.trim()));
    assert!(out.contains(CONTRADICTION_NOTE_HEADER.trim()));
}

#[test]
fn render_includes_constraint_id_and_verifier_reason() {
    // helper sets verifier_reason = Some("CONSTRAINT-HLE-1 verifier reason")
    let events = vec![pruned("generic", &["CONSTRAINT-HLE-1"])];
    let out = render_contradiction(&events, 5, 4);
    assert!(out.contains("CONSTRAINT-HLE-1"));
    assert!(out.contains("CONSTRAINT-HLE-1 verifier reason"));
}

#[test]
fn render_prefers_verifier_reason_over_generic_reason() {
    let events = vec![pruned("generic reason", &["CONSTRAINT-HLE-1"])];
    // helper sets verifier_reason = Some("CONSTRAINT-HLE-1 verifier reason")
    let out = render_contradiction(&events, 5, 4);
    assert!(out.contains("CONSTRAINT-HLE-1 verifier reason"));
    // generic reason should NOT appear because verifier_reason takes precedence
    assert!(!out.contains("generic reason"));
}

#[test]
fn render_correct_counts_in_output() {
    let events = vec![pruned("flaw A", &["C-1"]), pruned("flaw B", &["C-2"])];
    let out = render_contradiction(&events, 5, 3);
    assert!(out.contains("3 of 5"));
}

#[test]
fn render_no_pruned_omits_detected_header() {
    let out = render_contradiction(&[], 3, 3);
    assert!(out.contains("3 of 3"));
    assert!(out.contains(CONTRADICTION_NOTE_HEADER.trim()));
    assert!(!out.contains(CONTRADICTION_DETECTED_HEADER.trim()));
}

#[test]
fn render_fallback_reason_when_no_violated_constraints() {
    let events = vec![BranchPrunedEvent {
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
    let out = render_contradiction(&events, 3, 2);
    assert!(out.contains("semilattice rejection"));
}
