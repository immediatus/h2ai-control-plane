use h2ai_autonomic::repair::{build_repair_context, PartialPass, RepairInput};
use h2ai_constraints::conflict::ConstraintConflictGraph;

fn empty_graph() -> ConstraintConflictGraph {
    ConstraintConflictGraph::build(&[])
}

fn base_input<'a>(
    retry_count: u32,
    checks: &'a [String],
    partial_passes: &'a [PartialPass],
    graph: &'a ConstraintConflictGraph,
) -> RepairInput<'a> {
    RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: graph,
        retry_count,
        attempts_remaining: 1,
        system_context_with_rubric: "SYSTEM CONTEXT",
        checks,
        partial_passes,
        prior_best_score: None,
        domain_syntheses: &[],
        coupled_constraint_hints: &[],
        passing_constraint_pins: &[],
    }
}

#[test]
fn test_b1_checklist_injected_at_retry_1() {
    let graph = empty_graph();
    let checks = vec![
        "Check A is present".to_string(),
        "Check B is present".to_string(),
    ];
    let input = base_input(1, &checks, &[], &graph);
    let output = build_repair_context(input);
    assert!(
        output.contains("COMPLIANCE CHECKLIST"),
        "checklist header missing"
    );
    assert!(output.contains("1. Check A is present"));
    assert!(output.contains("2. Check B is present"));
}

#[test]
fn test_b1_checklist_not_injected_at_retry_0() {
    let graph = empty_graph();
    let checks = vec!["Check A is present".to_string()];
    let input = base_input(0, &checks, &[], &graph);
    let output = build_repair_context(input);
    assert!(
        !output.contains("COMPLIANCE CHECKLIST"),
        "checklist must not appear at retry 0"
    );
}

#[test]
fn test_b1_no_injection_when_checks_empty() {
    let graph = empty_graph();
    let checks: Vec<String> = vec![];
    let input = base_input(1, &checks, &[], &graph);
    let output = build_repair_context(input);
    assert!(!output.contains("COMPLIANCE CHECKLIST"));
}

#[test]
fn test_b1_checklist_position_before_prior_proposal() {
    let graph = empty_graph();
    let checks = vec!["Check A".to_string()];
    let mut input = base_input(1, &checks, &[], &graph);
    input.prior_proposal_text = "THE PRIOR PROPOSAL TEXT";
    let output = build_repair_context(input);
    let checklist_pos = output.find("COMPLIANCE CHECKLIST").unwrap();
    let proposal_pos = output.find("PRIOR PROPOSAL").unwrap();
    assert!(
        checklist_pos < proposal_pos,
        "checklist must appear before prior proposal block"
    );
}
