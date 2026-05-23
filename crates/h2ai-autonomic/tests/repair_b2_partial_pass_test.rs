use h2ai_autonomic::repair::{build_repair_context, PartialPass, RepairInput};
use h2ai_constraints::conflict::ConstraintConflictGraph;

fn empty_graph() -> ConstraintConflictGraph {
    ConstraintConflictGraph::build(&[])
}

fn make_partial(proposal: &str, check_results: Vec<(usize, String, bool)>) -> PartialPass {
    let passed = check_results.iter().filter(|(_, _, p)| *p).count();
    let total = check_results.len();
    PartialPass {
        proposal_text: proposal.to_string(),
        check_results,
        score: passed as f64 / total.max(1) as f64,
    }
}

#[test]
fn test_b2_no_partials_no_block() {
    let graph = empty_graph();
    let input = RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 1,
        system_context_with_rubric: "CTX",
        checks: &[],
        partial_passes: &[],
        prior_best_score: None,
        domain_syntheses: &[],
    };
    let output = build_repair_context(input);
    assert!(!output.contains("PARTIAL EXAMPLE"));
}

#[test]
fn test_b2_single_partial_example_rendered() {
    let graph = empty_graph();
    let partial = make_partial(
        "My proposal text",
        vec![
            (0, "Check A".to_string(), true),
            (1, "Check B".to_string(), false),
        ],
    );
    let partials = vec![partial];
    let input = RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 1,
        system_context_with_rubric: "CTX",
        checks: &[],
        partial_passes: &partials,
        prior_best_score: None,
        domain_syntheses: &[],
    };
    let output = build_repair_context(input);
    assert!(output.contains("PARTIAL EXAMPLE 1"), "block header missing");
    assert!(output.contains('✓'), "passed check tick missing");
    assert!(output.contains('✗'), "failed check cross missing");
    assert!(output.contains("My proposal text"));
}

#[test]
fn test_b2_two_partials_rendered() {
    let graph = empty_graph();
    let p1 = make_partial("Proposal 1", vec![(0, "Check A".to_string(), true)]);
    let p2 = make_partial(
        "Proposal 2",
        vec![
            (0, "Check A".to_string(), false),
            (1, "Check B".to_string(), true),
        ],
    );
    let partials = vec![p1, p2];
    let input = RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 1,
        system_context_with_rubric: "CTX",
        checks: &[],
        partial_passes: &partials,
        prior_best_score: None,
        domain_syntheses: &[],
    };
    let output = build_repair_context(input);
    assert!(output.contains("PARTIAL EXAMPLE 1"));
    assert!(output.contains("PARTIAL EXAMPLE 2"));
}

#[test]
fn test_b2_synthesis_instruction_present_when_partials() {
    let graph = empty_graph();
    let partial = make_partial("P", vec![(0, "C".to_string(), true)]);
    let partials = vec![partial];
    let input = RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 1,
        system_context_with_rubric: "CTX",
        checks: &[],
        partial_passes: &partials,
        prior_best_score: None,
        domain_syntheses: &[],
    };
    let output = build_repair_context(input);
    assert!(
        output.contains("SYNTHESIS TASK"),
        "synthesis instruction missing"
    );
}

#[test]
fn test_b2_synthesis_instruction_absent_when_no_partials() {
    let graph = empty_graph();
    let input = RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 1,
        system_context_with_rubric: "CTX",
        checks: &[],
        partial_passes: &[],
        prior_best_score: None,
        domain_syntheses: &[],
    };
    let output = build_repair_context(input);
    assert!(!output.contains("SYNTHESIS TASK"));
}
