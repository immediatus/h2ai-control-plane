use h2ai_autonomic::repair::{build_synthesis_context, PartialPass, SynthesisInput};

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
fn test_build_synthesis_context_empty_checks_skips_checklist() {
    // checks.is_empty() → the `if !checks.is_empty()` block is skipped (line 569 else branch)
    let partial = make_partial("p", vec![]);
    let input = SynthesisInput {
        partial_passes: &[partial],
        checks: &[],
        system_context_with_rubric: "CTX",
    };
    let output = build_synthesis_context(input);
    assert!(output.contains("CTX"));
    assert!(
        !output.contains("COMPLIANCE CHECKLIST"),
        "empty checks must skip checklist block"
    );
}

#[test]
fn test_build_synthesis_context_contains_system_context() {
    let partial = make_partial("p", vec![(0, "C".to_string(), true)]);
    let input = SynthesisInput {
        partial_passes: &[partial],
        checks: &["C".to_string()],
        system_context_with_rubric: "MY_SYSTEM_CONTEXT",
    };
    let output = build_synthesis_context(input);
    assert!(output.contains("MY_SYSTEM_CONTEXT"));
}

#[test]
fn test_build_synthesis_context_single_partial_renders_all_sections() {
    let partial = make_partial(
        "proposal body",
        vec![
            (0, "Check A".to_string(), true),
            (1, "Check B".to_string(), false),
        ],
    );
    let checks = vec!["Check A".to_string(), "Check B".to_string()];
    let input = SynthesisInput {
        partial_passes: &[partial],
        checks: &checks,
        system_context_with_rubric: "CTX",
    };
    let output = build_synthesis_context(input);
    assert!(output.contains("SYNTHESIS WAVE"));
    assert!(output.contains("COMPLIANCE CHECKLIST"));
    assert!(output.contains("PARTIAL EXAMPLE 1"));
    assert!(output.contains("FINAL SYNTHESIS TASK"));
    assert!(output.contains("COHERENCE MANDATE"));
}

#[test]
fn test_build_synthesis_context_contains_coherence_mandate() {
    let partial = make_partial("p", vec![(0, "C".to_string(), true)]);
    let input = SynthesisInput {
        partial_passes: &[partial],
        checks: &["C".to_string()],
        system_context_with_rubric: "CTX",
    };
    let output = build_synthesis_context(input);
    assert!(output.contains("COHERENCE MANDATE"));
    assert!(
        output.contains("mutually exclusive") || output.contains("architecturally incompatible"),
        "must warn about architectural incompatibility"
    );
}

#[test]
fn test_build_synthesis_context_proposals_truncated() {
    // Truncation happens upstream in partial_pass_from_event; build_synthesis_context
    // passes proposal_text through unchanged. Simulate an already-truncated proposal.
    let truncated_proposal = format!(
        "{}\n[... truncated at 5000 chars; full text omitted to preserve context budget ...]",
        "x".repeat(1500)
    );
    let partial = make_partial(&truncated_proposal, vec![(0, "C".to_string(), true)]);
    let input = SynthesisInput {
        partial_passes: &[partial],
        checks: &["C".to_string()],
        system_context_with_rubric: "CTX",
    };
    let output = build_synthesis_context(input);
    assert!(
        output.contains("truncated at"),
        "truncation notice must appear in synthesis context"
    );
}

#[test]
fn test_build_synthesis_context_caps_at_three_examples() {
    let make = |i: usize| make_partial(&format!("p{i}"), vec![(i % 3, format!("C{i}"), true)]);
    let partials: Vec<PartialPass> = (0..4).map(make).collect();
    let checks: Vec<String> = (0..4).map(|i| format!("C{i}")).collect();
    let input = SynthesisInput {
        partial_passes: &partials,
        checks: &checks,
        system_context_with_rubric: "CTX",
    };
    let output = build_synthesis_context(input);
    assert!(
        output.contains("PARTIAL EXAMPLE 3"),
        "should render 3 examples"
    );
    assert!(
        !output.contains("PARTIAL EXAMPLE 4"),
        "must cap at 3 examples"
    );
}
