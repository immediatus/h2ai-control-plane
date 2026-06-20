use h2ai_orchestrator::thinking_loop::{
    find_uncovered_constraints, parse_archetypes_from_markdown, synthesize_coverage_archetype,
};
use h2ai_types::thinking::ArchetypeSpec;

fn make_archetype(focus: &[&str]) -> ArchetypeSpec {
    use h2ai_types::manifest::CotStyle;
    use h2ai_types::thinking::ModelTier;
    ArchetypeSpec {
        name: "test-archetype".to_string(),
        persona: "test persona".to_string(),
        scope: "test scope".to_string(),
        confidence: 0.8,
        tau: 0.3,
        model_tier: ModelTier::Standard,
        cot_style: CotStyle::StepByStep,
        focus_constraints: focus.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
fn find_uncovered_constraints_returns_empty_when_all_covered() {
    let archetypes = vec![
        make_archetype(&["C-001", "C-002"]),
        make_archetype(&["C-003"]),
    ];
    let ids = vec![
        "C-001".to_string(),
        "C-002".to_string(),
        "C-003".to_string(),
    ];
    let uncovered = find_uncovered_constraints(&archetypes, &ids);
    assert!(
        uncovered.is_empty(),
        "expected no uncovered; got {uncovered:?}"
    );
}

#[test]
fn find_uncovered_constraints_detects_missing_constraint() {
    let archetypes = vec![make_archetype(&["C-001", "C-002"])];
    let ids = vec![
        "C-001".to_string(),
        "C-002".to_string(),
        "C-TAU-2".to_string(),
    ];
    let uncovered = find_uncovered_constraints(&archetypes, &ids);
    assert_eq!(uncovered, vec!["C-TAU-2".to_string()]);
}

#[test]
fn find_uncovered_constraints_treats_empty_focus_as_no_coverage() {
    let archetypes = vec![make_archetype(&[])]; // empty focus = covers nothing specifically
    let ids = vec!["C-001".to_string()];
    let uncovered = find_uncovered_constraints(&archetypes, &ids);
    assert_eq!(uncovered, vec!["C-001".to_string()]);
}

#[test]
fn find_uncovered_constraints_all_ids_empty_returns_empty() {
    let archetypes = vec![make_archetype(&["C-001"])];
    let uncovered = find_uncovered_constraints(&archetypes, &[]);
    assert!(uncovered.is_empty());
}

#[test]
fn synthesize_coverage_archetype_includes_constraint_id_in_name_and_persona() {
    let uncovered_id = "C-TAU-2";
    let description = "transaction must be auditable within 5 seconds";
    let corpus =
        vec![h2ai_constraints::types::ConstraintDoc::new_with_description("C-TAU-2", description)];
    let archetype = synthesize_coverage_archetype(uncovered_id, &corpus);
    assert!(
        archetype.name.contains("C-TAU-2"),
        "synthesized archetype name must reference constraint id; got: {}",
        archetype.name
    );
    assert!(
        archetype.persona.contains("C-TAU-2"),
        "synthesized archetype persona must reference constraint id; got: {}",
        archetype.persona
    );
    assert!(
        archetype.persona.contains("transaction must be auditable"),
        "synthesized archetype persona must contain the constraint description; got: {}",
        archetype.persona
    );
    assert_eq!(
        archetype.focus_constraints,
        vec!["C-TAU-2".to_string()],
        "focus_constraints must contain the uncovered constraint id"
    );
}

#[test]
fn synthesize_coverage_archetype_no_corpus_entry_still_produces_valid_archetype() {
    let archetype = synthesize_coverage_archetype("C-BFT-1", &[]);
    assert!(
        archetype.name.contains("C-BFT-1"),
        "archetype name must include constraint id even without corpus: {}",
        archetype.name
    );
    assert_eq!(archetype.focus_constraints, vec!["C-BFT-1".to_string()]);
}

#[test]
fn parse_archetypes_from_markdown_populates_focus_constraints() {
    let md = "\
## Archetype 1: tau-specialist
**Lens:** Transaction audit specialist
**Persona:** You are a compliance engineer who ensures all transactions are auditable.
**Scope:** Audit trail and transaction logging compliance.
**Confidence:** 0.85
**Tau:** 0.3
**Model tier:** standard
**CoT style:** step_by_step
**Focus Constraints:** C-TAU-2, C-BFT-1
";
    let archetypes = parse_archetypes_from_markdown(md).expect("parse should succeed");
    assert_eq!(archetypes.len(), 1);
    assert!(
        archetypes[0]
            .focus_constraints
            .contains(&"C-TAU-2".to_string()),
        "focus_constraints must include C-TAU-2; got: {:?}",
        archetypes[0].focus_constraints
    );
    assert!(
        archetypes[0]
            .focus_constraints
            .contains(&"C-BFT-1".to_string()),
        "focus_constraints must include C-BFT-1; got: {:?}",
        archetypes[0].focus_constraints
    );
}

#[test]
fn parse_archetypes_from_markdown_all_keyword_produces_empty_focus() {
    let md = "\
## Archetype 1: generalist
**Lens:** General engineer
**Persona:** You are a generalist who covers all aspects.
**Scope:** Complete solution.
**Confidence:** 0.7
**Tau:** 0.5
**Model tier:** standard
**CoT style:** none
**Focus Constraints:** all
";
    let archetypes = parse_archetypes_from_markdown(md).expect("parse should succeed");
    assert_eq!(archetypes.len(), 1);
    assert!(
        archetypes[0].focus_constraints.is_empty(),
        "\"all\" keyword must produce empty focus_constraints; got: {:?}",
        archetypes[0].focus_constraints
    );
}
