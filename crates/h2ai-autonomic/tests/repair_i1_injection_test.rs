use h2ai_autonomic::repair::{build_repair_context, RepairInput};
use h2ai_constraints::conflict::ConstraintConflictGraph;
use h2ai_types::gap_i1::DomainSynthesis;

fn empty_graph() -> ConstraintConflictGraph {
    ConstraintConflictGraph::build(&[])
}

fn make_repair_input_with_syntheses<'a>(
    syntheses: &'a [DomainSynthesis],
    graph: &'a ConstraintConflictGraph,
) -> RepairInput<'a> {
    RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: graph,
        retry_count: 1,
        attempts_remaining: 2,
        system_context_with_rubric: "SYSTEM CONTEXT",
        checks: &[],
        partial_passes: &[],
        prior_best_score: None,
        domain_syntheses: syntheses,
        coupled_constraint_hints: &[],
        passing_constraint_pins: &[],
    }
}

#[test]
fn no_syntheses_does_not_add_correction_slot() {
    let graph = empty_graph();
    let syntheses: Vec<DomainSynthesis> = vec![];
    let input = make_repair_input_with_syntheses(&syntheses, &graph);
    let ctx = build_repair_context(input);
    assert!(
        !ctx.contains("DOMAIN KNOWLEDGE CORRECTION"),
        "no syntheses → no correction slot"
    );
}

#[test]
fn single_synthesis_injects_correction_slot() {
    let graph = empty_graph();
    let syntheses = vec![DomainSynthesis {
        check_id: ("CONSTRAINT-008".to_string(), 1),
        incorrect_pattern: "SETNX as standalone idempotency primitive".to_string(),
        correct_pattern: "SET key val NX EX ttl inside Lua EVAL".to_string(),
        mechanistic_reason: "Lua EVAL is atomic; SETNX alone does not protect concurrent updates"
            .to_string(),
        source: Some("https://redis.io/docs/commands/set/".to_string()),
        confidence: 0.85,
        injected_at_wave: None,
        pre_injection_pass_rate: None,
        post_injection_pass_rates: vec![],
    }];
    let input = make_repair_input_with_syntheses(&syntheses, &graph);
    let ctx = build_repair_context(input);
    assert!(
        ctx.contains("DOMAIN KNOWLEDGE CORRECTION"),
        "correction slot must appear"
    );
    assert!(
        ctx.contains("SETNX as standalone"),
        "wrong belief must appear"
    );
    assert!(ctx.contains("SET key val NX"), "correct belief must appear");
    assert!(ctx.contains("Lua EVAL is atomic"), "reason must appear");
}

#[test]
fn correction_slot_appears_before_repair_target() {
    let graph = empty_graph();
    let syntheses = vec![DomainSynthesis {
        check_id: ("C".to_string(), 0),
        incorrect_pattern: "wrong".to_string(),
        correct_pattern: "right".to_string(),
        mechanistic_reason: "because".to_string(),
        source: None,
        confidence: 0.9,
        injected_at_wave: None,
        pre_injection_pass_rate: None,
        post_injection_pass_rates: vec![],
    }];
    let input = make_repair_input_with_syntheses(&syntheses, &graph);
    let ctx = build_repair_context(input);
    let correction_pos = ctx.find("DOMAIN KNOWLEDGE CORRECTION");
    let repair_target_pos = ctx.find("REPAIR TARGET");
    match (correction_pos, repair_target_pos) {
        (Some(c), Some(r)) => assert!(c < r, "correction slot must precede repair target block"),
        (None, _) => panic!("correction slot not found"),
        (_, None) => {} // REPAIR TARGET may not always appear; that's OK
    }
}

#[test]
fn multiple_syntheses_all_injected() {
    let graph = empty_graph();
    let syntheses = vec![
        DomainSynthesis {
            check_id: ("C".to_string(), 0),
            incorrect_pattern: "wrong-a".to_string(),
            correct_pattern: "right-a".to_string(),
            mechanistic_reason: "reason-a".to_string(),
            source: None,
            confidence: 0.8,
            injected_at_wave: None,
            pre_injection_pass_rate: None,
            post_injection_pass_rates: vec![],
        },
        DomainSynthesis {
            check_id: ("C".to_string(), 1),
            incorrect_pattern: "wrong-b".to_string(),
            correct_pattern: "right-b".to_string(),
            mechanistic_reason: "reason-b".to_string(),
            source: None,
            confidence: 0.9,
            injected_at_wave: None,
            pre_injection_pass_rate: None,
            post_injection_pass_rates: vec![],
        },
    ];
    let input = make_repair_input_with_syntheses(&syntheses, &graph);
    let ctx = build_repair_context(input);
    assert!(ctx.contains("wrong-a"));
    assert!(ctx.contains("wrong-b"));
    assert!(ctx.contains("right-a"));
    assert!(ctx.contains("right-b"));
}
