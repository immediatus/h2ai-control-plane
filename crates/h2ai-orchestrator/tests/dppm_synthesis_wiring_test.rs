/// Regression tests for the DPPM synthesis wiring:
/// `all_domain_syntheses()` must return the full cache, and those syntheses
/// must flow into the cluster-solver repair context so the LLM receives the
/// incorrect→correct pattern corrections accumulated by gap researchers.
///
/// These tests catch the original bug where `engine.rs` hardcoded
/// `domain_syntheses: &[]`, silently discarding all gap_i1 knowledge
/// before every DPPM cluster-solver call.
use h2ai_autonomic::repair::{build_repair_context, RepairInput};
use h2ai_constraints::conflict::ConstraintConflictGraph;
use h2ai_orchestrator::mape_k::MapeKController;
use h2ai_types::gap_i1::DomainSynthesis;

fn empty_graph() -> ConstraintConflictGraph {
    ConstraintConflictGraph::build(&[])
}

fn make_synthesis(
    constraint_id: &str,
    check_idx: usize,
    wrong: &str,
    right: &str,
) -> DomainSynthesis {
    DomainSynthesis {
        check_id: (constraint_id.to_string(), check_idx),
        incorrect_pattern: wrong.to_string(),
        correct_pattern: right.to_string(),
        mechanistic_reason: format!("{wrong} is wrong because it has a race window"),
        source: None,
        confidence: 0.5,
        injected_at_wave: None,
        pre_injection_pass_rate: None,
        post_injection_pass_rates: vec![],
    }
}

// ── all_domain_syntheses ──────────────────────────────────────────────────────

#[test]
fn empty_cache_returns_empty_vec() {
    let ctrl = MapeKController::new_minimal();
    assert!(ctrl.all_domain_syntheses().is_empty());
}

#[test]
fn single_entry_cache_returns_it() {
    let mut ctrl = MapeKController::new_minimal();
    ctrl.seed_synthesis(
        "CONSTRAINT-008",
        1,
        make_synthesis("CONSTRAINT-008", 1, "GET-SETEX", "SET NX EX 30"),
    );

    let result = ctrl.all_domain_syntheses();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].incorrect_pattern, "GET-SETEX");
    assert_eq!(result[0].correct_pattern, "SET NX EX 30");
}

#[test]
fn multiple_entries_all_returned() {
    let mut ctrl = MapeKController::new_minimal();
    ctrl.seed_synthesis(
        "CONSTRAINT-008",
        1,
        make_synthesis("CONSTRAINT-008", 1, "GET-SETEX", "SET NX EX 30"),
    );
    ctrl.seed_synthesis(
        "CONSTRAINT-004",
        2,
        make_synthesis(
            "CONSTRAINT-004",
            2,
            "no TTL on idempotency key",
            "30s TTL via EX 30",
        ),
    );
    ctrl.seed_synthesis(
        "CONSTRAINT-TAU-1",
        2,
        make_synthesis(
            "CONSTRAINT-TAU-1",
            2,
            "ARGV[3] tenant_id",
            "JWT-derived tenant_id",
        ),
    );

    let result = ctrl.all_domain_syntheses();
    assert_eq!(result.len(), 3);
}

// ── full chain: cache → all_domain_syntheses → build_repair_context ──────────
//
// This is the critical wiring test. If engine.rs were to pass `&[]` instead of
// `controller.all_domain_syntheses()`, the solver context would have no
// "DOMAIN KNOWLEDGE CORRECTION" block and the LLM would never learn what it
// did wrong.

#[test]
fn solver_context_is_empty_when_cache_is_empty() {
    let ctrl = MapeKController::new_minimal();
    let syntheses = ctrl.all_domain_syntheses();

    let graph = empty_graph();
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 1,
        system_context_with_rubric: "SYS",
        checks: &[],
        partial_passes: &[],
        prior_best_score: None,
        domain_syntheses: &syntheses,
        coupled_constraint_hints: &[],
        passing_constraint_pins: &[],
    });

    assert!(
        !ctx.contains("DOMAIN KNOWLEDGE CORRECTION"),
        "empty cache must produce no correction block"
    );
}

#[test]
fn solver_context_contains_correction_when_cache_has_entries() {
    let mut ctrl = MapeKController::new_minimal();
    ctrl.seed_synthesis(
        "CONSTRAINT-008",
        1,
        make_synthesis(
            "CONSTRAINT-008",
            1,
            "GET-SETEX race condition",
            "SET key NX EX 30",
        ),
    );

    // This is what engine.rs must do — call all_domain_syntheses(), NOT pass &[].
    let syntheses = ctrl.all_domain_syntheses();
    assert!(
        !syntheses.is_empty(),
        "syntheses must be collected from cache"
    );

    let graph = empty_graph();
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 1,
        system_context_with_rubric: "SYS",
        checks: &[],
        partial_passes: &[],
        prior_best_score: None,
        domain_syntheses: &syntheses,
        coupled_constraint_hints: &[],
        passing_constraint_pins: &[],
    });

    assert!(
        ctx.contains("DOMAIN KNOWLEDGE CORRECTION"),
        "correction block must appear when cache is non-empty; got:\n{ctx}"
    );
    assert!(
        ctx.contains("GET-SETEX race condition"),
        "incorrect pattern must be injected into solver context"
    );
    assert!(
        ctx.contains("SET key NX EX 30"),
        "correct pattern must be injected into solver context"
    );
}

#[test]
fn solver_context_contains_all_cached_corrections() {
    let mut ctrl = MapeKController::new_minimal();
    ctrl.seed_synthesis(
        "CONSTRAINT-008",
        1,
        make_synthesis("CONSTRAINT-008", 1, "GET-SETEX wrong", "SET NX EX right"),
    );
    ctrl.seed_synthesis(
        "CONSTRAINT-TAU-1",
        2,
        make_synthesis(
            "CONSTRAINT-TAU-1",
            2,
            "ARGV3-tenant wrong",
            "JWT-tenant right",
        ),
    );

    let syntheses = ctrl.all_domain_syntheses();
    let graph = empty_graph();
    let ctx = build_repair_context(RepairInput {
        prior_proposal_text: "",
        targets: &[],
        zone3_hints: None,
        conflict_graph: &graph,
        retry_count: 1,
        attempts_remaining: 1,
        system_context_with_rubric: "SYS",
        checks: &[],
        partial_passes: &[],
        prior_best_score: None,
        domain_syntheses: &syntheses,
        coupled_constraint_hints: &[],
        passing_constraint_pins: &[],
    });

    // Both corrections must appear — the merge must not drop any entry.
    assert!(ctx.contains("GET-SETEX wrong"), "first correction missing");
    assert!(
        ctx.contains("SET NX EX right"),
        "first correction's fix missing"
    );
    assert!(
        ctx.contains("ARGV3-tenant wrong"),
        "second correction missing"
    );
    assert!(
        ctx.contains("JWT-tenant right"),
        "second correction's fix missing"
    );
}

// ── starting_wave_for_checkpoint ──────────────────────────────────────────────

#[test]
fn starting_wave_for_wave_completed_k() {
    use h2ai_orchestrator::engine::starting_wave_for_checkpoint;
    use h2ai_types::reasoning_checkpoint::ReasoningCheckpointPhase;
    assert_eq!(
        starting_wave_for_checkpoint(&ReasoningCheckpointPhase::WaveCompleted(3)),
        4
    );
    assert_eq!(
        starting_wave_for_checkpoint(&ReasoningCheckpointPhase::WaveCompleted(0)),
        1
    );
}

#[test]
fn starting_wave_for_other_phases_returns_zero() {
    use h2ai_orchestrator::engine::starting_wave_for_checkpoint;
    use h2ai_types::reasoning_checkpoint::ReasoningCheckpointPhase;
    assert_eq!(
        starting_wave_for_checkpoint(&ReasoningCheckpointPhase::Created),
        0
    );
    assert_eq!(
        starting_wave_for_checkpoint(&ReasoningCheckpointPhase::Resolved),
        0
    );
}
