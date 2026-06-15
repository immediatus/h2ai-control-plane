#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use std::collections::HashSet;

use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
use h2ai_orchestrator::coherence::CoherenceState;
use h2ai_orchestrator::phases::llm_coverage;
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::RoleErrorCost;

fn make_constraint(id: &str, domains: &[&str]) -> ConstraintDoc {
    ConstraintDoc {
        id: id.to_string(),
        source_file: "test.md".into(),
        description: format!("Constraint {id}"),
        severity: ConstraintSeverity::Hard { threshold: 0.45 },
        predicate: ConstraintPredicate::LlmJudge {
            rubric: format!("Constraint {id} rubric"),
        },
        remediation_hint: None,
        domains: domains.iter().map(|s| s.to_string()).collect(),
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }
}

fn make_pruned(task_id: &TaskId, violated_ids: &[&str]) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: task_id.clone(),
        explorer_id: ExplorerId::new(),
        reason: "test".into(),
        raw_output: String::new(),
        constraint_error_cost: RoleErrorCost::new(0.5).unwrap(),
        violated_constraints: violated_ids
            .iter()
            .map(|id| ConstraintViolation {
                constraint_id: id.to_string(),
                score: 0.0,
                severity_label: "Hard".into(),
                remediation_hint: None,
                constraint_description: String::new(),
                verifier_reason: None,
                check_verdicts: vec![],
                criteria_pass: None,
                check_reasons: None,
            })
            .collect(),
        timestamp: chrono::Utc::now(),
        retry_count: 0,
        bypass_reason: None,
    }
}

#[test]
fn is_closed_when_no_pruned_proposals() {
    let corpus = vec![
        make_constraint("C1", &["security"]),
        make_constraint("C2", &["performance"]),
    ];
    let state = CoherenceState::from_pruned(&corpus, &[]);
    assert!(state.is_closed());
    assert!(state.uncovered_domains.is_empty());
}

#[test]
fn is_closed_when_pruned_proposals_have_no_violations() {
    let corpus = vec![make_constraint("C1", &["security"])];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &[])];
    let state = CoherenceState::from_pruned(&corpus, &pruned);
    assert!(state.is_closed());
}

#[test]
fn uncovered_domains_from_violated_constraints() {
    let corpus = vec![
        make_constraint("C1", &["security"]),
        make_constraint("C2", &["performance"]),
    ];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C1"])];
    let state = CoherenceState::from_pruned(&corpus, &pruned);
    assert!(!state.is_closed());
    assert!(state.uncovered_domains.contains(&"security".to_string()));
    assert!(!state.uncovered_domains.contains(&"performance".to_string()));
}

#[test]
fn multi_domain_constraint_uncovers_all_its_domains() {
    let corpus = vec![
        make_constraint("C1", &["security", "compliance"]),
        make_constraint("C2", &["performance"]),
    ];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C1"])];
    let state = CoherenceState::from_pruned(&corpus, &pruned);
    assert!(state.uncovered_domains.contains(&"security".to_string()));
    assert!(state.uncovered_domains.contains(&"compliance".to_string()));
    assert!(!state.uncovered_domains.contains(&"performance".to_string()));
}

#[test]
fn uncovered_domains_are_sorted() {
    let corpus = vec![
        make_constraint("C1", &["security"]),
        make_constraint("C2", &["auth"]),
        make_constraint("C3", &["performance"]),
    ];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C1", "C2", "C3"])];
    let state = CoherenceState::from_pruned(&corpus, &pruned);
    let mut sorted = state.uncovered_domains.clone();
    sorted.sort();
    assert_eq!(state.uncovered_domains, sorted);
}

#[test]
fn constraints_not_in_corpus_are_ignored() {
    let corpus = vec![make_constraint("C1", &["security"])];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C-unknown"])];
    let state = CoherenceState::from_pruned(&corpus, &pruned);
    assert!(state.is_closed());
}

// ── with_contradictions ────────────────────────────────────────────────────

fn make_explorers(n: usize) -> Vec<ExplorerId> {
    (0..n).map(|_| ExplorerId::new()).collect()
}

#[test]
fn no_contradictions_when_proposals_agree() {
    let corpus = vec![make_constraint("C1", &["security"])];
    let explorers = make_explorers(2);
    let matrix = vec![vec![0.8], vec![0.9]];
    let state = CoherenceState::from_pruned(&corpus, &[]).with_contradictions(
        &corpus,
        &explorers,
        &matrix,
        &["C1".to_string()],
    );
    assert!(state.active_contradictions.is_empty());
    assert!(state.is_closed());
}

#[test]
fn contradiction_detected_when_proposals_diverge_on_domain() {
    let corpus = vec![make_constraint("C1", &["security"])];
    let explorers = make_explorers(2);
    let matrix = vec![vec![0.8], vec![0.2]];
    let state = CoherenceState::from_pruned(&corpus, &[]).with_contradictions(
        &corpus,
        &explorers,
        &matrix,
        &["C1".to_string()],
    );
    assert_eq!(state.active_contradictions.len(), 1);
    assert_eq!(state.active_contradictions[0].2, "security");
    assert!(!state.is_closed());
}

#[test]
fn contradiction_deduped_per_domain_across_multiple_constraints() {
    let corpus = vec![
        make_constraint("C1", &["security"]),
        make_constraint("C2", &["security"]),
    ];
    let explorers = make_explorers(2);
    let matrix = vec![vec![0.9, 0.8], vec![0.1, 0.2]];
    let state = CoherenceState::from_pruned(&corpus, &[]).with_contradictions(
        &corpus,
        &explorers,
        &matrix,
        &["C1".to_string(), "C2".to_string()],
    );
    assert_eq!(
        state.active_contradictions.len(),
        1,
        "one entry per (pair, domain)"
    );
    assert_eq!(state.active_contradictions[0].2, "security");
}

#[test]
fn no_contradiction_when_both_proposals_fail() {
    let corpus = vec![make_constraint("C1", &["security"])];
    let explorers = make_explorers(2);
    let matrix = vec![vec![0.1], vec![0.2]];
    let state = CoherenceState::from_pruned(&corpus, &[]).with_contradictions(
        &corpus,
        &explorers,
        &matrix,
        &["C1".to_string()],
    );
    assert!(state.active_contradictions.is_empty());
}

#[test]
fn contradiction_across_different_domains_reported_separately() {
    let corpus = vec![
        make_constraint("C1", &["security"]),
        make_constraint("C2", &["performance"]),
    ];
    let explorers = make_explorers(2);
    let matrix = vec![vec![0.9, 0.8], vec![0.1, 0.2]];
    let state = CoherenceState::from_pruned(&corpus, &[]).with_contradictions(
        &corpus,
        &explorers,
        &matrix,
        &["C1".to_string(), "C2".to_string()],
    );
    assert_eq!(state.active_contradictions.len(), 2);
    let domains: Vec<&str> = state
        .active_contradictions
        .iter()
        .map(|c| c.2.as_str())
        .collect();
    assert!(domains.contains(&"security"));
    assert!(domains.contains(&"performance"));
}

#[test]
fn is_closed_requires_both_fields_empty() {
    let corpus = vec![make_constraint("C1", &["security"])];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C1"])];
    let explorers = make_explorers(2);
    let matrix = vec![vec![0.9], vec![0.1]];
    let state = CoherenceState::from_pruned(&corpus, &pruned).with_contradictions(
        &corpus,
        &explorers,
        &matrix,
        &["C1".to_string()],
    );
    assert!(!state.uncovered_domains.is_empty());
    assert!(!state.active_contradictions.is_empty());
    assert!(!state.is_closed());
}

#[test]
fn is_closed_false_when_contradictions_present_but_no_uncovered_domains() {
    let corpus = vec![make_constraint("C1", &["security"])];
    let explorers = make_explorers(2);
    let matrix = vec![vec![0.9], vec![0.1]];
    let state = CoherenceState::from_pruned(&corpus, &[]).with_contradictions(
        &corpus,
        &explorers,
        &matrix,
        &["C1".to_string()],
    );
    // No uncovered domains (nothing pruned), but active contradiction → not closed
    assert!(state.uncovered_domains.is_empty());
    assert!(!state.active_contradictions.is_empty());
    assert!(!state.is_closed());
}

#[test]
fn is_closed_true_at_exact_boundary_scores() {
    // When proposals score exactly at the pass threshold (0.5) they are NOT contradicting
    // each other — both are on the pass side. is_closed() must be true.
    let corpus = vec![make_constraint("C1", &["security"])];
    let explorers = make_explorers(2);
    let matrix = vec![vec![0.5], vec![0.51]]; // both ≥ 0.5: no contradiction
    let state = CoherenceState::from_pruned(&corpus, &[]).with_contradictions(
        &corpus,
        &explorers,
        &matrix,
        &["C1".to_string()],
    );
    assert!(state.uncovered_domains.is_empty());
    assert!(state.active_contradictions.is_empty());
    assert!(state.is_closed());
}

// ── filter_covered_by_survivors ────────────────────────────────────────────

#[test]
fn domain_removed_from_uncovered_when_survivor_covers_it() {
    // Reproduces the GAP-C1 false positive: a proposal is pruned for violating C1
    // (domains=["billing","compliance"]) but the winning proposal correctly handles it.
    let corpus = vec![
        make_constraint("C1", &["billing", "compliance"]),
        make_constraint("C2", &["consistency"]),
    ];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C1"])];
    let base = CoherenceState::from_pruned(&corpus, &pruned);
    // Before filtering: billing and compliance are uncovered
    assert!(base.uncovered_domains.contains(&"billing".to_string()));
    assert!(base.uncovered_domains.contains(&"compliance".to_string()));

    // Survivor matrix: 1 survivor, scores [1.0, 0.8] on [C1, C2]
    let matrix = vec![vec![1.0_f64, 0.8]];
    let filtered =
        base.filter_covered_by_survivors(&corpus, &matrix, &["C1".to_string(), "C2".to_string()]);
    // Survivor scores 1.0 on C1 → billing and compliance are now covered → removed
    assert!(
        !filtered.uncovered_domains.contains(&"billing".to_string()),
        "billing should be covered by survivor"
    );
    assert!(
        !filtered
            .uncovered_domains
            .contains(&"compliance".to_string()),
        "compliance should be covered by survivor"
    );
    assert!(filtered.is_closed());
}

#[test]
fn domain_stays_uncovered_when_no_survivor_passes_threshold() {
    let corpus = vec![make_constraint("C1", &["billing"])];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C1"])];
    let base = CoherenceState::from_pruned(&corpus, &pruned);

    // Survivor scores 0.3 on C1 — below 0.5 threshold
    let matrix = vec![vec![0.3_f64]];
    let filtered = base.filter_covered_by_survivors(&corpus, &matrix, &["C1".to_string()]);
    assert!(
        filtered.uncovered_domains.contains(&"billing".to_string()),
        "billing still uncovered"
    );
    assert!(!filtered.is_closed());
}

#[test]
fn partial_cover_removes_only_covered_domains() {
    // C1 has domains [billing, compliance], C2 has domain [consistency]
    // Survivor passes C1 (billing+compliance covered) but fails C2 (consistency uncovered)
    let corpus = vec![
        make_constraint("C1", &["billing", "compliance"]),
        make_constraint("C2", &["consistency"]),
    ];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C1", "C2"])];
    let base = CoherenceState::from_pruned(&corpus, &pruned);

    let matrix = vec![vec![0.9_f64, 0.2]]; // passes C1, fails C2
    let filtered =
        base.filter_covered_by_survivors(&corpus, &matrix, &["C1".to_string(), "C2".to_string()]);
    assert!(!filtered.uncovered_domains.contains(&"billing".to_string()));
    assert!(!filtered
        .uncovered_domains
        .contains(&"compliance".to_string()));
    assert!(
        filtered
            .uncovered_domains
            .contains(&"consistency".to_string()),
        "consistency still uncovered"
    );
}

#[test]
fn empty_matrix_leaves_uncovered_domains_unchanged() {
    let corpus = vec![make_constraint("C1", &["billing"])];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C1"])];
    let base = CoherenceState::from_pruned(&corpus, &pruned);

    let filtered = base.filter_covered_by_survivors(&corpus, &[], &["C1".to_string()]);
    assert!(filtered.uncovered_domains.contains(&"billing".to_string()));
}

// ── subtract_covered_domains ───────────────────────────────────────────────

fn make_state_with_uncovered(uncovered: &[&str]) -> CoherenceState {
    CoherenceState {
        uncovered_domains: uncovered.iter().map(|s| s.to_string()).collect(),
        active_contradictions: vec![],
    }
}

#[test]
fn subtract_removes_matching_domain() {
    let state = make_state_with_uncovered(&["billing", "compliance"]);
    let covered: HashSet<String> = HashSet::from(["billing".to_string()]);
    let result = state.subtract_covered_domains(&covered);
    assert_eq!(result.uncovered_domains, vec!["compliance".to_string()]);
}

#[test]
fn subtract_removes_all_when_fully_covered() {
    let state = make_state_with_uncovered(&["billing", "compliance"]);
    let covered: HashSet<String> = HashSet::from(["billing".to_string(), "compliance".to_string()]);
    let result = state.subtract_covered_domains(&covered);
    assert!(result.uncovered_domains.is_empty());
}

#[test]
fn subtract_leaves_non_matching_intact() {
    let state = make_state_with_uncovered(&["billing"]);
    let covered: HashSet<String> = HashSet::from(["audit".to_string()]);
    let result = state.subtract_covered_domains(&covered);
    assert_eq!(result.uncovered_domains, vec!["billing".to_string()]);
}

#[test]
fn subtract_empty_set_is_noop() {
    let state = make_state_with_uncovered(&["billing", "compliance"]);
    let covered: HashSet<String> = HashSet::new();
    let result = state.subtract_covered_domains(&covered);
    assert_eq!(
        result.uncovered_domains,
        vec!["billing".to_string(), "compliance".to_string()]
    );
}

#[test]
fn subtract_on_empty_uncovered_is_noop() {
    let state = make_state_with_uncovered(&[]);
    let covered: HashSet<String> = HashSet::from(["billing".to_string()]);
    let result = state.subtract_covered_domains(&covered);
    assert!(result.uncovered_domains.is_empty());
}

#[test]
fn subtract_does_not_touch_active_contradictions() {
    let corpus = vec![make_constraint("C1", &["billing"])];
    let explorers = make_explorers(2);
    let matrix = vec![vec![0.9], vec![0.1]];
    let state = CoherenceState {
        uncovered_domains: vec!["billing".to_string()],
        active_contradictions: vec![],
    }
    .with_contradictions(&corpus, &explorers, &matrix, &["C1".to_string()]);

    assert_eq!(state.active_contradictions.len(), 1);

    let covered: HashSet<String> = HashSet::from(["billing".to_string()]);
    let result = state.subtract_covered_domains(&covered);
    assert!(result.uncovered_domains.is_empty());
    assert_eq!(
        result.active_contradictions.len(),
        1,
        "contradictions must be untouched"
    );
}

#[test]
fn subtract_makes_is_closed_true_when_both_fields_clear() {
    let state = make_state_with_uncovered(&["billing"]);
    assert!(!state.is_closed(), "should be open before subtraction");
    let covered: HashSet<String> = HashSet::from(["billing".to_string()]);
    let result = state.subtract_covered_domains(&covered);
    assert!(
        result.is_closed(),
        "should be closed after all domains removed"
    );
}

// ── GAP-C1 composition / regression tests ─────────────────────────────────

fn make_soft_constraint(id: &str, domains: &[&str]) -> ConstraintDoc {
    let mut c = ConstraintDoc::new_soft_llm_judge(id, "rubric");
    c.domains = domains.iter().map(|s| s.to_string()).collect();
    c
}

/// GAP-C1 core regression: Hard constraint, all branches pruned, survivor clears false positive.
///
/// Before the GAP-C1 fix, `from_pruned` would mark "billing" and "compliance" as uncovered
/// even when surviving proposals correctly handled C1. `llm_coverage::run` + `subtract_covered_domains`
/// clears the false positive when survivor_count >= 1.
#[test]
fn gap_c1_false_positive_cleared_by_survivor() {
    let corpus = vec![make_constraint("C1", &["billing", "compliance"])];
    let task_id = TaskId::new();
    let pruned = vec![
        make_pruned(&task_id, &["C1"]),
        make_pruned(&task_id, &["C1"]),
    ];
    let state = CoherenceState::from_pruned(&corpus, &pruned);
    // Both "billing" and "compliance" are initially marked uncovered (false positive)
    assert!(state.uncovered_domains.contains(&"billing".to_string()));
    assert!(state.uncovered_domains.contains(&"compliance".to_string()));

    let llm_out = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    let covered: HashSet<String> = llm_out.covered_domains.into_iter().collect();
    // llm_coverage should have covered both domains from the Hard constraint
    assert!(covered.contains("billing"));
    assert!(covered.contains("compliance"));

    let final_state = state.subtract_covered_domains(&covered);
    assert!(
        final_state.uncovered_domains.is_empty(),
        "all domains should be cleared when survivor covers them"
    );
    assert!(final_state.is_closed());
}

/// GAP-C1 boundary: zero survivors means no domains are covered — false positive stays.
#[test]
fn gap_c1_stays_open_with_no_survivors() {
    let corpus = vec![make_constraint("C1", &["billing", "compliance"])];
    let task_id = TaskId::new();
    let pruned = vec![
        make_pruned(&task_id, &["C1"]),
        make_pruned(&task_id, &["C1"]),
    ];
    let state = CoherenceState::from_pruned(&corpus, &pruned);

    let llm_out = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 0,
        bypassed_ids: &HashSet::new(),
    });
    assert!(
        llm_out.covered_domains.is_empty(),
        "zero survivors → no covered domains"
    );

    let covered: HashSet<String> = llm_out.covered_domains.into_iter().collect();
    let final_state = state.subtract_covered_domains(&covered);
    assert!(
        !final_state.uncovered_domains.is_empty(),
        "domains must remain uncovered when no survivor exists"
    );
    assert!(!final_state.is_closed());
}

/// GAP-C1 design boundary: Soft constraints are excluded from llm_coverage by design.
/// Even when survivors exist, a Soft-domain violation is NOT cleared by subtract_covered_domains.
#[test]
fn soft_violated_domain_stays_uncovered_by_design() {
    let corpus = vec![make_soft_constraint("C2", &["audit"])];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C2"])];
    let state = CoherenceState::from_pruned(&corpus, &pruned);
    assert!(
        state.uncovered_domains.contains(&"audit".to_string()),
        "audit should be initially uncovered"
    );

    // llm_coverage excludes Soft constraints, so covered_domains is empty even with survivors
    let llm_out = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert!(
        llm_out.covered_domains.is_empty(),
        "Soft constraint domains must not appear in llm_coverage output"
    );

    let covered: HashSet<String> = llm_out.covered_domains.into_iter().collect();
    let final_state = state.subtract_covered_domains(&covered);
    assert!(
        final_state.uncovered_domains.contains(&"audit".to_string()),
        "audit must remain uncovered — Soft domains are not cleared by llm_coverage"
    );
    assert!(!final_state.is_closed());
}

/// GAP-C1 partial clear: Hard domain cleared by survivor, Soft domain stays uncovered.
#[test]
fn hard_covered_soft_uncovered_partial_clear() {
    let corpus = vec![
        make_constraint("C1", &["billing"]),
        make_soft_constraint("C2", &["audit"]),
    ];
    let task_id = TaskId::new();
    let pruned = vec![make_pruned(&task_id, &["C1", "C2"])];
    let state = CoherenceState::from_pruned(&corpus, &pruned);
    // Both domains initially uncovered
    assert!(state.uncovered_domains.contains(&"audit".to_string()));
    assert!(state.uncovered_domains.contains(&"billing".to_string()));

    // llm_coverage only covers Hard domains → ["billing"]
    let llm_out = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert_eq!(llm_out.covered_domains, vec!["billing".to_string()]);

    let covered: HashSet<String> = llm_out.covered_domains.into_iter().collect();
    let final_state = state.subtract_covered_domains(&covered);
    // "billing" removed (Hard, covered), "audit" remains (Soft, not cleared)
    assert_eq!(
        final_state.uncovered_domains,
        vec!["audit".to_string()],
        "only Hard domain should be cleared"
    );
    assert!(!final_state.is_closed());
}

/// GAP-C1 invariant: subtract_covered_domains never touches active_contradictions.
#[test]
fn full_chain_subtract_does_not_clear_active_contradictions() {
    let corpus = vec![make_constraint("C1", &["billing"])];
    let explorers = make_explorers(2);
    // Survivor matrix: explorer 0 passes C1, explorer 1 fails → contradiction on "billing"
    let matrix = vec![vec![0.9_f64], vec![0.1]];
    let state = CoherenceState {
        uncovered_domains: vec!["billing".to_string()],
        active_contradictions: vec![],
    }
    .with_contradictions(&corpus, &explorers, &matrix, &["C1".to_string()]);

    assert_eq!(
        state.active_contradictions.len(),
        1,
        "should have one contradiction before subtract"
    );
    assert!(state.uncovered_domains.contains(&"billing".to_string()));

    let llm_out = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    let covered: HashSet<String> = llm_out.covered_domains.into_iter().collect();
    let final_state = state.subtract_covered_domains(&covered);

    assert!(
        final_state.uncovered_domains.is_empty(),
        "billing should be removed from uncovered_domains"
    );
    assert_eq!(
        final_state.active_contradictions.len(),
        1,
        "contradictions must not be touched by subtract_covered_domains"
    );
    // Contradiction is still present → not closed
    assert!(!final_state.is_closed());
}
