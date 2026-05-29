use h2ai_autonomic::repair::{
    build_graft_context, graft_is_redundant, graft_token_projection_exceeds,
    grafted_ids_cycle_detected, missing_constraint_ids, GraftInput, PartialPass,
};

#[test]
fn graft_context_contains_base_text() {
    let input = GraftInput {
        base_text: "BASE PROPOSAL",
        candidate_text: "CANDIDATE TEXT",
        constraint_ids: &["C-001".to_string(), "C-002".to_string()],
        system_context: "SYSTEM",
    };
    let out = build_graft_context(&input);
    assert!(
        out.contains("BASE PROPOSAL"),
        "base text must appear in graft context"
    );
}

#[test]
fn graft_context_contains_candidate_text() {
    let input = GraftInput {
        base_text: "base",
        candidate_text: "CANDIDATE TEXT",
        constraint_ids: &["C-003".to_string()],
        system_context: "SYS",
    };
    let out = build_graft_context(&input);
    assert!(out.contains("CANDIDATE TEXT"), "candidate text must appear");
}

#[test]
fn graft_context_contains_all_constraint_ids() {
    let input = GraftInput {
        base_text: "b",
        candidate_text: "c",
        constraint_ids: &["C-001".to_string(), "C-999".to_string()],
        system_context: "SYS",
    };
    let out = build_graft_context(&input);
    assert!(out.contains("C-001"), "first constraint id must appear");
    assert!(out.contains("C-999"), "second constraint id must appear");
}

#[test]
fn graft_context_starts_with_system_context() {
    let input = GraftInput {
        base_text: "b",
        candidate_text: "c",
        constraint_ids: &["C-001".to_string()],
        system_context: "MY_SYSTEM_CTX",
    };
    let out = build_graft_context(&input);
    assert!(
        out.starts_with("MY_SYSTEM_CTX"),
        "system context must be first"
    );
}

#[test]
fn graft_context_contains_graft_directive() {
    let input = GraftInput {
        base_text: "b",
        candidate_text: "c",
        constraint_ids: &["C-001".to_string()],
        system_context: "SYS",
    };
    let out = build_graft_context(&input);
    assert!(
        out.contains("GRAFT STEP"),
        "graft step header must appear in context"
    );
}

#[test]
fn graft_context_empty_constraint_ids_still_renders() {
    let input = GraftInput {
        base_text: "b",
        candidate_text: "c",
        constraint_ids: &[],
        system_context: "SYS",
    };
    let out = build_graft_context(&input);
    assert!(out.contains("GRAFT STEP"));
    assert!(out.contains("b"));
    assert!(out.contains("c"));
}

#[test]
fn missing_constraint_ids_returns_ids_in_candidate_not_in_base() {
    let base = h2ai_autonomic::repair::PartialPass {
        proposal_text: String::new(),
        check_results: vec![
            (0, "Check A".to_string(), true),
            (1, "Check B".to_string(), false),
        ],
        score: 0.5,
    };
    let candidate = h2ai_autonomic::repair::PartialPass {
        proposal_text: String::new(),
        check_results: vec![
            (0, "Check A".to_string(), false),
            (1, "Check B".to_string(), true),
        ],
        score: 0.5,
    };
    // offsets: check index 0 belongs to "C-001", check index 1 belongs to "C-002"
    let offsets = vec![
        ("C-001".to_string(), 0usize, 1usize),
        ("C-002".to_string(), 1usize, 1usize),
    ];
    let missing = missing_constraint_ids(&base, &candidate, &offsets);
    assert_eq!(missing, vec!["C-002".to_string()]);
}

#[test]
fn missing_constraint_ids_empty_when_base_covers_all() {
    let base = h2ai_autonomic::repair::PartialPass {
        proposal_text: String::new(),
        check_results: vec![
            (0, "Check A".to_string(), true),
            (1, "Check B".to_string(), true),
        ],
        score: 1.0,
    };
    let candidate = h2ai_autonomic::repair::PartialPass {
        proposal_text: String::new(),
        check_results: vec![
            (0, "Check A".to_string(), true),
            (1, "Check B".to_string(), true),
        ],
        score: 1.0,
    };
    let offsets = vec![
        ("C-001".to_string(), 0usize, 1usize),
        ("C-002".to_string(), 1usize, 1usize),
    ];
    assert!(missing_constraint_ids(&base, &candidate, &offsets).is_empty());
}

#[test]
fn missing_constraint_ids_only_includes_constraints_candidate_actually_passes() {
    let base = h2ai_autonomic::repair::PartialPass {
        proposal_text: String::new(),
        check_results: vec![(0, "Check A".to_string(), false)],
        score: 0.0,
    };
    let candidate = h2ai_autonomic::repair::PartialPass {
        proposal_text: String::new(),
        check_results: vec![(0, "Check A".to_string(), false)],
        score: 0.0,
    };
    let offsets = vec![("C-001".to_string(), 0usize, 1usize)];
    assert!(
        missing_constraint_ids(&base, &candidate, &offsets).is_empty(),
        "candidate must pass the check for it to count as coverage"
    );
}

#[test]
fn missing_constraint_ids_cluster_covered_when_any_check_passes() {
    let base = h2ai_autonomic::repair::PartialPass {
        proposal_text: String::new(),
        check_results: vec![
            (0, "a".to_string(), false),
            (1, "b".to_string(), false),
            (2, "c".to_string(), false),
        ],
        score: 0.0,
    };
    let candidate = h2ai_autonomic::repair::PartialPass {
        proposal_text: String::new(),
        check_results: vec![
            (0, "a".to_string(), false),
            (1, "b".to_string(), true), // only check 1 passes
            (2, "c".to_string(), false),
        ],
        score: 0.33,
    };
    // One constraint spanning checks 0..3; candidate passes only check 1.
    let offsets = vec![("C-001".to_string(), 0usize, 3usize)];
    let result = missing_constraint_ids(&base, &candidate, &offsets);
    assert_eq!(result, vec!["C-001".to_string()]);
}

fn partial(checks: Vec<(usize, bool)>) -> PartialPass {
    PartialPass {
        proposal_text: "x".repeat(200),
        check_results: checks
            .into_iter()
            .map(|(i, p)| (i, "txt".to_string(), p))
            .collect(),
        score: 0.5,
    }
}

// ── graft_is_redundant ───────────────────────────────────────────────────

#[test]
fn graft_is_redundant_true_when_high_overlap() {
    // base passes 0,1,2,3 ; candidate covers 0,1,2,3,4
    // shared/union = 4/5 = 0.80 > 0.60 → redundant
    let base = partial(vec![(0, true), (1, true), (2, true), (3, true)]);
    let candidate = partial(vec![(0, true), (1, true), (2, true), (3, true), (4, false)]);
    assert!(
        graft_is_redundant(&base, &candidate, 0.6),
        "high overlap must be detected as redundant"
    );
}

#[test]
fn graft_is_redundant_false_when_low_overlap() {
    // base passes 0,1 ; candidate covers 2,3,4 → shared=0, union=5 → 0.0 ≤ 0.6
    let base = partial(vec![(0, true), (1, true)]);
    let candidate = partial(vec![(2, false), (3, false), (4, false)]);
    assert!(
        !graft_is_redundant(&base, &candidate, 0.6),
        "low overlap must not be flagged as redundant"
    );
}

// ── grafted_ids_cycle_detected ───────────────────────────────────────────

#[test]
fn cycle_detected_when_missing_all_already_grafted() {
    let already_grafted: std::collections::HashSet<String> =
        ["C-001", "C-002"].iter().map(|s| s.to_string()).collect();
    let missing = vec!["C-001".to_string(), "C-002".to_string()];
    assert!(
        grafted_ids_cycle_detected(&missing, &already_grafted),
        "all missing IDs already grafted → cycle"
    );
}

#[test]
fn no_cycle_when_missing_has_new_ids() {
    let already_grafted: std::collections::HashSet<String> =
        ["C-001"].iter().map(|s| s.to_string()).collect();
    let missing = vec!["C-001".to_string(), "C-003".to_string()];
    assert!(
        !grafted_ids_cycle_detected(&missing, &already_grafted),
        "at least one new ID → no cycle"
    );
}

// ── graft_token_projection_exceeds ───────────────────────────────────────

#[test]
fn token_projection_exceeds_when_texts_too_large() {
    // base = 400 chars (~100 tokens), candidate = 400 chars (~100 tokens)
    // projected = 200 tokens; 200 > 100 * 1.3 = 130 → true
    let base_text = "a".repeat(400);
    let candidate_text = "b".repeat(400);
    assert!(
        graft_token_projection_exceeds(&base_text, &candidate_text, 1.3),
        "large candidate must trigger token projection guard"
    );
}

#[test]
fn token_projection_passes_when_texts_small() {
    // base = 400 chars (~100 tokens), candidate = 40 chars (~10 tokens)
    // projected = 110 tokens; 110 ≤ 100 * 1.3 = 130 → false
    let base_text = "a".repeat(400);
    let candidate_text = "b".repeat(40);
    assert!(
        !graft_token_projection_exceeds(&base_text, &candidate_text, 1.3),
        "small candidate must not trigger token projection guard"
    );
}
