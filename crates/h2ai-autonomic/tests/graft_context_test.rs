use h2ai_autonomic::repair::{build_graft_context, missing_constraint_ids, GraftInput};

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
