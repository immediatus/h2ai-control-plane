use chrono::Utc;
use h2ai_autonomic::repair::{
    build_integration_wave_context, find_oscillation_pairs, seed_for_cluster, PartialPass,
    SolverOutput,
};
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::RoleErrorCost;

fn pruned(constraint_ids: &[&str], wave: u32, raw: &str) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: "test".to_owned(),
        raw_output: raw.to_owned(),
        constraint_error_cost: RoleErrorCost::new(0.0).unwrap(),
        violated_constraints: constraint_ids
            .iter()
            .map(|id| ConstraintViolation {
                constraint_id: id.to_string(),
                score: 0.0,
                severity_label: "Hard".to_owned(),
                remediation_hint: None,
                constraint_description: String::new(),
                verifier_reason: None,
                check_verdicts: vec![],
                criteria_pass: None,
                check_reasons: None,
            })
            .collect(),
        timestamp: Utc::now(),
        retry_count: wave,
        bypass_reason: None,
    }
}

fn partial(proposal: &str, checks: &[(usize, &str, bool)]) -> PartialPass {
    PartialPass {
        proposal_text: proposal.to_owned(),
        check_results: checks
            .iter()
            .map(|(i, s, p)| (*i, s.to_string(), *p))
            .collect(),
        score: {
            let passed = checks.iter().filter(|c| c.2).count();
            let total = checks.len().max(1);
            passed as f64 / total as f64
        },
    }
}

// ── find_oscillation_pairs ────────────────────────────────────────────────────

#[test]
fn no_pairs_when_pruned_is_empty() {
    let pairs = find_oscillation_pairs(&[], &[]);
    assert!(pairs.is_empty());
}

#[test]
fn no_pairs_when_only_one_wave() {
    let events = vec![pruned(&["C-001"], 0, "")];
    let pairs = find_oscillation_pairs(&events, &[]);
    assert!(pairs.is_empty());
}

#[test]
fn oscillation_detected_c1_fixed_at_n1_but_broken_at_n2() {
    // Wave 0: C-001 fails, C-002 passes
    // Wave 1: C-001 passes, C-002 fails
    // Wave 2: C-001 fails again → (C-001, C-002) oscillation
    let events = vec![
        pruned(&["C-001"], 0, ""),
        pruned(&["C-002"], 1, ""),
        pruned(&["C-001"], 2, ""),
    ];
    let pairs = find_oscillation_pairs(&events, &[]);
    let found = pairs
        .iter()
        .any(|(a, b)| (a == "C-001" && b == "C-002") || (a == "C-002" && b == "C-001"));
    assert!(found, "expected (C-001, C-002) pair, got: {:?}", pairs);
}

#[test]
fn no_oscillation_when_constraint_fails_consistently() {
    let events = vec![
        pruned(&["C-001"], 0, ""),
        pruned(&["C-001"], 1, ""),
        pruned(&["C-001"], 2, ""),
    ];
    let pairs = find_oscillation_pairs(&events, &[]);
    let involves = pairs.iter().any(|(a, b)| a == "C-001" || b == "C-001");
    assert!(
        !involves,
        "consistently failing constraint is not oscillating"
    );
}

#[test]
fn no_duplicate_pairs_in_output() {
    let events = vec![
        pruned(&["C-001"], 0, ""),
        pruned(&["C-002"], 1, ""),
        pruned(&["C-001"], 2, ""),
        pruned(&["C-002"], 3, ""),
        pruned(&["C-001"], 4, ""),
    ];
    let pairs = find_oscillation_pairs(&events, &[]);
    let count = pairs
        .iter()
        .filter(|(a, b)| (a == "C-001" && b == "C-002") || (a == "C-002" && b == "C-001"))
        .count();
    assert_eq!(count, 1, "same pair must appear only once");
}

// ── seed_for_cluster ─────────────────────────────────────────────────────────

#[test]
fn seed_returns_none_when_no_partials() {
    let result = seed_for_cluster(&[0, 1], &[]);
    assert!(result.is_none());
}

#[test]
fn seed_returns_partial_with_most_overlap_with_cluster_indices() {
    let cluster_indices = vec![0usize, 1];
    let p1 = partial("A", &[(0, "c0", true), (1, "c1", false), (2, "c2", false)]);
    let p2 = partial("B", &[(0, "c0", true), (1, "c1", true), (2, "c2", false)]);
    let result = seed_for_cluster(&cluster_indices, &[p1, p2]).unwrap();
    assert_eq!(result.proposal_text, "B");
}

#[test]
fn seed_returns_none_when_no_partial_covers_cluster() {
    let cluster_indices = vec![5usize];
    let p1 = partial("A", &[(0, "c", true), (1, "d", true)]);
    assert!(seed_for_cluster(&cluster_indices, &[p1]).is_none());
}

// ── build_integration_wave_context ───────────────────────────────────────────

#[test]
fn integration_context_starts_with_system_context() {
    let outputs = vec![SolverOutput {
        cluster_ids: vec!["C-001".to_owned()],
        proposal_text: "proposal text".to_owned(),
        seed_passed_checks: vec![],
    }];
    let ctx = build_integration_wave_context("SYSTEM_CTX", "", &outputs, 1, &[]);
    assert!(ctx.starts_with("SYSTEM_CTX"));
}

#[test]
fn integration_context_contains_integration_wave_header() {
    let outputs = vec![SolverOutput {
        cluster_ids: vec!["C-001".to_owned()],
        proposal_text: "proposal text".to_owned(),
        seed_passed_checks: vec![],
    }];
    let ctx = build_integration_wave_context("SYS", "", &outputs, 1, &[]);
    assert!(
        ctx.contains("INTEGRATION WAVE"),
        "must embed INTEGRATION_WAVE_PROMPT body"
    );
}

#[test]
fn integration_context_embeds_balancing_instruction_when_non_empty() {
    let outputs = vec![SolverOutput {
        cluster_ids: vec!["C-001".to_owned()],
        proposal_text: "proposal".to_owned(),
        seed_passed_checks: vec![],
    }];
    let ctx = build_integration_wave_context("SYS", "OSCILLATION: C-001 ↔ C-002", &outputs, 1, &[]);
    assert!(ctx.contains("OSCILLATION: C-001"));
}

#[test]
fn integration_context_omits_balancing_block_when_empty() {
    let outputs = vec![SolverOutput {
        cluster_ids: vec!["C-001".to_owned()],
        proposal_text: "proposal".to_owned(),
        seed_passed_checks: vec![],
    }];
    let ctx = build_integration_wave_context("SYS", "", &outputs, 1, &[]);
    assert!(
        !ctx.contains("META-OBSERVER"),
        "no meta-observer block when instruction is empty"
    );
}

#[test]
fn integration_context_includes_cluster_labels_and_proposals() {
    let outputs = vec![
        SolverOutput {
            cluster_ids: vec!["C-001".to_owned(), "C-005".to_owned()],
            proposal_text: "alpha solution".to_owned(),
            seed_passed_checks: vec![0, 1],
        },
        SolverOutput {
            cluster_ids: vec!["C-TAU-1".to_owned()],
            proposal_text: "beta solution".to_owned(),
            seed_passed_checks: vec![],
        },
    ];
    let ctx = build_integration_wave_context("SYS", "", &outputs, 2, &[]);
    assert!(ctx.contains("alpha solution"));
    assert!(ctx.contains("beta solution"));
    assert!(ctx.contains("C-001"));
    assert!(ctx.contains("C-TAU-1"));
}

#[test]
fn integration_context_constraint_count_substituted() {
    let outputs = vec![SolverOutput {
        cluster_ids: vec!["C-001".to_owned()],
        proposal_text: "p".to_owned(),
        seed_passed_checks: vec![],
    }];
    let ctx = build_integration_wave_context("SYS", "", &outputs, 7, &[]);
    assert!(ctx.contains('7'), "constraint_count=7 must appear");
}

#[test]
fn integration_context_embeds_binary_check_texts_per_cluster() {
    let outputs = vec![
        SolverOutput {
            cluster_ids: vec!["C-004".to_owned()],
            proposal_text: "idempotency solution".to_owned(),
            seed_passed_checks: vec![],
        },
        SolverOutput {
            cluster_ids: vec!["C-005".to_owned()],
            proposal_text: "audit solution".to_owned(),
            seed_passed_checks: vec![],
        },
    ];
    let checks = vec![
        (
            "C-004".to_owned(),
            vec![
                "Uses SETNX for atomic key acquisition.".to_owned(),
                "Lua script performs single atomic compare-and-set.".to_owned(),
            ],
        ),
        (
            "C-005".to_owned(),
            vec!["Audit log is append-only (no deletes or updates).".to_owned()],
        ),
    ];
    let ctx = build_integration_wave_context("SYS", "", &outputs, 2, &checks);
    assert!(ctx.contains("Uses SETNX"), "C-004 check text must appear");
    assert!(
        ctx.contains("Lua script performs single atomic"),
        "C-004 check[2] must appear"
    );
    assert!(ctx.contains("append-only"), "C-005 check text must appear");
    assert!(ctx.contains("Required binary checks"), "header must appear");
    assert!(
        ctx.find("Uses SETNX").unwrap() < ctx.find("idempotency solution").unwrap(),
        "checks must precede proposal text"
    );
}
