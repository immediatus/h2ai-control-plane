use h2ai_autonomic::repair::select_orthogonal_partials;
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation};

fn make_pruned_with_violations(violations: Vec<ConstraintViolation>) -> BranchPrunedEvent {
    use chrono::Utc;
    use h2ai_types::identity::{ExplorerId, TaskId};
    use h2ai_types::sizing::RoleErrorCost;
    BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: "a".repeat(100),
        raw_output: String::new(),
        constraint_error_cost: RoleErrorCost::new(0.0).unwrap(),
        violated_constraints: violations,
        timestamp: Utc::now(),
    }
}

fn make_violation(desc: &str) -> ConstraintViolation {
    ConstraintViolation {
        constraint_id: format!("C-{}", desc),
        score: 0.0,
        severity_label: "Hard".into(),
        remediation_hint: None,
        constraint_description: desc.to_string(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
    }
}

fn checks(descriptions: &[&str]) -> Vec<String> {
    descriptions.iter().map(|s| s.to_string()).collect()
}

/// Build 1-check-per-constraint offsets whose constraint_ids match make_violation("desc") → "C-{desc}".
fn offsets_for(descriptions: &[&str]) -> Vec<(String, usize, usize)> {
    descriptions
        .iter()
        .enumerate()
        .map(|(i, d)| (format!("C-{d}"), i, 1))
        .collect()
}

#[test]
fn test_orthogonal_empty_input() {
    let result = select_orthogonal_partials(
        &[],
        &checks(&["check A", "check B"]),
        &offsets_for(&["check A", "check B"]),
        2,
        usize::MAX,
    );
    assert!(result.is_empty());
}

#[test]
fn test_orthogonal_all_failed_excluded() {
    // All checks violated = no partial passes.
    let pruned = vec![make_pruned_with_violations(vec![
        make_violation("check A"),
        make_violation("check B"),
    ])];
    let result = select_orthogonal_partials(
        &pruned,
        &checks(&["check A", "check B"]),
        &offsets_for(&["check A", "check B"]),
        2,
        usize::MAX,
    );
    assert!(result.is_empty());
}

#[test]
fn test_orthogonal_single_candidate_returned() {
    // Violates only "check B" → passes "check A"
    let pruned = vec![make_pruned_with_violations(vec![make_violation("check B")])];
    let result = select_orthogonal_partials(
        &pruned,
        &checks(&["check A", "check B"]),
        &offsets_for(&["check A", "check B"]),
        2,
        usize::MAX,
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].passed_count(), 1);
}

#[test]
fn test_orthogonal_max_k_respected() {
    let checks_list = checks(&["A", "B", "C", "D"]);
    let pruned: Vec<BranchPrunedEvent> = (0..5)
        .map(|_| make_pruned_with_violations(vec![make_violation("A")]))
        .collect();
    let result = select_orthogonal_partials(
        &pruned,
        &checks_list,
        &offsets_for(&["A", "B", "C", "D"]),
        2,
        usize::MAX,
    );
    assert!(result.len() <= 2);
}

#[test]
fn test_orthogonal_return_order_widest_first() {
    // p1 passes checks A and B (wider), p2 passes only C (narrower).
    let p1 = make_pruned_with_violations(vec![make_violation("C")]);
    let p2 = make_pruned_with_violations(vec![make_violation("A"), make_violation("B")]);
    let pruned = vec![p1, p2];
    let result = select_orthogonal_partials(
        &pruned,
        &checks(&["A", "B", "C"]),
        &offsets_for(&["A", "B", "C"]),
        3,
        usize::MAX,
    );
    assert!(!result.is_empty());
    // First result must have the most passed checks.
    assert!(result[0].passed_count() >= result.last().unwrap().passed_count());
}

#[test]
fn test_orthogonal_prefers_coverage_diversity() {
    // A passes {0,1}, B passes {2}, C passes {0,1} (duplicate of A).
    // With k=2: should pick A then B (covers all 3), not A then C.
    let checks_list = checks(&["X0", "X1", "X2"]);
    let a = make_pruned_with_violations(vec![make_violation("X2")]); // passes X0, X1
    let b = make_pruned_with_violations(vec![make_violation("X0"), make_violation("X1")]); // passes X2
    let c = make_pruned_with_violations(vec![make_violation("X2")]); // same as A
    let pruned = vec![a, b, c];
    let result = select_orthogonal_partials(
        &pruned,
        &checks_list,
        &offsets_for(&["X0", "X1", "X2"]),
        2,
        usize::MAX,
    );
    assert_eq!(result.len(), 2);
    let covered: std::collections::HashSet<usize> = result
        .iter()
        .flat_map(|p| p.passed_check_indices())
        .collect();
    assert_eq!(covered.len(), 3, "should cover all 3 checks");
}
