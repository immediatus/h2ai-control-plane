use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
use h2ai_orchestrator::coherence::CoherenceState;
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
    }
}

fn make_pruned(task_id: &TaskId, violated_ids: &[&str]) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: task_id.clone(),
        explorer_id: ExplorerId::new(),
        reason: "test".into(),
        constraint_error_cost: RoleErrorCost::new(0.5).unwrap(),
        violated_constraints: violated_ids
            .iter()
            .map(|id| ConstraintViolation {
                constraint_id: id.to_string(),
                score: 0.0,
                severity_label: "Hard".into(),
                remediation_hint: None,
            })
            .collect(),
        timestamp: chrono::Utc::now(),
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

fn make_explorers(n: usize) -> Vec<h2ai_types::identity::ExplorerId> {
    (0..n)
        .map(|_| h2ai_types::identity::ExplorerId::new())
        .collect()
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
