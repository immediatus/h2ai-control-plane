use chrono::Utc;
use h2ai_autonomic::meta_observer::{
    build_balancing_instruction, divergence_events_from_pruned, sharpen_balancing_instruction,
    wave_violation_history_from_pruned, DivergenceEvent,
};
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::RoleErrorCost;

fn pruned(constraint_ids: &[&str], wave: u32) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: "test".to_owned(),
        raw_output: String::new(),
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

#[test]
fn empty_pruned_yields_empty_history() {
    let history = wave_violation_history_from_pruned(&[]);
    assert!(history.is_empty());
}

#[test]
fn single_event_maps_constraint_to_its_wave() {
    let events = vec![pruned(&["C-001"], 0)];
    let history = wave_violation_history_from_pruned(&events);
    assert_eq!(history.get("C-001"), Some(&vec![0u32]));
}

#[test]
fn same_constraint_violated_at_multiple_waves_collects_all() {
    let events = vec![
        pruned(&["C-001"], 0),
        pruned(&["C-001"], 2),
        pruned(&["C-001"], 4),
    ];
    let history = wave_violation_history_from_pruned(&events);
    let mut waves = history["C-001"].clone();
    waves.sort();
    assert_eq!(waves, vec![0, 2, 4]);
}

#[test]
fn multiple_constraints_in_same_event_all_recorded() {
    let events = vec![pruned(&["C-001", "C-002"], 1)];
    let history = wave_violation_history_from_pruned(&events);
    assert!(history.contains_key("C-001"));
    assert!(history.contains_key("C-002"));
}

#[test]
fn no_divergence_when_constraint_fails_across_all_waves() {
    let events = vec![
        pruned(&["C-001"], 0),
        pruned(&["C-001"], 1),
        pruned(&["C-001"], 2),
    ];
    let divergences = divergence_events_from_pruned(&events);
    assert!(
        divergences.is_empty(),
        "always-failing constraint is not a divergence"
    );
}

#[test]
fn divergence_detected_when_new_constraint_fails_at_later_wave() {
    let events = vec![pruned(&["C-001"], 0), pruned(&["C-001", "C-002"], 1)];
    let divergences = divergence_events_from_pruned(&events);
    assert_eq!(divergences.len(), 1);
    assert_eq!(divergences[0].constraint_id, "C-002");
    assert_eq!(divergences[0].passed_wave, 0);
    assert_eq!(divergences[0].failed_wave, 1);
}

#[test]
fn multiple_divergences_detected_in_same_transition() {
    let events = vec![
        pruned(&["C-001"], 0),
        pruned(&["C-001", "C-002", "C-003"], 1),
    ];
    let divergences = divergence_events_from_pruned(&events);
    let ids: std::collections::HashSet<String> = divergences
        .iter()
        .map(|d| d.constraint_id.clone())
        .collect();
    assert!(ids.contains("C-002"));
    assert!(ids.contains("C-003"));
    assert!(
        !ids.contains("C-001"),
        "C-001 was already failing — not a divergence"
    );
}

#[test]
fn balancing_instruction_empty_when_no_oscillations_or_divergences() {
    let instruction = build_balancing_instruction(&[], &[]);
    assert!(instruction.is_empty());
}

#[test]
fn balancing_instruction_contains_oscillating_pair_ids() {
    let pairs = vec![("C-TAU-1".to_owned(), "C-005".to_owned())];
    let instruction = build_balancing_instruction(&pairs, &[]);
    assert!(instruction.contains("C-TAU-1"));
    assert!(instruction.contains("C-005"));
    assert!(
        instruction.contains("OSCILLATION"),
        "must name the detected pattern"
    );
}

#[test]
fn balancing_instruction_contains_diverged_constraint_id() {
    let divergences = vec![DivergenceEvent {
        constraint_id: "C-007".to_owned(),
        passed_wave: 1,
        failed_wave: 2,
    }];
    let instruction = build_balancing_instruction(&[], &divergences);
    assert!(instruction.contains("C-007"));
    assert!(
        instruction.contains("REGRESSION"),
        "must name the detected pattern"
    );
}

#[test]
fn divergence_events_from_pruned_all_same_wave_returns_empty() {
    // all events at wave 0 → after dedup all_waves.len() = 1 < 2 → return vec![] (line 41)
    let events = vec![pruned(&["C-001"], 0), pruned(&["C-002"], 0)];
    let divergences = divergence_events_from_pruned(&events);
    assert!(
        divergences.is_empty(),
        "single distinct wave → no divergence"
    );
}

#[test]
fn sharpen_balancing_instruction_empty_cluster_ids_returns_empty() {
    // cluster_ids.is_empty() → return String::new() (line 133)
    let prior = "OSCILLATION: C-001 and C-002 conflict";
    let sharpened = sharpen_balancing_instruction(prior, &[]);
    assert!(
        sharpened.is_empty(),
        "empty cluster_ids must return empty string"
    );
}

#[test]
fn sharpen_returns_empty_when_no_cluster_ids_mentioned() {
    let prior = "OSCILLATION: C-001 and C-002 conflict";
    let sharpened = sharpen_balancing_instruction(prior, &["C-999".to_owned()]);
    assert!(
        sharpened.is_empty(),
        "no cluster constraints mentioned → empty sharpened"
    );
}

#[test]
fn sharpen_passes_through_when_cluster_id_is_mentioned() {
    let prior = "OSCILLATION: C-001 and C-002 conflict";
    let sharpened = sharpen_balancing_instruction(prior, &["C-001".to_owned()]);
    assert!(sharpened.contains("C-001"));
}
