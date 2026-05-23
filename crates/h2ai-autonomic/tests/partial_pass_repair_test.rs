#![allow(clippy::missing_panics_doc)]
//! TDD tests for the fixed `partial_pass_from_event` — drives the three-bug fix:
//!
//! Bug 1: proposal_text must come from `event.raw_output`, not `event.reason`
//!        ("verification compliance 0.00" was the old text — 28 chars, garbage)
//!
//! Bug 2: check attribution must use `ConstraintViolation.check_verdicts`, not
//!        substring matching between constraint titles and check questions.
//!
//! Bug 3: LlmJudge per-check verdicts stored in `check_verdicts` (populated by
//!        `parse_check_verdicts`) must actually be read and used.
//!
//! New signature:
//!   partial_pass_from_event(
//!       event: &BranchPrunedEvent,
//!       checks: &[String],
//!       offsets: &[(String, usize, usize)],  // (constraint_id, start, count)
//!   ) -> Option<PartialPass>

use chrono::Utc;
use h2ai_autonomic::repair::partial_pass_from_event;
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::RoleErrorCost;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_event(raw_output: &str, violations: Vec<ConstraintViolation>) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: "verification compliance 0.00".into(), // the broken status string
        raw_output: raw_output.to_owned(),
        constraint_error_cost: RoleErrorCost::new(0.5).unwrap(),
        violated_constraints: violations,
        timestamp: Utc::now(),
    }
}

fn violation(id: &str, check_verdicts: Vec<bool>) -> ConstraintViolation {
    ConstraintViolation {
        constraint_id: id.into(),
        score: 0.0,
        severity_label: "Hard".into(),
        remediation_hint: None,
        constraint_description: format!("{id} must be satisfied"),
        verifier_reason: None,
        check_verdicts,
        criteria_pass: None,
    }
}

fn violation_with_score(id: &str, score: f64) -> ConstraintViolation {
    ConstraintViolation {
        constraint_id: id.into(),
        score,
        severity_label: "Hard".into(),
        remediation_hint: None,
        constraint_description: format!("{id} must be satisfied"),
        verifier_reason: None,
        check_verdicts: vec![], // LLM skipped CHECK format
        criteria_pass: None,
    }
}

/// `offsets`: vec of (constraint_id, start_in_flat_checks, count)
fn offsets(entries: &[(&str, usize, usize)]) -> Vec<(String, usize, usize)> {
    entries
        .iter()
        .map(|(id, start, count)| (id.to_string(), *start, *count))
        .collect()
}

fn checks(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

// ── Bug 1: proposal_text source ───────────────────────────────────────────────

#[test]
fn proposal_text_comes_from_raw_output_not_reason() {
    let real_proposal =
        "Use OAuth 2.0 with PKCE for authentication. All tokens stored in HttpOnly cookies.";
    let event = make_event(real_proposal, vec![violation("C-001", vec![false])]);
    let cs = checks(&["stateless auth", "secure token storage"]);
    let offs = offsets(&[("C-001", 0, 1), ("C-002", 1, 1)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX)
        .expect("check 1 (index 1) passes — C-002 not violated");

    assert_eq!(partial.proposal_text, real_proposal);
    assert_ne!(partial.proposal_text, "verification compliance 0.00");
}

#[test]
fn proposal_text_never_the_short_status_string() {
    let real_proposal = "The system uses stateless JWT tokens for auth.";
    let event = make_event(real_proposal, vec![violation("C-001", vec![false, true])]);
    let cs = checks(&["stateless auth", "HTTPS only"]);
    let offs = offsets(&[("C-001", 0, 2)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX).unwrap();
    assert!(
        partial.proposal_text.len() > 28,
        "partial.proposal_text has {} chars but must be > 28",
        partial.proposal_text.len()
    );
}

// ── Bug 2 + 3: check attribution via check_verdicts ──────────────────────────

#[test]
fn check_verdicts_used_for_attribution_not_title_matching() {
    let event = make_event(
        "A good proposal text.",
        vec![violation("C-001", vec![false, true])],
    );
    let cs = checks(&["uses stateless auth", "stores tokens securely"]);
    let offs = offsets(&[("C-001", 0, 2)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX)
        .expect("check 1 (index 1) passed via check_verdicts");

    let results = &partial.check_results;
    assert_eq!(results.len(), 2);
    assert!(
        !results[0].2,
        "check 0 must be failed (check_verdicts[0]=false)"
    );
    assert!(
        results[1].2,
        "check 1 must be passed (check_verdicts[1]=true)"
    );
}

#[test]
fn two_constraints_checks_attributed_correctly() {
    let event = make_event(
        "Proposal covering two constraints.",
        vec![
            violation("C-001", vec![true, false]),
            violation("C-002", vec![false, true]),
        ],
    );
    let cs = checks(&["auth stateless", "no PII", "HTTPS", "rate limiting"]);
    let offs = offsets(&[("C-001", 0, 2), ("C-002", 2, 2)]);

    let partial =
        partial_pass_from_event(&event, &cs, &offs, usize::MAX).expect("checks 0 and 3 pass");

    let results = &partial.check_results;
    assert_eq!(results.len(), 4);
    assert!(results[0].2, "check 0 (C-001[0]=true) must pass");
    assert!(!results[1].2, "check 1 (C-001[1]=false) must fail");
    assert!(!results[2].2, "check 2 (C-002[0]=false) must fail");
    assert!(results[3].2, "check 3 (C-002[1]=true) must pass");
}

#[test]
fn unviolated_constraint_all_checks_pass() {
    let event = make_event("Good proposal.", vec![violation("C-001", vec![false])]);
    let cs = checks(&["auth", "HTTPS", "rate limit"]);
    let offs = offsets(&[("C-001", 0, 1), ("C-002", 1, 2)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX)
        .expect("checks 1 and 2 pass (C-002 not violated)");

    let results = &partial.check_results;
    assert!(!results[0].2, "check 0 failed (C-001 violated)");
    assert!(results[1].2, "check 1 passed (C-002 not violated)");
    assert!(results[2].2, "check 2 passed (C-002 not violated)");
}

#[test]
fn violated_constraint_without_check_verdicts_all_fail() {
    let event = make_event("Proposal.", vec![violation("C-001", vec![])]);
    let cs = checks(&["check A", "check B"]);
    let offs = offsets(&[("C-001", 0, 2)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX);
    assert!(
        partial.is_none(),
        "no partial pass when violated constraint has no check_verdicts"
    );
}

#[test]
fn violated_constraint_with_empty_check_verdicts_partial_fallback() {
    let event = make_event("Proposal.", vec![violation("C-001", vec![])]);
    let cs = checks(&["A", "B", "C"]);
    let offs = offsets(&[("C-001", 0, 1), ("C-002", 1, 2)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX)
        .expect("checks 1 and 2 pass (C-002 unviolated)");

    assert!(
        !partial.check_results[0].2,
        "check 0 fails (C-001 no verdicts)"
    );
    assert!(
        partial.check_results[1].2,
        "check 1 passes (C-002 unviolated)"
    );
    assert!(
        partial.check_results[2].2,
        "check 2 passes (C-002 unviolated)"
    );
}

// ── None cases ────────────────────────────────────────────────────────────────

#[test]
fn returns_none_when_checks_empty() {
    let event = make_event("Proposal.", vec![]);
    let result = partial_pass_from_event(&event, &[], &[], usize::MAX);
    assert!(result.is_none());
}

#[test]
fn returns_none_when_all_checks_failed() {
    let event = make_event("Proposal.", vec![violation("C-001", vec![false, false])]);
    let cs = checks(&["A", "B"]);
    let offs = offsets(&[("C-001", 0, 2)]);

    let result = partial_pass_from_event(&event, &cs, &offs, usize::MAX);
    assert!(result.is_none());
}

#[test]
fn returns_none_when_all_constraints_violated_and_no_verdicts() {
    let event = make_event(
        "Proposal.",
        vec![violation("C-001", vec![]), violation("C-002", vec![])],
    );
    let cs = checks(&["A", "B"]);
    let offs = offsets(&[("C-001", 0, 1), ("C-002", 1, 1)]);

    let result = partial_pass_from_event(&event, &cs, &offs, usize::MAX);
    assert!(result.is_none());
}

// ── score computation ─────────────────────────────────────────────────────────

#[test]
fn score_is_fraction_of_passed_checks() {
    let event = make_event("Proposal.", vec![violation("C-001", vec![false, false])]);
    let cs = checks(&["A", "B", "C", "D"]);
    let offs = offsets(&[("C-001", 0, 2), ("C-002", 2, 2)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX).unwrap();
    assert!(
        (partial.score - 0.5).abs() < 1e-9,
        "score must be 2/4=0.5, got {}",
        partial.score
    );
}

#[test]
fn score_one_when_all_checks_pass() {
    let event = make_event("Perfect proposal.", vec![]);
    let cs = checks(&["A", "B", "C"]);
    let offs = offsets(&[("C-001", 0, 3)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX).unwrap();
    assert!(
        (partial.score - 1.0).abs() < 1e-9,
        "score must be 1.0, got {}",
        partial.score
    );
}

// ── truncation ────────────────────────────────────────────────────────────────

#[test]
fn long_raw_output_is_truncated() {
    let long_text = "A".repeat(2000);
    let event = make_event(&long_text, vec![violation("C-001", vec![false])]);
    let cs = checks(&["check A", "check B"]);
    let offs = offsets(&[("C-001", 0, 1), ("C-002", 1, 1)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, 1500).unwrap();
    assert!(
        partial.proposal_text.len() < long_text.len(),
        "must be truncated"
    );
    assert!(
        partial.proposal_text.contains("truncated"),
        "must have truncation marker"
    );
}

#[test]
fn short_raw_output_not_truncated() {
    let short_text = "Short proposal.";
    let event = make_event(short_text, vec![violation("C-001", vec![false])]);
    let cs = checks(&["A", "B"]);
    let offs = offsets(&[("C-001", 0, 1), ("C-002", 1, 1)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX).unwrap();
    assert_eq!(partial.proposal_text, short_text);
}

// ── offset mapping edge cases ─────────────────────────────────────────────────

#[test]
fn check_not_in_any_offset_defaults_to_pass() {
    let event = make_event("Proposal.", vec![violation("C-001", vec![false])]);
    let cs = checks(&["A", "B", "C"]);
    let offs = offsets(&[("C-001", 0, 1)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX).unwrap();
    assert!(
        !partial.check_results[0].2,
        "check 0 failed (C-001 violated)"
    );
    assert!(
        partial.check_results[1].2,
        "check 1 no offset → defaults to pass"
    );
    assert!(
        partial.check_results[2].2,
        "check 2 no offset → defaults to pass"
    );
}

#[test]
fn empty_offsets_all_unviolated_constraints_all_pass() {
    let event = make_event("Perfect proposal.", vec![]);
    let cs = checks(&["A", "B"]);

    let partial = partial_pass_from_event(&event, &cs, &[], usize::MAX).unwrap();
    assert!(partial.check_results.iter().all(|(_, _, p)| *p));
    assert!((partial.score - 1.0).abs() < 1e-9);
}

// ── regression: the real HLE bug scenario ────────────────────────────────────

#[test]
fn regression_hle_partial_chars_28_scenario() {
    let status_string = "verification compliance 0.00";
    assert_eq!(status_string.len(), 28);

    let real_proposal =
        "I will answer this HLE question using the following expert consensus approach: ...";
    let event = make_event(
        real_proposal,
        vec![violation("HLE-C001", vec![true, false, true])],
    );
    let cs = checks(&["clear reasoning", "correct calculation", "cites sources"]);
    let offs = offsets(&[("HLE-C001", 0, 3)]);

    let partial = partial_pass_from_event(&event, &cs, &offs, usize::MAX).unwrap();

    assert_ne!(partial.proposal_text.len(), 28);
    assert_eq!(partial.proposal_text, real_proposal);
    assert!(partial.check_results[0].2);
    assert!(!partial.check_results[1].2);
    assert!(partial.check_results[2].2);
    assert!(
        (partial.score - 2.0 / 3.0).abs() < 1e-9,
        "score must be 2/3, got {}",
        partial.score
    );
}

// ── Score-based fallback when LLM skips CHECK N format ───────────────────────

/// When check_verdicts is empty but score > 0, infer passing checks from the score.
/// HLE scenario: verifier returns {"score": 0.67, "reason": "..."} without CHECK lines.
#[test]
fn partial_pass_score_fallback_two_of_three_checks() {
    let checks = vec![
        "Check A".to_string(),
        "Check B".to_string(),
        "Check C".to_string(),
    ];
    let offs = vec![("C-1".to_string(), 0, 3)];
    let event = make_event(
        "proposal text",
        vec![violation_with_score("C-1", 2.0 / 3.0)],
    );
    let partial = partial_pass_from_event(&event, &checks, &offs, usize::MAX);

    assert!(
        partial.is_some(),
        "score=0.67 with empty check_verdicts must yield a partial pass (not None)"
    );
    let partial = partial.unwrap();
    assert_eq!(
        partial.check_results.iter().filter(|(_, _, p)| *p).count(),
        2,
        "2 of 3 checks must be inferred as passing for score=0.67"
    );
    assert!(
        (partial.score - 2.0 / 3.0).abs() < 1e-9,
        "partial score must be 2/3, got {}",
        partial.score
    );
}

/// Score=0.0 with empty check_verdicts → still None (no passing checks).
#[test]
fn partial_pass_score_fallback_zero_score_returns_none() {
    let checks = vec!["Check A".to_string(), "Check B".to_string()];
    let offs = vec![("C-1".to_string(), 0, 2)];
    let event = make_event("proposal", vec![violation_with_score("C-1", 0.0)]);
    assert!(
        partial_pass_from_event(&event, &checks, &offs, usize::MAX).is_none(),
        "score=0.0 with empty check_verdicts must return None"
    );
}

/// Score=0.5 on a 2-check constraint → 1 check inferred as PRESENT.
#[test]
fn partial_pass_score_fallback_one_of_two_checks() {
    let checks = vec!["Check A".to_string(), "Check B".to_string()];
    let offs = vec![("C-1".to_string(), 0, 2)];
    let event = make_event("proposal", vec![violation_with_score("C-1", 0.5)]);
    let partial = partial_pass_from_event(&event, &checks, &offs, usize::MAX).unwrap();
    assert_eq!(
        partial.check_results.iter().filter(|(_, _, p)| *p).count(),
        1,
        "score=0.5 on 2-check constraint → 1 check inferred as passing"
    );
}

/// Explicit check_verdicts must take precedence over score.
#[test]
fn partial_pass_explicit_verdicts_override_score() {
    let checks = vec![
        "Check A".to_string(),
        "Check B".to_string(),
        "Check C".to_string(),
    ];
    let offs = vec![("C-1".to_string(), 0, 3)];
    let mut v = violation_with_score("C-1", 0.0);
    v.check_verdicts = vec![true, false, true];
    let event = make_event("proposal", vec![v]);
    let partial = partial_pass_from_event(&event, &checks, &offs, usize::MAX).unwrap();
    assert!(
        partial.check_results[0].2,
        "check 0 must pass per check_verdicts"
    );
    assert!(
        !partial.check_results[1].2,
        "check 1 must fail per check_verdicts"
    );
    assert!(
        partial.check_results[2].2,
        "check 2 must pass per check_verdicts"
    );
}
