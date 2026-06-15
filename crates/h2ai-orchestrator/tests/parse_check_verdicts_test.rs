#![allow(clippy::missing_panics_doc)]
//! TDD tests for `parse_check_verdicts` — the function that extracts per-check
//! PRESENT/MISSING verdicts from a LlmJudge CoT reason string.
//!
//! The EVALUATOR_SYSTEM_PROMPT instructs the model to emit:
//!   CHECK N: <text> → PRESENT or MISSING
//! These tests drive the complete specification of that parser.

use h2ai_orchestrator::verification::{parse_check_verdicts, score_from_verdicts};

// ── normal output ────────────────────────────────────────────────────────────

#[test]
fn single_check_present() {
    let reason = "CHECK 1: response uses stateless JWT → PRESENT\nscore = 1/1";
    let verdicts = parse_check_verdicts(reason, 1);
    assert_eq!(verdicts, vec![true], "CHECK 1 PRESENT must map to true");
}

#[test]
fn single_check_missing() {
    let reason = "CHECK 1: response avoids PII storage → MISSING\nscore = 0/1";
    let verdicts = parse_check_verdicts(reason, 1);
    assert_eq!(verdicts, vec![false], "CHECK 1 MISSING must map to false");
}

#[test]
fn two_checks_present_then_missing() {
    let reason = concat!(
        "CHECK 1: stateless JWT → PRESENT\n",
        "CHECK 2: no PII logged → MISSING\n",
        "score = 1/2"
    );
    let verdicts = parse_check_verdicts(reason, 2);
    assert_eq!(verdicts, vec![true, false]);
}

#[test]
fn two_checks_missing_then_present() {
    let reason = concat!(
        "CHECK 1: auth is stateless → MISSING\n",
        "CHECK 2: HTTPS only → PRESENT\n",
        "score = 1/2"
    );
    let verdicts = parse_check_verdicts(reason, 2);
    assert_eq!(verdicts, vec![false, true]);
}

#[test]
fn three_checks_all_present() {
    let reason = concat!(
        "CHECK 1: uses JWT → PRESENT\n",
        "CHECK 2: no PII → PRESENT\n",
        "CHECK 3: HTTPS → PRESENT\n",
        "score = 3/3"
    );
    let verdicts = parse_check_verdicts(reason, 3);
    assert_eq!(verdicts, vec![true, true, true]);
}

#[test]
fn three_checks_all_missing() {
    let reason = concat!(
        "CHECK 1: uses JWT → MISSING\n",
        "CHECK 2: no PII → MISSING\n",
        "CHECK 3: HTTPS → MISSING\n",
        "score = 0/3"
    );
    let verdicts = parse_check_verdicts(reason, 3);
    assert_eq!(verdicts, vec![false, false, false]);
}

// ── edge cases ───────────────────────────────────────────────────────────────

#[test]
fn empty_reason_returns_all_false() {
    // Conservative default: unknown = failed
    let verdicts = parse_check_verdicts("", 3);
    assert_eq!(verdicts, vec![false, false, false]);
}

#[test]
fn zero_checks_returns_empty() {
    let reason = "CHECK 1: something → PRESENT";
    let verdicts = parse_check_verdicts(reason, 0);
    assert!(verdicts.is_empty(), "n_checks=0 must return empty vec");
}

#[test]
fn reason_with_no_check_lines_returns_all_false() {
    let reason = "The proposal looks reasonable. Score: 0.7";
    let verdicts = parse_check_verdicts(reason, 2);
    assert_eq!(
        verdicts,
        vec![false, false],
        "no CHECK lines → all conservative false"
    );
}

#[test]
fn partial_check_lines_unknown_checks_default_false() {
    // Only CHECK 1 is present; CHECK 2 is absent from reason.
    let reason = "CHECK 1: JWT auth → PRESENT\nsome other text\nscore = 1/2";
    let verdicts = parse_check_verdicts(reason, 2);
    assert_eq!(
        verdicts,
        vec![true, false],
        "CHECK 2 absent from reason → conservative false"
    );
}

#[test]
fn out_of_range_check_number_ignored() {
    // Reason mentions CHECK 5 but n_checks=3; should not panic, CHECK 5 is ignored.
    let reason = concat!(
        "CHECK 1: A → PRESENT\n",
        "CHECK 2: B → PRESENT\n",
        "CHECK 3: C → PRESENT\n",
        "CHECK 5: E → MISSING\n", // beyond n_checks
    );
    let verdicts = parse_check_verdicts(reason, 3);
    assert_eq!(verdicts, vec![true, true, true]);
}

#[test]
fn case_insensitive_present() {
    let reason = "CHECK 1: something → present";
    let verdicts = parse_check_verdicts(reason, 1);
    assert_eq!(
        verdicts,
        vec![true],
        "lowercase 'present' must parse as true"
    );
}

#[test]
fn case_insensitive_missing() {
    let reason = "CHECK 1: something → Missing";
    let verdicts = parse_check_verdicts(reason, 1);
    assert_eq!(
        verdicts,
        vec![false],
        "mixed-case 'Missing' must parse as false"
    );
}

#[test]
fn check_number_uses_last_occurrence_when_duplicated() {
    // If a check number appears twice (shouldn't happen but shouldn't panic either),
    // the last occurrence wins (models sometimes re-score).
    let reason = concat!(
        "CHECK 1: JWT → MISSING\n", // first occurrence
        "CHECK 1: JWT → PRESENT\n", // second occurrence overrides
        "score = 1/1"
    );
    let verdicts = parse_check_verdicts(reason, 1);
    // Either [true] (last wins) or [false] (first wins) is acceptable; just no panic.
    assert_eq!(verdicts.len(), 1);
}

#[test]
fn rich_cot_output_with_prose_around_checks() {
    // Real LlmJudge output wraps CHECK lines in prose.
    let reason = concat!(
        "I evaluated the proposal against each criterion:\n\n",
        "The proposal uses JWT tokens for authentication.\n",
        "CHECK 1: stateless JWT authentication → PRESENT\n\n",
        "However, the logging section reveals that user IDs are written to logs.\n",
        "CHECK 2: no PII in logs → MISSING\n\n",
        "All API endpoints use HTTPS.\n",
        "CHECK 3: HTTPS for all endpoints → PRESENT\n\n",
        "score = 2/3\n",
        "{\"score\": 0.667, \"reason\": \"...\"}"
    );
    let verdicts = parse_check_verdicts(reason, 3);
    assert_eq!(verdicts, vec![true, false, true]);
}

#[test]
fn arrow_variants_with_spaces() {
    // Arrow may have variable spacing around it.
    let reason = "CHECK 1: something  →  PRESENT";
    let verdicts = parse_check_verdicts(reason, 1);
    assert_eq!(verdicts, vec![true]);
}

#[test]
fn n_checks_larger_than_found_pads_with_false() {
    // n_checks=4 but only CHECK 1 and CHECK 2 appear in reason.
    let reason = "CHECK 1: A → PRESENT\nCHECK 2: B → MISSING";
    let verdicts = parse_check_verdicts(reason, 4);
    assert_eq!(
        verdicts,
        vec![true, false, false, false],
        "missing checks 3 and 4 must default to false"
    );
}

// ── score_from_verdicts ───────────────────────────────────────────────────────

#[test]
fn score_from_verdicts_computes_fraction() {
    assert_eq!(
        score_from_verdicts(&[true, false, true, true], 4, 0.5),
        0.75
    );
    assert_eq!(score_from_verdicts(&[false, false], 2, 0.9), 0.0);
    assert_eq!(score_from_verdicts(&[true, true, true], 3, 0.1), 1.0);
}

#[test]
fn score_from_verdicts_falls_back_when_no_checks() {
    assert_eq!(score_from_verdicts(&[], 0, 0.42), 0.42);
    assert_eq!(score_from_verdicts(&[], 3, 0.7), 0.7);
}

#[test]
fn score_from_verdicts_falls_back_on_empty_verdicts_with_nonzero_n() {
    // verdicts empty but n_checks > 0: fallback (parse yielded nothing)
    assert_eq!(score_from_verdicts(&[], 4, 0.6), 0.6);
}

// ── Format A: PRESENT/MISSING without arrow ────────────────────────────────────

#[test]
fn format_a_present_missing() {
    // Format A: "CHECK N: PRESENT (reason)" / "CHECK N: MISSING (reason)"
    let reason = "CHECK 1: PRESENT (Lua script found)\nCHECK 2: MISSING (no audit log)\nCHECK 3: PRESENT (JWT used)";
    let v = parse_check_verdicts(reason, 3);
    assert_eq!(v, vec![true, false, true]);
}

#[test]
fn format_b_arrow() {
    // Format B: "CHECK N: explanation → PRESENT" / "CHECK N: explanation → MISSING"
    let reason = "CHECK 1: some reason → PRESENT\nCHECK 2: another → MISSING";
    let v = parse_check_verdicts(reason, 2);
    assert_eq!(v, vec![true, false]);
}

#[test]
fn missing_defaults_to_false() {
    let v = parse_check_verdicts("CHECK 1: MISSING (not implemented)", 2);
    assert_eq!(v, vec![false, false]);
}

#[test]
fn mixed_formats_same_reason() {
    let reason = "CHECK 1: PRESENT (ok), CHECK 2: description → PRESENT";
    let v = parse_check_verdicts(reason, 2);
    assert_eq!(v, vec![true, true]);
}

// ── parse_check_reasons ────────────────────────────────────────────────────────

use h2ai_orchestrator::verification::parse_check_reasons;

#[test]
fn parse_check_reasons_extracts_per_check_text() {
    let reason = "CHECK 1: key uses tenant_id → PRESENT\nCHECK 2: key uses request_id only → MISSING\nCHECK 3: TTL set → PRESENT";
    let reasons = parse_check_reasons(reason, 3);
    assert_eq!(reasons.len(), 3);
    assert!(
        reasons[0].contains("PRESENT"),
        "check 1 should contain PRESENT"
    );
    assert!(
        reasons[1].contains("MISSING"),
        "check 2 should contain MISSING"
    );
    assert!(
        reasons[2].contains("PRESENT"),
        "check 3 should contain PRESENT"
    );
}

#[test]
fn parse_check_reasons_returns_empty_vec_for_zero_checks() {
    let reasons = parse_check_reasons("any text", 0);
    assert!(reasons.is_empty());
}

#[test]
fn parse_check_reasons_handles_missing_check_gracefully() {
    let reason = "CHECK 1: key present → PRESENT";
    let reasons = parse_check_reasons(reason, 3);
    assert_eq!(reasons.len(), 3);
    assert!(!reasons[0].is_empty(), "check 1 should have text");
    assert!(
        reasons[1].is_empty(),
        "check 2 not in reason — should be empty string"
    );
    assert!(
        reasons[2].is_empty(),
        "check 3 not in reason — should be empty string"
    );
}
