#![allow(clippy::missing_panics_doc)]

use h2ai_api::oracle_worker::fuse_reduce_by_family;
use h2ai_types::sizing::OracleFamily;

// ── fuse_reduce_by_family ─────────────────────────────────────────────────────

#[test]
fn fuse_reduce_empty_returns_false_zero() {
    let (passed, score) = fuse_reduce_by_family(&[]);
    assert!(!passed, "empty verdicts must return passed=false");
    assert!(
        (score - 0.0).abs() < 1e-9,
        "empty verdicts must return score=0.0"
    );
}

#[test]
fn fuse_reduce_single_family_uses_min_score() {
    // Two Syntactic oracles: scores 0.9 and 0.3 → family min = 0.3
    let verdicts = [
        (OracleFamily::Syntactic, true, 0.9),
        (OracleFamily::Syntactic, true, 0.3),
    ];
    let (_, score) = fuse_reduce_by_family(&verdicts);
    assert!(
        (score - 0.3).abs() < 1e-9,
        "within-family min must be 0.3, got {score}"
    );
}

#[test]
fn fuse_reduce_correlated_failure_counts_as_one_vote() {
    // JSON Schema (Syntactic) + Z3 (Syntactic) both fail → one family vote, not two
    let verdicts = [
        (OracleFamily::Syntactic, false, 0.0),
        (OracleFamily::Syntactic, false, 0.0),
        (OracleFamily::Semantic, true, 0.9),
    ];
    let (_, score) = fuse_reduce_by_family(&verdicts);
    // Syntactic family score = min(0.0, 0.0) = 0.0
    // Semantic family score = 0.9
    // Final = mean(0.0, 0.9) = 0.45
    assert!(
        (score - 0.45).abs() < 1e-9,
        "two correlated syntactic failures count as one: score must be 0.45, got {score}"
    );
}

#[test]
fn fuse_reduce_cross_family_averages_family_scores() {
    // Syntactic: 1.0, Semantic: 0.8, Human: 0.6 → mean = 0.8
    let verdicts = [
        (OracleFamily::Syntactic, true, 1.0),
        (OracleFamily::Semantic, true, 0.8),
        (OracleFamily::Human, true, 0.6),
    ];
    let (passed, score) = fuse_reduce_by_family(&verdicts);
    assert!(passed, "all families pass → result must pass");
    assert!(
        (score - 0.8).abs() < 1e-9,
        "mean of family scores must be 0.8, got {score}"
    );
}

#[test]
fn fuse_reduce_passes_when_score_at_or_above_half() {
    let verdicts = [(OracleFamily::Semantic, true, 0.5)];
    let (passed, _) = fuse_reduce_by_family(&verdicts);
    assert!(passed, "score >= 0.5 must pass");
}

#[test]
fn fuse_reduce_fails_when_score_below_half() {
    let verdicts = [(OracleFamily::Semantic, false, 0.4)];
    let (passed, _) = fuse_reduce_by_family(&verdicts);
    assert!(!passed, "score < 0.5 must not pass");
}
