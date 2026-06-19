#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::wildcard_imports
)]
//! Pure-logic unit tests for `h2ai_orchestrator::phases::oracle`.
//!
//! Covers `apply_on_fail_policy` (all branches) and `run_post_selection`
//! when the gate is disabled or no NATS client is available — no NATS
//! connection required.

use h2ai_config::OracleGateConfig;
use h2ai_orchestrator::phases::oracle::{
    apply_on_fail_policy, run_post_selection, PostSelectionDecision, PostSelectionInput,
};

// ── apply_on_fail_policy ──────────────────────────────────────────────────────

#[test]
fn apply_on_fail_policy_gate_passed_is_always_accept() {
    // gate_passed = Some(true) → Accept regardless of on_fail value
    assert_eq!(
        apply_on_fail_policy(Some(true), "evict"),
        PostSelectionDecision::Accept
    );
    assert_eq!(
        apply_on_fail_policy(Some(true), "clarify"),
        PostSelectionDecision::Accept
    );
    assert_eq!(
        apply_on_fail_policy(Some(true), "pass"),
        PostSelectionDecision::Accept
    );
}

#[test]
fn apply_on_fail_policy_gate_none_is_always_accept() {
    // gate_passed = None (gate disabled / skipped) → Accept
    assert_eq!(
        apply_on_fail_policy(None, "evict"),
        PostSelectionDecision::Accept
    );
    assert_eq!(
        apply_on_fail_policy(None, "clarify"),
        PostSelectionDecision::Accept
    );
}

#[test]
fn apply_on_fail_policy_gate_failed_on_fail_clarify() {
    assert_eq!(
        apply_on_fail_policy(Some(false), "clarify"),
        PostSelectionDecision::Clarify
    );
}

#[test]
fn apply_on_fail_policy_gate_failed_on_fail_pass() {
    assert_eq!(
        apply_on_fail_policy(Some(false), "pass"),
        PostSelectionDecision::Accept
    );
}

#[test]
fn apply_on_fail_policy_gate_failed_on_fail_evict() {
    // "evict" is the default catch-all
    assert_eq!(
        apply_on_fail_policy(Some(false), "evict"),
        PostSelectionDecision::Evict
    );
}

#[test]
fn apply_on_fail_policy_gate_failed_unknown_on_fail_defaults_to_evict() {
    // Any unknown on_fail value should fall through to Evict
    assert_eq!(
        apply_on_fail_policy(Some(false), "unknown_policy"),
        PostSelectionDecision::Evict
    );
    assert_eq!(
        apply_on_fail_policy(Some(false), ""),
        PostSelectionDecision::Evict
    );
}

// ── run_post_selection: gate disabled ────────────────────────────────────────

#[tokio::test]
async fn run_post_selection_gate_disabled_returns_accept() {
    let cfg = OracleGateConfig {
        enabled: false,
        ..OracleGateConfig::default()
    };
    let result = run_post_selection(PostSelectionInput {
        task_id: "task-1".to_string(),
        winner_text: "some winner output",
        oracle_config: &cfg,
        nats: None,
    })
    .await;
    assert_eq!(result, PostSelectionDecision::Accept);
}

// ── run_post_selection: gate enabled, no NATS client ─────────────────────────

#[tokio::test]
async fn run_post_selection_gate_enabled_no_nats_returns_accept() {
    let cfg = OracleGateConfig {
        enabled: true,
        subject: "h2ai.oracle.post".to_string(),
        timeout_secs: 5,
        on_timeout: "pass".to_string(),
        on_fail: "evict".to_string(),
        ..OracleGateConfig::default()
    };
    let result = run_post_selection(PostSelectionInput {
        task_id: "task-2".to_string(),
        winner_text: "winner output",
        oracle_config: &cfg,
        nats: None,
    })
    .await;
    // No NATS client → early return Accept
    assert_eq!(result, PostSelectionDecision::Accept);
}

// ── PostSelectionDecision: PartialEq / Debug coverage ───────────────────────

#[test]
fn post_selection_decision_debug_and_eq() {
    let a = PostSelectionDecision::Evict;
    let b = PostSelectionDecision::Evict;
    assert_eq!(a, b);
    let c = PostSelectionDecision::Clarify;
    assert_ne!(a, c);
    // Debug must not panic
    let _ = format!("{a:?}");
    let _ = format!("{c:?}");
    let _ = format!("{:?}", PostSelectionDecision::Accept);
}
