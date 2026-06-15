use h2ai_config::ClarificationTemplate;
use h2ai_orchestrator::oracle_gate::{
    aggregate_failure_summary, fill_template_placeholders, match_clarification_template,
};
use h2ai_types::events::OracleGateResultEvent;
use std::collections::HashMap;

#[test]
fn fill_placeholders_replaces_known_keys() {
    let mut vals = HashMap::new();
    vals.insert("task_id".to_string(), "t-123".to_string());
    let result = fill_template_placeholders("Task {task_id} failed", &vals);
    assert_eq!(result, "Task t-123 failed");
}

#[test]
fn fill_placeholders_leaves_unknown_keys() {
    let vals = HashMap::new();
    let result = fill_template_placeholders("Task {unknown} failed", &vals);
    assert_eq!(result, "Task {unknown} failed");
}

#[test]
fn fill_placeholders_multiple_keys() {
    let mut vals = HashMap::new();
    vals.insert("task_id".to_string(), "t-123".to_string());
    vals.insert("reason".to_string(), "timeout".to_string());
    let result = fill_template_placeholders("Task {task_id} failed: {reason}", &vals);
    assert_eq!(result, "Task t-123 failed: timeout");
}

#[test]
fn match_clarification_template_finds_match() {
    let templates = vec![ClarificationTemplate {
        pattern: "timeout".to_string(),
        question_template: "Did you mean to set a timeout?".to_string(),
    }];
    let result = match_clarification_template("oracle timeout exceeded", &templates);
    assert!(result.is_some());
    assert_eq!(
        result.unwrap().question_template,
        "Did you mean to set a timeout?"
    );
}

#[test]
fn match_clarification_template_no_match() {
    let templates = vec![ClarificationTemplate {
        pattern: "timeout".to_string(),
        question_template: "timeout question".to_string(),
    }];
    let result = match_clarification_template("unrelated failure", &templates);
    assert!(result.is_none());
}

#[test]
fn match_clarification_template_invalid_regex() {
    let templates = vec![ClarificationTemplate {
        pattern: "[invalid".to_string(),
        question_template: "invalid regex".to_string(),
    }];
    let result = match_clarification_template("any text", &templates);
    assert!(result.is_none());
}

#[test]
fn match_clarification_template_returns_first_match() {
    let templates = vec![
        ClarificationTemplate {
            pattern: "timeout".to_string(),
            question_template: "first match".to_string(),
        },
        ClarificationTemplate {
            pattern: "timeout".to_string(),
            question_template: "second match".to_string(),
        },
    ];
    let result = match_clarification_template("oracle timeout exceeded", &templates);
    assert!(result.is_some());
    assert_eq!(result.unwrap().question_template, "first match");
}

#[test]
fn aggregate_failure_summary_empty() {
    assert_eq!(aggregate_failure_summary(&[]), "no oracle results");
}

#[test]
fn aggregate_failure_summary_single_result() {
    use chrono::Utc;
    let results = vec![OracleGateResultEvent {
        task_id: "task-1".to_string(),
        gate_passed: true,
        confidence: 0.9,
        summary: "all passed".to_string(),
        checked_proposals: 4,
        passed_proposals: 4,
        timestamp: Utc::now(),
    }];
    let summary = aggregate_failure_summary(&results);
    assert!(summary.contains("4/4 proposals passed"));
    assert!(summary.contains("confidence: 0.90"));
}

#[test]
fn aggregate_failure_summary_multiple_results() {
    use chrono::Utc;
    let results = vec![
        OracleGateResultEvent {
            task_id: "task-1".to_string(),
            gate_passed: true,
            confidence: 0.8,
            summary: "partial".to_string(),
            checked_proposals: 4,
            passed_proposals: 3,
            timestamp: Utc::now(),
        },
        OracleGateResultEvent {
            task_id: "task-2".to_string(),
            gate_passed: false,
            confidence: 0.5,
            summary: "poor".to_string(),
            checked_proposals: 4,
            passed_proposals: 1,
            timestamp: Utc::now(),
        },
    ];
    let summary = aggregate_failure_summary(&results);
    assert!(summary.contains("4/8 proposals passed"));
    assert!(summary.contains("confidence: 0.65"));
}

#[test]
fn effective_concurrency_zero_total() {
    use h2ai_orchestrator::oracle_gate::effective_concurrency;
    assert!((effective_concurrency(0, 0) - 1.0).abs() < 1e-9);
}

#[test]
fn effective_concurrency_partial() {
    use h2ai_orchestrator::oracle_gate::effective_concurrency;
    assert!((effective_concurrency(3, 4) - 0.75).abs() < 1e-9);
}

#[test]
fn effective_concurrency_perfect() {
    use h2ai_orchestrator::oracle_gate::effective_concurrency;
    assert!((effective_concurrency(5, 5) - 1.0).abs() < 1e-9);
}

#[test]
fn effective_concurrency_zero_passed() {
    use h2ai_orchestrator::oracle_gate::effective_concurrency;
    assert!((effective_concurrency(0, 5) - 0.0).abs() < 1e-9);
}

// ── apply_on_fail_policy ──────────────────────────────────────────────────────

#[test]
fn evict_policy_on_failed_gate() {
    use h2ai_orchestrator::phases::oracle::{apply_on_fail_policy, PostSelectionDecision};
    assert_eq!(
        apply_on_fail_policy(Some(false), "evict"),
        PostSelectionDecision::Evict
    );
}

#[test]
fn pass_policy_ignores_failure() {
    use h2ai_orchestrator::phases::oracle::{apply_on_fail_policy, PostSelectionDecision};
    assert_eq!(
        apply_on_fail_policy(Some(false), "pass"),
        PostSelectionDecision::Accept
    );
}

#[test]
fn accept_when_gate_passed() {
    use h2ai_orchestrator::phases::oracle::{apply_on_fail_policy, PostSelectionDecision};
    assert_eq!(
        apply_on_fail_policy(Some(true), "evict"),
        PostSelectionDecision::Accept
    );
}

#[test]
fn accept_when_gate_not_run() {
    use h2ai_orchestrator::phases::oracle::{apply_on_fail_policy, PostSelectionDecision};
    assert_eq!(
        apply_on_fail_policy(None, "evict"),
        PostSelectionDecision::Accept
    );
}

#[test]
fn clarify_policy_on_failure() {
    use h2ai_orchestrator::phases::oracle::{apply_on_fail_policy, PostSelectionDecision};
    assert_eq!(
        apply_on_fail_policy(Some(false), "clarify"),
        PostSelectionDecision::Clarify
    );
}
