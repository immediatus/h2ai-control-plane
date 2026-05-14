//! Helper functions for oracle gate request construction, template matching, and result aggregation.
//!
//! Provides pure functions for:
//! - Template placeholder substitution with simple key=value maps.
//! - Pattern-based clarification template matching using regex.
//! - Failure summary aggregation from multiple oracle results.
//! - Concurrency ratio computation for probabilistic interpretation.

use h2ai_config::ClarificationTemplate;
use h2ai_types::events::OracleGateResultEvent;
use regex::Regex;
use std::collections::HashMap;

/// Fill {placeholder} variables in a template string using a simple key=value map.
/// Unknown placeholders are left as-is.
///
/// # Example
/// ```
/// let mut vals = std::collections::HashMap::new();
/// vals.insert("task_id".to_string(), "t-123".to_string());
/// let result = h2ai_orchestrator::oracle_gate::fill_template_placeholders("Task {task_id} failed", &vals);
/// assert_eq!(result, "Task t-123 failed");
/// ```
pub fn fill_template_placeholders(template: &str, values: &HashMap<String, String>) -> String {
    let mut result = template.to_string();
    for (key, value) in values {
        result = result.replace(&format!("{{{}}}", key), value);
    }
    result
}

/// Find the first ClarificationTemplate whose pattern regex matches the given failure reason.
/// Returns None if no template matches or if pattern is invalid.
///
/// # Example
/// ```
/// use h2ai_config::ClarificationTemplate;
/// let templates = vec![
///     ClarificationTemplate {
///         pattern: "timeout".to_string(),
///         question_template: "Did you mean to set a timeout?".to_string(),
///     },
/// ];
/// let result = h2ai_orchestrator::oracle_gate::match_clarification_template("oracle timeout exceeded", &templates);
/// assert!(result.is_some());
/// ```
pub fn match_clarification_template<'a>(
    failure_reason: &str,
    templates: &'a [ClarificationTemplate],
) -> Option<&'a ClarificationTemplate> {
    templates.iter().find(|t| {
        Regex::new(&t.pattern)
            .ok()
            .map(|re| re.is_match(failure_reason))
            .unwrap_or(false)
    })
}

/// Aggregate failure summary from multiple oracle results for logging.
/// Returns a short string like "2/4 proposals passed (confidence: 0.45)"
///
/// Returns "no oracle results" for empty input.
///
/// # Example
/// ```
/// let summary = h2ai_orchestrator::oracle_gate::aggregate_failure_summary(&[]);
/// assert_eq!(summary, "no oracle results");
/// ```
pub fn aggregate_failure_summary(results: &[OracleGateResultEvent]) -> String {
    if results.is_empty() {
        return "no oracle results".to_string();
    }
    let total_passed: u32 = results.iter().map(|r| r.passed_proposals).sum();
    let total_checked: u32 = results.iter().map(|r| r.checked_proposals).sum();
    let avg_confidence: f64 =
        results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;
    format!(
        "{}/{} proposals passed (confidence: {:.2})",
        total_passed, total_checked, avg_confidence
    )
}

/// Compute the effective concurrency ratio: passed_proposals / total_checked.
/// Returns 1.0 if total_checked == 0 (no data = assume pass).
///
/// # Example
/// ```
/// assert!((h2ai_orchestrator::oracle_gate::effective_concurrency(3, 4) - 0.75).abs() < 1e-9);
/// assert!((h2ai_orchestrator::oracle_gate::effective_concurrency(0, 0) - 1.0).abs() < 1e-9);
/// ```
pub fn effective_concurrency(passed: u32, total: u32) -> f64 {
    if total == 0 {
        1.0
    } else {
        passed as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!((effective_concurrency(0, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn effective_concurrency_partial() {
        assert!((effective_concurrency(3, 4) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn effective_concurrency_perfect() {
        assert!((effective_concurrency(5, 5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn effective_concurrency_zero_passed() {
        assert!((effective_concurrency(0, 5) - 0.0).abs() < 1e-9);
    }
}
