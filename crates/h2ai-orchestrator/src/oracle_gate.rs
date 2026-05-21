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
#[must_use]
pub fn fill_template_placeholders(template: &str, values: &HashMap<String, String>) -> String {
    let mut result = template.to_string();
    for (key, value) in values {
        result = result.replace(&format!("{{{key}}}"), value);
    }
    result
}

/// Find the first `ClarificationTemplate` whose pattern regex matches the given failure reason.
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
#[must_use]
pub fn match_clarification_template<'a>(
    failure_reason: &str,
    templates: &'a [ClarificationTemplate],
) -> Option<&'a ClarificationTemplate> {
    templates.iter().find(|t| {
        Regex::new(&t.pattern)
            .ok()
            .is_some_and(|re| re.is_match(failure_reason))
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
#[must_use]
pub fn aggregate_failure_summary(results: &[OracleGateResultEvent]) -> String {
    if results.is_empty() {
        return "no oracle results".to_string();
    }
    let total_passed: u32 = results.iter().map(|r| r.passed_proposals).sum();
    let total_checked: u32 = results.iter().map(|r| r.checked_proposals).sum();
    let avg_confidence: f64 =
        results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;
    format!("{total_passed}/{total_checked} proposals passed (confidence: {avg_confidence:.2})")
}

/// Compute the effective concurrency ratio: `passed_proposals` / `total_checked`.
/// Returns 1.0 if `total_checked` == 0 (no data = assume pass).
///
/// # Example
/// ```
/// assert!((h2ai_orchestrator::oracle_gate::effective_concurrency(3, 4) - 0.75).abs() < 1e-9);
/// assert!((h2ai_orchestrator::oracle_gate::effective_concurrency(0, 0) - 1.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn effective_concurrency(passed: u32, total: u32) -> f64 {
    if total == 0 {
        1.0
    } else {
        f64::from(passed) / f64::from(total)
    }
}
