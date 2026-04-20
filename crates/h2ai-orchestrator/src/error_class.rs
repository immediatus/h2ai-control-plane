/// The four canonical error classes for the H2AI orchestrator.
/// Each class maps to a distinct retry semantic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorClass {
    /// Transient — network blip, timeout, rate limit. Retry with exponential backoff.
    Transient,
    /// Recoverable — bad output shape, validation failure. MAPE-K retry with topology change.
    Recoverable,
    /// UserFixable — context underflow, invalid config. No automatic retry; surface to human.
    UserFixable,
    /// Unexpected — internal logic error, impossible state. Emit TaskFailedEvent, no retry.
    Unexpected,
}

pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff_ms: u64,
    pub escalate_to_mape_k: bool,
}

impl RetryPolicy {
    pub fn for_class(class: &ErrorClass) -> Self {
        match class {
            ErrorClass::Transient => Self {
                max_attempts: 3,
                backoff_ms: 500,
                escalate_to_mape_k: false,
            },
            ErrorClass::Recoverable => Self {
                max_attempts: 1,
                backoff_ms: 0,
                escalate_to_mape_k: true,
            },
            ErrorClass::UserFixable => Self {
                max_attempts: 0,
                backoff_ms: 0,
                escalate_to_mape_k: false,
            },
            ErrorClass::Unexpected => Self {
                max_attempts: 0,
                backoff_ms: 0,
                escalate_to_mape_k: false,
            },
        }
    }
}

/// Classify an error message string into one of the four canonical error classes.
/// Matching is keyword-based and case-insensitive.
pub fn classify_error(msg: &str) -> ErrorClass {
    let lower = msg.to_lowercase();
    if lower.contains("timeout") || lower.contains("rate limit") || lower.contains("connection") {
        ErrorClass::Transient
    } else if lower.contains("context underflow")
        || lower.contains("invalid config")
        || lower.contains("user")
    {
        ErrorClass::UserFixable
    } else if lower.contains("parse") || lower.contains("schema") || lower.contains("validation") {
        ErrorClass::Recoverable
    } else {
        ErrorClass::Unexpected
    }
}
