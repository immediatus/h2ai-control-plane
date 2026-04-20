use h2ai_orchestrator::error_class::{classify_error, ErrorClass, RetryPolicy};

#[test]
fn transient_error_retries_with_backoff() {
    let class = classify_error("connection refused");
    assert_eq!(class, ErrorClass::Transient);
    let policy = RetryPolicy::for_class(&class);
    assert_eq!(policy.max_attempts, 3);
    assert!(policy.backoff_ms > 0);
    assert!(!policy.escalate_to_mape_k);
}

#[test]
fn recoverable_error_escalates_to_mape_k() {
    let class = classify_error("parse error in output");
    assert_eq!(class, ErrorClass::Recoverable);
    let policy = RetryPolicy::for_class(&class);
    assert_eq!(policy.max_attempts, 1);
    assert!(policy.escalate_to_mape_k);
}

#[test]
fn user_fixable_error_no_retry() {
    let class = classify_error("context underflow: J_eff=0.3 < 0.4");
    assert_eq!(class, ErrorClass::UserFixable);
    let policy = RetryPolicy::for_class(&class);
    assert_eq!(policy.max_attempts, 0);
    assert!(!policy.escalate_to_mape_k);
}

#[test]
fn unexpected_error_no_retry() {
    let class = classify_error("assertion failed: impossible state");
    assert_eq!(class, ErrorClass::Unexpected);
    let policy = RetryPolicy::for_class(&class);
    assert_eq!(policy.max_attempts, 0);
    assert!(!policy.escalate_to_mape_k);
}

#[test]
fn timeout_classifies_as_transient() {
    assert_eq!(classify_error("TAO timeout"), ErrorClass::Transient);
}

#[test]
fn rate_limit_classifies_as_transient() {
    assert_eq!(classify_error("rate limit exceeded"), ErrorClass::Transient);
}
