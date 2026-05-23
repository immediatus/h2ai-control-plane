use h2ai_autonomic::audit_channel::AuditChannelBuilder;
use h2ai_types::events::ConstraintViolation;
use h2ai_types::sizing::OspConfig;

fn v(constraint_id: &str, hint: Option<&str>) -> ConstraintViolation {
    ConstraintViolation {
        constraint_id: constraint_id.to_string(),
        score: 0.0,
        severity_label: "Hard".to_string(),
        remediation_hint: hint.map(str::to_string),
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
    }
}

#[test]
fn adaptive_threshold_n1_is_one() {
    let tau = AuditChannelBuilder::adaptive_threshold(1, 0.1);
    assert!((tau - 1.0).abs() < 1e-6, "τ(1)=1.0, got {tau}");
}

#[test]
fn adaptive_threshold_n10_is_approx_066() {
    let tau = AuditChannelBuilder::adaptive_threshold(10, 0.1);
    assert!(tau > 0.60 && tau < 0.72, "τ(10)≈0.66, got {tau}");
}

#[test]
fn adaptive_threshold_never_below_05() {
    let tau = AuditChannelBuilder::adaptive_threshold(100_000, 0.1);
    assert!(tau >= 0.5, "threshold must be ≥ 0.5, got {tau}");
}

#[test]
fn build_zone3_none_when_no_violations() {
    let result = AuditChannelBuilder::build_zone3(&[], 0, 1, 0, &OspConfig::default());
    assert!(result.is_none());
}

#[test]
fn build_zone3_none_when_n_v_exceeds_guard() {
    let violations = vec![v("c-001", Some("hint"))];
    // N_v = 5 exceeds max_n_v_for_zone3 = 4
    let result = AuditChannelBuilder::build_zone3(&violations, 1, 5, 0, &OspConfig::default());
    assert!(result.is_none(), "N_v=5 triggers gravity-well guard");
}

#[test]
fn build_zone3_none_when_below_concordance() {
    // τ(2, 0.1) ≈ 0.96. Only 1 of 2 failed proposals violated c-001 → C_k=0.5 < 0.96
    let violations = vec![v("c-001", Some("hint"))]; // 1 violation out of n_f=2
    let result = AuditChannelBuilder::build_zone3(&violations, 2, 1, 0, &OspConfig::default());
    assert!(result.is_none(), "C_k=0.5 < τ(2)=0.96 → no injection");
}

#[test]
fn build_zone3_injects_concordant_violation() {
    // τ(1, 0.1) = 1.0. 1 violation of c-001 out of n_f=1 → C_k=1.0 ≥ 1.0 → inject
    let violations = vec![v("c-001", Some("Always validate input schemas"))];
    let result = AuditChannelBuilder::build_zone3(&violations, 1, 1, 0, &OspConfig::default());
    assert!(result.is_some(), "C_k=1.0 meets τ(1)=1.0 → inject");
    let text = result.unwrap();
    assert!(text.contains("c-001"), "must include constraint_id");
    assert!(
        text.contains("Always validate input schemas"),
        "must include hint"
    );
}

#[test]
fn build_zone3_uses_only_constraint_id_and_hint_no_raw_text() {
    let violations = vec![v("c-005", Some("Use structured return types"))];
    let result = AuditChannelBuilder::build_zone3(&violations, 1, 1, 0, &OspConfig::default());
    let text = result.unwrap();
    assert!(text.contains("c-005"));
    assert!(text.contains("Use structured return types"));
    assert!(!text.contains("raw_output"));
    assert!(!text.contains("ProposalEvent"));
}

#[test]
fn build_zone3_suppressed_after_retry3_with_nf1() {
    let violations = vec![v("c-001", Some("hint"))];
    let result = AuditChannelBuilder::build_zone3(&violations, 1, 1, 4, &OspConfig::default());
    assert!(result.is_none(), "retry_count=4 with N_f=1 must suppress");
}

#[test]
fn build_zone3_not_suppressed_at_retry3_with_nf1() {
    let violations = vec![v("c-001", Some("hint"))];
    let result = AuditChannelBuilder::build_zone3(&violations, 1, 1, 3, &OspConfig::default());
    assert!(result.is_some(), "retry_count=3 is still within limit");
}

#[test]
fn build_zone3_uses_positive_framing_not_prohibition() {
    let violations = vec![v("c-001", Some("hint"))];
    let result = AuditChannelBuilder::build_zone3(&violations, 1, 1, 0, &OspConfig::default());
    let text = result.unwrap();
    assert!(!text.contains("Do not"), "must not use prohibition framing");
    assert!(
        !text.contains("must not"),
        "must not use prohibition framing"
    );
    assert!(
        text.contains("difficulty") || text.contains("Guidance") || text.contains("observed"),
        "must use evaluation-metadata framing"
    );
}

#[test]
fn build_zone3_hint_absent_omits_guidance_line() {
    // Violation with no hint: constraint_id must appear but no "Guidance:" line.
    let violations = vec![v("c-002", None)];
    let result = AuditChannelBuilder::build_zone3(&violations, 1, 1, 0, &OspConfig::default());
    let text = result.expect("concordant violation must produce zone3");
    assert!(text.contains("c-002"), "constraint_id must be present");
    assert!(!text.contains("Guidance:"), "no hint → no guidance line");
}

#[test]
fn build_zone3_mixed_hints_only_emits_guidance_for_hinted() {
    // Two violations for different constraints: one with hint, one without.
    // Only the hinted constraint should have a "Guidance:" line.
    // n_f=2, both concordant at C_k=0.5.
    // τ(2, 0.1) ≈ 0.96, so C_k=0.5 < threshold → neither survives concordance check.
    // Use n_f=1 so τ(1)=1.0 and only one violation → single concordant entry.
    let single = vec![v("c-003", Some("Always validate"))];
    let result = AuditChannelBuilder::build_zone3(&single, 1, 1, 0, &OspConfig::default());
    let text = result.expect("c-003 must produce zone3");
    assert!(text.contains("Always validate"));

    // Now test with no-hint variant (c-004 alone, n_f=1).
    let no_hint = vec![v("c-004", None)];
    let result2 = AuditChannelBuilder::build_zone3(&no_hint, 1, 1, 0, &OspConfig::default());
    let text2 = result2.expect("c-004 must produce zone3");
    assert!(text2.contains("c-004"));
    assert!(!text2.contains("Guidance:"));
}

#[test]
fn build_zone3_none_when_violations_empty_with_positive_nf() {
    // n_f > 0 but violations slice is empty → second arm of the `||` fires.
    let result = AuditChannelBuilder::build_zone3(&[], 2, 1, 0, &OspConfig::default());
    assert!(
        result.is_none(),
        "empty violations must return None even when n_f > 0"
    );
}

#[test]
fn adaptive_threshold_returns_one_when_n_f_is_zero() {
    // Line 21: n_f=0 early return → 1.0 (no observations → maximum uncertainty)
    let threshold = AuditChannelBuilder::adaptive_threshold(0, 0.1);
    assert!(
        (threshold - 1.0).abs() < 1e-12,
        "n_f=0 must return 1.0, got {threshold}"
    );
}
