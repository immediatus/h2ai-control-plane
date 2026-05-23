use h2ai_types::approval::{
    compute_risk_level, ApprovalDecision, ApprovalRecord, HumanOracleRating, HumanOracleRequest,
};
use h2ai_types::events::{ApprovalRiskLevel, ApprovalTrigger};
use h2ai_types::identity::TenantId;

// ── compute_risk_level ────────────────────────────────────────────────────────

#[test]
fn low_confidence_below_0_3_is_high_risk() {
    assert_eq!(
        compute_risk_level(&ApprovalTrigger::ManifestFlag, 0.29),
        ApprovalRiskLevel::High
    );
    assert_eq!(
        compute_risk_level(&ApprovalTrigger::LowConfidence, 0.0),
        ApprovalRiskLevel::High
    );
}

#[test]
fn confidence_exactly_0_3_is_not_high() {
    assert_ne!(
        compute_risk_level(&ApprovalTrigger::LowConfidence, 0.3),
        ApprovalRiskLevel::High
    );
}

#[test]
fn manifest_flag_above_0_3_is_medium() {
    assert_eq!(
        compute_risk_level(&ApprovalTrigger::ManifestFlag, 0.5),
        ApprovalRiskLevel::Medium
    );
}

#[test]
fn low_confidence_trigger_above_0_3_is_medium() {
    assert_eq!(
        compute_risk_level(&ApprovalTrigger::LowConfidence, 0.8),
        ApprovalRiskLevel::Medium
    );
}

// ── struct serde roundtrips ───────────────────────────────────────────────────

#[test]
fn approval_record_serde_roundtrip() {
    let rec = ApprovalRecord {
        task_id: "task-123".into(),
        tenant_id: TenantId::default_tenant(),
        proposed_output: "output text".into(),
        q_confidence: 0.75,
        triggered_by: ApprovalTrigger::ManifestFlag,
        created_at_ms: 1_000_000,
        timeout_at_ms: 2_000_000,
    };
    let json = serde_json::to_string(&rec).unwrap();
    let back: ApprovalRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(back.task_id, rec.task_id);
    assert_eq!(back.q_confidence, rec.q_confidence);
}

#[test]
fn approval_decision_serde_roundtrip() {
    let dec = ApprovalDecision {
        approved: true,
        reviewer_note: Some("looks good".into()),
        operator_id: "ops-1".into(),
        decided_at_ms: 1_500_000,
    };
    let json = serde_json::to_string(&dec).unwrap();
    let back: ApprovalDecision = serde_json::from_str(&json).unwrap();
    assert!(back.approved);
    assert_eq!(back.reviewer_note.unwrap(), "looks good");
}

#[test]
fn human_oracle_request_serde_roundtrip() {
    let req = HumanOracleRequest {
        oracle_id: "oracle-1".into(),
        tenant_id: TenantId::default_tenant(),
        task_description: "Rate this output".into(),
        winning_output: "The answer is 42".into(),
        rubric: "Is it correct?".into(),
        created_at_ms: 1_000_000,
        expires_at_ms: 2_000_000,
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: HumanOracleRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.oracle_id, req.oracle_id);
    assert_eq!(back.winning_output, req.winning_output);
}

#[test]
fn human_oracle_rating_serde_roundtrip() {
    let rating = HumanOracleRating {
        oracle_id: "oracle-1".into(),
        passed: true,
        likert_score: Some(4),
        rater_note: Some("Good answer".into()),
        rater_id: "rater-abc".into(),
        rated_at_ms: 1_200_000,
    };
    let json = serde_json::to_string(&rating).unwrap();
    let back: HumanOracleRating = serde_json::from_str(&json).unwrap();
    assert!(back.passed);
    assert_eq!(back.likert_score, Some(4));
}
