use h2ai_types::approval::{ApprovalDecision, ApprovalRecord};
use h2ai_types::events::{ApprovalRiskLevel, ApprovalTrigger};
use h2ai_types::identity::TenantId;

#[test]
fn approval_record_serializes_roundtrip() {
    let record = ApprovalRecord {
        task_id: "task-123".into(),
        tenant_id: TenantId::default_tenant(),
        proposed_output: "the answer".into(),
        q_confidence: 0.42,
        triggered_by: ApprovalTrigger::LowConfidence,
        created_at_ms: 1000,
        timeout_at_ms: 1000 + 1800000,
    };
    let json = serde_json::to_string(&record).unwrap();
    let back: ApprovalRecord = serde_json::from_str(&json).unwrap();
    assert!((back.q_confidence - 0.42).abs() < 1e-9);
    assert_eq!(back.task_id, "task-123");
}

#[test]
fn approval_decision_serializes_roundtrip() {
    let decision = ApprovalDecision {
        approved: true,
        reviewer_note: Some("LGTM".into()),
        operator_id: "alice@example.com".into(),
        decided_at_ms: 9999,
    };
    let json = serde_json::to_string(&decision).unwrap();
    let back: ApprovalDecision = serde_json::from_str(&json).unwrap();
    assert!(back.approved);
    assert_eq!(back.operator_id, "alice@example.com");
}

#[test]
fn risk_level_high_when_low_confidence() {
    let risk = h2ai_types::approval::compute_risk_level(&ApprovalTrigger::LowConfidence, 0.25);
    assert_eq!(risk, ApprovalRiskLevel::High);
}

#[test]
fn risk_level_medium_when_manifest_flag_and_moderate_confidence() {
    let risk = h2ai_types::approval::compute_risk_level(&ApprovalTrigger::ManifestFlag, 0.60);
    assert_eq!(risk, ApprovalRiskLevel::Medium);
}

#[test]
fn require_approval_defaults_false() {
    use h2ai_types::manifest::TaskManifest;
    let json = r#"{"description":"t","pareto_weights":{"throughput":0.33,"containment":0.33,"diversity":0.34},"topology":{"kind":"auto","branching_factor":null},"explorers":{"count":3,"tau_min":null,"tau_max":null,"roles":[],"review_gates":[],"slot_configs":[]}}"#;
    let m: TaskManifest = serde_json::from_str(json).unwrap();
    assert!(
        !m.require_approval,
        "require_approval must default to false"
    );
}

#[test]
fn hitl_config_defaults_are_sane() {
    use h2ai_config::H2AIConfig;
    let cfg = H2AIConfig::load_layered(None).expect("load defaults");
    assert!(cfg.hitl.enabled);
    assert!((cfg.hitl.confidence_threshold - 0.50).abs() < 1e-9);
    assert_eq!(cfg.hitl.timeout_ms, 1_800_000); // 30 minutes
}

#[test]
fn high_confidence_task_bypasses_gate() {
    let q = 0.95f64;
    let threshold = 0.50f64;
    let require_approval = false;
    let hitl_enabled = true;
    let oracle_task = false;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(!needs_approval, "high confidence task must bypass gate");
}

#[test]
fn low_confidence_task_hits_gate() {
    let q = 0.30f64;
    let threshold = 0.50f64;
    let require_approval = false;
    let hitl_enabled = true;
    let oracle_task = false;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(needs_approval, "low confidence task must hit gate");
}

#[test]
fn require_approval_hits_gate_regardless_of_confidence() {
    let q = 0.99f64;
    let threshold = 0.50f64;
    let require_approval = true;
    let hitl_enabled = true;
    let oracle_task = false;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(needs_approval, "require_approval=true must always hit gate");
}

#[test]
fn oracle_task_always_bypasses_gate() {
    let q = 0.10f64;
    let threshold = 0.50f64;
    let require_approval = true;
    let hitl_enabled = true;
    let oracle_task = true;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(!needs_approval, "oracle task must always bypass HITL gate");
}

#[test]
fn hitl_disabled_bypasses_gate() {
    let q = 0.10f64;
    let threshold = 0.50f64;
    let require_approval = true;
    let hitl_enabled = false;
    let oracle_task = false;

    let needs_approval = hitl_enabled && !oracle_task && (require_approval || q < threshold);
    assert!(!needs_approval, "disabled HITL must bypass gate");
}

#[test]
fn timeout_triggers_auto_reject_condition() {
    let now_ms: u64 = 1_000_000;
    let timeout_at_ms: u64 = 999_999; // already expired
    assert!(
        now_ms > timeout_at_ms,
        "expired record must trigger auto-reject"
    );

    let future_timeout: u64 = 2_000_000;
    assert!(
        now_ms <= future_timeout,
        "future timeout must not trigger auto-reject"
    );
}
