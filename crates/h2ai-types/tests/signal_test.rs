use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::signal::{ApproveSignal, ResumeSignal, SignalPayload, WaveContinueSignal};

fn make_task_id() -> TaskId {
    TaskId::from_uuid(uuid::Uuid::new_v4())
}

#[test]
fn approve_signal_round_trips() {
    let sig = ResumeSignal {
        task_id: make_task_id(),
        tenant_id: TenantId::from("payments-team"),
        payload: SignalPayload::Approve(ApproveSignal {
            approved: true,
            reviewer_note: Some("LGTM".into()),
            operator_id: "ops@example.com".into(),
        }),
        timeout_at_ms: 9_999_999_999,
        issued_at_ms: 1_000_000_000,
    };
    let json = serde_json::to_string(&sig).unwrap();
    let back: ResumeSignal = serde_json::from_str(&json).unwrap();
    let SignalPayload::Approve(a) = back.payload else {
        panic!("wrong variant")
    };
    assert!(a.approved);
    assert_eq!(a.operator_id, "ops@example.com");
}

#[test]
fn wave_continue_round_trips() {
    let sig = ResumeSignal {
        task_id: make_task_id(),
        tenant_id: TenantId::from("team"),
        payload: SignalPayload::WaveContinue(WaveContinueSignal {
            grounding: Some("Use Redis Lua script".into()),
            mandate_override: None,
        }),
        timeout_at_ms: 9_999_999_999,
        issued_at_ms: 1_000_000_000,
    };
    let json = serde_json::to_string(&sig).unwrap();
    let back: ResumeSignal = serde_json::from_str(&json).unwrap();
    let SignalPayload::WaveContinue(w) = back.payload else {
        panic!("wrong variant")
    };
    assert_eq!(w.grounding.unwrap(), "Use Redis Lua script");
    assert!(w.mandate_override.is_none());
}

#[test]
fn unknown_variant_deserializes_to_unknown() {
    let raw = r#"{"task_id":"00000000-0000-0000-0000-000000000001","tenant_id":"team","payload":{"kind":"InjectKnowledge","data":{"nodes":["foo"]}},"timeout_at_ms":9999999999,"issued_at_ms":1000000000}"#;
    let sig: ResumeSignal = serde_json::from_str(raw).unwrap();
    assert!(matches!(sig.payload, SignalPayload::Unknown));
}

#[test]
fn wire_format_has_kind_and_data_fields() {
    let payload = SignalPayload::Approve(ApproveSignal {
        approved: false,
        reviewer_note: None,
        operator_id: "alice".into(),
    });
    let sig = ResumeSignal {
        task_id: make_task_id(),
        tenant_id: TenantId::from("t"),
        payload,
        timeout_at_ms: 0,
        issued_at_ms: 0,
    };
    let json = serde_json::to_string(&sig).unwrap();
    assert!(json.contains(r#""kind":"Approve""#));
    assert!(json.contains(r#""data":"#));
}

// ── Unknown variant serializes and round-trips ────────────────────────────────

#[test]
fn unknown_payload_serializes_with_kind_unknown() {
    let sig = ResumeSignal {
        task_id: make_task_id(),
        tenant_id: TenantId::from("t"),
        payload: SignalPayload::Unknown,
        timeout_at_ms: 0,
        issued_at_ms: 0,
    };
    let json = serde_json::to_string(&sig).unwrap();
    // The Unknown variant serializes as kind=Unknown
    assert!(json.contains(r#""kind":"Unknown""#));
    // Re-deserialize: an explicit "Unknown" kind maps to the Unknown variant too
    let back: ResumeSignal = serde_json::from_str(&json).unwrap();
    assert!(matches!(back.payload, SignalPayload::Unknown));
}

// ── WaveContinue with mandate_override set ────────────────────────────────────

#[test]
fn wave_continue_with_mandate_override_round_trips() {
    let sig = ResumeSignal {
        task_id: make_task_id(),
        tenant_id: TenantId::from("team"),
        payload: SignalPayload::WaveContinue(WaveContinueSignal {
            grounding: None,
            mandate_override: Some("Switch to eventual consistency".into()),
        }),
        timeout_at_ms: 9_999_999_999,
        issued_at_ms: 1_000_000_000,
    };
    let json = serde_json::to_string(&sig).unwrap();
    let back: ResumeSignal = serde_json::from_str(&json).unwrap();
    let SignalPayload::WaveContinue(w) = back.payload else {
        panic!("wrong variant")
    };
    assert!(w.grounding.is_none());
    assert_eq!(
        w.mandate_override.unwrap(),
        "Switch to eventual consistency"
    );
}

// ── ApproveSignal with approved=false and no reviewer_note ───────────────────

#[test]
fn approve_signal_rejected_no_note_round_trips() {
    let sig = ResumeSignal {
        task_id: make_task_id(),
        tenant_id: TenantId::from("t"),
        payload: SignalPayload::Approve(ApproveSignal {
            approved: false,
            reviewer_note: None,
            operator_id: "bot".into(),
        }),
        timeout_at_ms: 0,
        issued_at_ms: 0,
    };
    let json = serde_json::to_string(&sig).unwrap();
    let back: ResumeSignal = serde_json::from_str(&json).unwrap();
    let SignalPayload::Approve(a) = back.payload else {
        panic!("wrong variant")
    };
    assert!(!a.approved);
    assert!(a.reviewer_note.is_none());
}
