// Unit tests for resolve_action — tests the signal dispatch mapping
#[cfg(test)]
use h2ai_orchestrator::signal_dispatch::{resolve_action, ResumeAction};
use h2ai_types::signal::{ApproveSignal, SignalPayload, WaveContinueSignal};

#[test]
fn wave_continue_maps_to_continue_action() {
    let payload = SignalPayload::WaveContinue(WaveContinueSignal {
        grounding: Some("Use Lua".into()),
        mandate_override: Some("Prefer atomic ops".into()),
    });
    let action = resolve_action(payload);
    let ResumeAction::ContinueToNextWave {
        grounding,
        mandate_override,
    } = action
    else {
        panic!("expected ContinueToNextWave");
    };
    assert_eq!(grounding.unwrap(), "Use Lua");
    assert_eq!(mandate_override.unwrap(), "Prefer atomic ops");
}

#[test]
fn approve_maps_to_finalize_approved() {
    let payload = SignalPayload::Approve(ApproveSignal {
        approved: true,
        reviewer_note: Some("looks good".into()),
        operator_id: "alice".into(),
    });
    let action = resolve_action(payload);
    let ResumeAction::Finalize {
        approved,
        reviewer_note,
        ..
    } = action
    else {
        panic!("expected Finalize");
    };
    assert!(approved);
    assert_eq!(reviewer_note.unwrap(), "looks good");
}

#[test]
fn reject_maps_to_finalize_rejected() {
    let payload = SignalPayload::Approve(ApproveSignal {
        approved: false,
        reviewer_note: Some("wrong approach".into()),
        operator_id: "bob".into(),
    });
    let action = resolve_action(payload);
    let ResumeAction::Finalize { approved, .. } = action else {
        panic!()
    };
    assert!(!approved);
}

#[test]
fn unknown_maps_to_ignore() {
    let action = resolve_action(SignalPayload::Unknown);
    assert!(matches!(action, ResumeAction::Ignore));
}

#[test]
fn adaptive_decay_formula() {
    let base_ms: u64 = 14_400_000;
    let decay: f64 = 0.5;
    let floor_ms: u64 = 300_000;

    let effective =
        |n: u32| -> u64 { (base_ms as f64 * decay.powi(n as i32)).max(floor_ms as f64) as u64 };

    assert_eq!(effective(0), 14_400_000);
    assert_eq!(effective(1), 7_200_000);
    assert_eq!(effective(2), 3_600_000);
    assert_eq!(effective(10), floor_ms);
}
