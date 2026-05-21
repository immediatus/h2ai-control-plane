#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
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
