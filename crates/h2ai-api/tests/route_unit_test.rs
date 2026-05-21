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
//! Unit tests verifying that `InMemoryStateBackend` satisfies all backend traits
//! end-to-end without a live NATS server.

use chrono::Utc;
use h2ai_state::backend::{CalibrationStore, EventPublisher, SnapshotStore};
use h2ai_state::in_memory::InMemoryStateBackend;
use h2ai_types::events::{
    CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode, H2AIEvent,
    TaskSnapshot,
};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold};

fn synthetic_calibration() -> CalibrationCompletedEvent {
    let coefficients = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71])
        .expect("valid coefficients");
    let coordination_threshold = CoordinationThreshold::from_calibration(&coefficients, 0.3);
    CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients,
        coordination_threshold,
        ensemble: None,
        eigen: None,
        timestamp: Utc::now(),
        pairwise_beta: None,
        cg_mode: CgMode::default(),
        adapter_families: vec!["Mock".into()],
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: CalibrationQuality::default(),
        calibration_source: CalibrationSource::Measured,
        beta_quality: None,
    }
}

#[tokio::test]
async fn in_memory_snapshot_store_roundtrip() {
    let store = InMemoryStateBackend::new();
    let task_id = TaskId::new();
    let snap = TaskSnapshot {
        task_id: task_id.clone(),
        last_sequence: 42,
        task_state_json: "{\"phase\":\"plan\"}".into(),
        taken_at: Utc::now(),
    };

    assert!(store.get_snapshot(&task_id).await.unwrap().is_none());

    store.put_snapshot(&snap).await.unwrap();

    let loaded = store
        .get_snapshot(&task_id)
        .await
        .unwrap()
        .expect("snapshot present");
    assert_eq!(loaded.task_id, task_id);
    assert_eq!(loaded.last_sequence, 42);
    assert_eq!(loaded.task_state_json, snap.task_state_json);
}

#[tokio::test]
async fn in_memory_calibration_store_roundtrip() {
    let store = InMemoryStateBackend::new();
    let cal = synthetic_calibration();

    assert!(store.get_calibration().await.unwrap().is_none());

    store.put_calibration(&cal).await.unwrap();
    let loaded = store
        .get_calibration()
        .await
        .unwrap()
        .expect("calibration present");
    assert_eq!(loaded.calibration_id, cal.calibration_id);
    assert_eq!(loaded.coefficients.alpha, cal.coefficients.alpha);
}

#[tokio::test]
async fn in_memory_event_publisher_captures_events() {
    let backend = InMemoryStateBackend::new();
    let task_id = TaskId::new();
    let cal = synthetic_calibration();
    let event = H2AIEvent::CalibrationCompleted(cal);

    backend
        .publish_event(&task_id, &event)
        .await
        .expect("publish_event ok");

    let subject = format!("h2ai.calibration.{task_id}");
    backend
        .publish_to(&subject, &event)
        .await
        .expect("publish_to ok");

    let seq = backend
        .publish_event_seq(&task_id, &event)
        .await
        .expect("publish_event_seq ok");
    assert!(seq > 0, "sequence numbers must be monotonic and non-zero");

    let captured = backend.events().await;
    assert_eq!(captured.len(), 3, "all three publishes recorded");

    assert_eq!(captured[0].subject, format!("h2ai.tasks.{task_id}"));
    assert_eq!(captured[1].subject, subject);
    assert_eq!(captured[2].subject, format!("h2ai.tasks.{task_id}"));

    assert!(captured[0].seq < captured[1].seq);
    assert!(captured[1].seq < captured[2].seq);

    let n = backend.event_count_for_task(&task_id).await;
    assert_eq!(n, 2);
}

/// Verify `InMemoryStateBackend` auto-implements the composite `StateBackend` trait
/// via blanket impl — no explicit `impl StateBackend` needed.
#[tokio::test]
async fn in_memory_backend_satisfies_state_backend_blanket_impl() {
    fn assert_state_backend<T: h2ai_state::backend::StateBackend>(_t: &T) {}
    let backend = InMemoryStateBackend::new();
    assert_state_backend(&backend);

    backend.get_snapshot(&TaskId::new()).await.unwrap();
    backend.get_calibration().await.unwrap();
}
