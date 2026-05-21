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
//! Direct tests of `InMemoryStateBackend` against the backend traits.

use chrono::Utc;
use h2ai_state::backend::{CalibrationStore, EventPublisher, SnapshotStore, StateBackend};
use h2ai_state::in_memory::InMemoryStateBackend;
use h2ai_types::calibration::{AuditorCircuitState, CalibrationRecord, ProbeSource};
use h2ai_types::events::{
    CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode, H2AIEvent,
    TaskSnapshot,
};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold};

fn cal_event() -> CalibrationCompletedEvent {
    let coefficients = CoherencyCoefficients::new(0.10, 0.020, vec![0.60, 0.70, 0.80])
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
        adapter_families: vec![],
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

fn cal_record(adapter_profile: &str) -> CalibrationRecord {
    CalibrationRecord {
        adapter_profile: adapter_profile.to_owned(),
        constraint_id: None,
        alpha: 0.12,
        alpha_measured: 0.12,
        beta_0: 0.021,
        k: 1.0,
        n_useful_history: vec![],
        probe_source: ProbeSource::Synthetic,
        fingerprint: None,
        circuit_state: AuditorCircuitState::Closed,
    }
}

#[tokio::test]
async fn snapshot_overwrite_replaces_previous() {
    let store = InMemoryStateBackend::new();
    let task_id = TaskId::new();
    let first = TaskSnapshot {
        task_id: task_id.clone(),
        last_sequence: 1,
        task_state_json: "{}".into(),
        taken_at: Utc::now(),
    };
    let second = TaskSnapshot {
        task_id: task_id.clone(),
        last_sequence: 99,
        task_state_json: "{\"p\":\"verify\"}".into(),
        taken_at: Utc::now(),
    };
    store.put_snapshot(&first).await.unwrap();
    store.put_snapshot(&second).await.unwrap();
    let loaded = store.get_snapshot(&task_id).await.unwrap().unwrap();
    assert_eq!(loaded.last_sequence, 99);
    assert_eq!(loaded.task_state_json, second.task_state_json);
}

#[tokio::test]
async fn calibration_record_keyed_by_adapter_profile() {
    let store = InMemoryStateBackend::new();
    let a = cal_record("adapter-a");
    let b = cal_record("adapter-b");
    store.put_calibration_record(&a).await.unwrap();
    store.put_calibration_record(&b).await.unwrap();
    let got_a = store.get_calibration_record("adapter-a").await.unwrap();
    let got_b = store.get_calibration_record("adapter-b").await.unwrap();
    let missing = store.get_calibration_record("nope").await.unwrap();
    assert!(got_a.is_some());
    assert!(got_b.is_some());
    assert!(missing.is_none());
    assert_eq!(got_a.unwrap().adapter_profile, "adapter-a");
    assert_eq!(got_b.unwrap().adapter_profile, "adapter-b");
}

#[tokio::test]
async fn event_publisher_assigns_strictly_increasing_seq() {
    let backend = InMemoryStateBackend::new();
    let task_id = TaskId::new();
    let event = H2AIEvent::CalibrationCompleted(cal_event());

    let s1 = backend.publish_event_seq(&task_id, &event).await.unwrap();
    let s2 = backend.publish_event_seq(&task_id, &event).await.unwrap();
    let s3 = backend.publish_event_seq(&task_id, &event).await.unwrap();
    assert!(s1 < s2 && s2 < s3, "seq must increase: {s1} {s2} {s3}");
    assert_eq!(backend.events().await.len(), 3);
}

#[tokio::test]
async fn calibration_store_idempotent_overwrite() {
    let store = InMemoryStateBackend::new();
    let c1 = cal_event();
    let c2 = cal_event();
    store.put_calibration(&c1).await.unwrap();
    store.put_calibration(&c2).await.unwrap();
    let loaded = store.get_calibration().await.unwrap().unwrap();
    assert_eq!(loaded.calibration_id, c2.calibration_id);
}

#[tokio::test]
async fn blanket_state_backend_impl_covers_in_memory() {
    fn requires_state_backend<T: StateBackend>(_t: &T) {}
    let backend = InMemoryStateBackend::new();
    requires_state_backend(&backend);
}
