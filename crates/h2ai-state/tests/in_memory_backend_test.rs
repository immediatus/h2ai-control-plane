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
use h2ai_state::backend::{
    CalibrationStore, EstimatorStore, EventPublisher, OproStore, SignalPublisher, SnapshotStore,
    StateBackend, TailEvents,
};
use h2ai_state::in_memory::InMemoryStateBackend;
use h2ai_types::calibration::{AuditorCircuitState, CalibrationRecord, ProbeSource};
use h2ai_types::events::{
    CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode, H2AIEvent,
    TaskSnapshot,
};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::prompt_variant::{AdapterOproState, PromptVariant};
use h2ai_types::signal::{ResumeSignal, SignalPayload};
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

// ── EventPublisher::publish_event and publish_to ──────────────────────────────

#[tokio::test]
async fn publish_event_stores_event() {
    let backend = InMemoryStateBackend::new();
    let task_id = TaskId::new();
    let event = H2AIEvent::CalibrationCompleted(cal_event());
    backend.publish_event(&task_id, &event).await.unwrap();
    assert_eq!(backend.events().await.len(), 1);
}

#[tokio::test]
async fn publish_to_stores_event_under_subject() {
    let backend = InMemoryStateBackend::new();
    let task_id = TaskId::new();
    let event = H2AIEvent::CalibrationCompleted(cal_event());
    backend.publish_to("custom.subject", &event).await.unwrap();
    // InMemoryStateBackend accumulates all published events in its event log.
    assert_eq!(backend.events().await.len(), 1);
    // The task_id-keyed path also still works independently.
    backend.publish_event(&task_id, &event).await.unwrap();
    assert_eq!(backend.events().await.len(), 2);
}

// ── Arc<T>: OproStore forwarding impl ────────────────────────────────────────

#[tokio::test]
async fn arc_wrapped_backend_put_get_adapter_opro_state() {
    use std::sync::Arc;
    let backend = Arc::new(InMemoryStateBackend::new());

    let state = AdapterOproState {
        adapter_name: "arc-adapter".into(),
        j_eff_ema: 0.75,
        n_tasks_total: 5,
        n_tasks_since_last_opro: 0,
        last_opro_started_at: None,
        suppress_until_n_tasks: 0,
        bandit_arms: Default::default(),
    };
    backend.put_adapter_opro_state(&state).await.unwrap();

    let loaded = backend
        .get_adapter_opro_state("arc-adapter")
        .await
        .unwrap()
        .expect("state must exist");

    assert!((loaded.j_eff_ema - 0.75).abs() < 1e-9);
    assert_eq!(loaded.n_tasks_total, 5);
}

#[tokio::test]
async fn arc_wrapped_backend_put_get_prompt_variant() {
    use std::sync::Arc;
    let backend = Arc::new(InMemoryStateBackend::new());

    use h2ai_types::prompt_variant::PromptVariantSource;
    let variant = PromptVariant {
        adapter_name: "arc-adapter".into(),
        prompt_key: "explore".into(),
        variant_id: "v1".into(),
        text: "You are a test agent.".into(),
        source: PromptVariantSource::Seed,
        created_at: Utc::now(),
        score: Some(0.8),
    };
    backend.put_prompt_variant(&variant).await.unwrap();

    let loaded = backend
        .get_prompt_variant("arc-adapter", "explore", "v1")
        .await
        .unwrap()
        .expect("variant must exist");

    assert_eq!(loaded.text, "You are a test agent.");
    assert!((loaded.score.unwrap() - 0.8).abs() < 1e-9);
}

#[tokio::test]
async fn arc_wrapped_backend_active_variant_ptr_roundtrip() {
    use std::sync::Arc;
    let backend = Arc::new(InMemoryStateBackend::new());

    // Initially absent.
    let initial = backend
        .get_active_variant_ptr("arc-adapter", "explore")
        .await
        .unwrap();
    assert!(initial.is_none());

    backend
        .set_active_variant_ptr("arc-adapter", "explore", "v2")
        .await
        .unwrap();

    let loaded = backend
        .get_active_variant_ptr("arc-adapter", "explore")
        .await
        .unwrap()
        .expect("ptr must exist after set");

    assert_eq!(loaded, "v2");
}

// ── EstimatorStore (tao, srani, bandit) ──────────────────────────────────────

#[tokio::test]
async fn estimator_store_tao_roundtrip() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    assert!(backend
        .get_tao_estimator_state(&tenant)
        .await
        .unwrap()
        .is_none());
    backend
        .put_tao_estimator_state(&tenant, 0.42, 7)
        .await
        .unwrap();
    let (ema, count) = backend
        .get_tao_estimator_state(&tenant)
        .await
        .unwrap()
        .unwrap();
    assert!((ema - 0.42).abs() < 1e-9);
    assert_eq!(count, 7);
}

#[tokio::test]
async fn estimator_store_srani_roundtrip() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    assert!(backend.get_srani_state(&tenant).await.unwrap().is_none());
    backend.put_srani_state(&tenant, 0.75, 3).await.unwrap();
    let (cfi, count) = backend.get_srani_state(&tenant).await.unwrap().unwrap();
    assert!((cfi - 0.75).abs() < 1e-9);
    assert_eq!(count, 3);
}

#[tokio::test]
async fn estimator_store_bandit_roundtrip() {
    let backend = InMemoryStateBackend::new();
    let tenant = TenantId::default_tenant();
    assert!(backend.get_bandit_state(&tenant).await.unwrap().is_none());
    let bytes = b"bandit-state-bytes".to_vec();
    backend
        .put_bandit_state(&tenant, bytes.clone())
        .await
        .unwrap();
    let loaded = backend.get_bandit_state(&tenant).await.unwrap().unwrap();
    assert_eq!(loaded, bytes);
}

// ── TailEvents::tail_task_events_boxed ───────────────────────────────────────

#[tokio::test]
async fn tail_task_events_returns_events_after_seq() {
    use futures::StreamExt;
    let backend = InMemoryStateBackend::new();
    let tid = TaskId::new();
    let event = H2AIEvent::TaskFailed(h2ai_types::events::TaskFailedEvent {
        task_id: tid.clone(),
        pruned_events: vec![],
        topologies_tried: vec![],
        tau_values_tried: vec![],
        multiplication_condition_failure: None,
        timestamp: Utc::now(),
    });
    backend.publish_event(&tid, &event).await.unwrap();
    backend.publish_event(&tid, &event).await.unwrap();

    let mut stream = backend.tail_task_events_boxed(&tid, 1).await.unwrap();
    let item = stream.next().await.unwrap().unwrap();
    assert_eq!(item.0, 2); // seq=2 is the second event
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn tail_task_events_empty_stream_from_seq_beyond_end() {
    use futures::StreamExt;
    let backend = InMemoryStateBackend::new();
    let tid = TaskId::new();
    let mut stream = backend.tail_task_events_boxed(&tid, 100).await.unwrap();
    assert!(stream.next().await.is_none());
}

// ── SignalPublisher::publish_signal ──────────────────────────────────────────

#[tokio::test]
async fn signal_publisher_publish_signal_succeeds() {
    let backend = InMemoryStateBackend::new();
    let signal = ResumeSignal {
        task_id: TaskId::new(),
        tenant_id: TenantId::default_tenant(),
        payload: SignalPayload::Unknown,
        timeout_at_ms: 9_999_999,
        issued_at_ms: 1_000_000,
    };
    backend.publish_signal(&signal).await.unwrap();
}

// ── publish_event_seq ─────────────────────────────────────────────────────────

#[tokio::test]
async fn publish_event_seq_returns_increasing_seq() {
    let backend = InMemoryStateBackend::new();
    let tid = TaskId::new();
    let event = H2AIEvent::TaskFailed(h2ai_types::events::TaskFailedEvent {
        task_id: tid.clone(),
        pruned_events: vec![],
        topologies_tried: vec![],
        tau_values_tried: vec![],
        multiplication_condition_failure: None,
        timestamp: Utc::now(),
    });
    let s1 = backend.publish_event_seq(&tid, &event).await.unwrap();
    let s2 = backend.publish_event_seq(&tid, &event).await.unwrap();
    assert!(s2 > s1);
}
