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
//! Integration tests covering the ~45 public async methods in `nats.rs` that
//! were not exercised by the existing test suite.
//!
//! All tests use graceful-skip: when NATS is unavailable the test simply returns
//! without failing.  No `#[ignore]` attributes are used.

use h2ai_state::nats::NatsClient;
use h2ai_types::calibration::{AuditorCircuitState, AuditorHealth, CalibrationRecord, ProbeSource};
use h2ai_types::checkpoint::TaskCheckpoint;
use h2ai_types::conflict::ConflictRateAccumulator;
use h2ai_types::events::{CalibrationCompletedEvent, TaskSnapshot};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::prompt_variant::{AdapterOproState, PromptVariant, PromptVariantSource};
use h2ai_types::reasoning_checkpoint::{
    ReasoningCheckpointPhase, TaskMetaState, TaskReasoningCheckpoint,
};
use h2ai_types::signal::{ResumeSignal, SignalPayload, WaveContinueSignal};
use h2ai_types::sizing::CoherencyCoefficients;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;

// Serialize shadow-promoted-domains tests: both write to the same fixed KV key.
static SHADOW_DOMAINS_LOCK: std::sync::LazyLock<Arc<Mutex<()>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(())));

// ── shared test helpers ───────────────────────────────────────────────────────

async fn connect() -> Option<NatsClient> {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    match NatsClient::connect(&url).await {
        Ok(c) => {
            if c.ensure_infrastructure().await.is_err() {
                return None;
            }
            Some(c)
        }
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            None
        }
    }
}

fn make_calibration_completed() -> CalibrationCompletedEvent {
    let cc = CoherencyCoefficients::new(0.10, 0.02, vec![0.70, 0.72]).unwrap();
    let theta = h2ai_types::sizing::CoordinationThreshold::from_calibration(&cc, 0.3);
    CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients: cc,
        coordination_threshold: theta,
        ensemble: None,
        eigen: None,
        timestamp: chrono::Utc::now(),
        pairwise_beta: None,
        cg_mode: Default::default(),
        adapter_families: Vec::new(),
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: Default::default(),
        calibration_source: Default::default(),
        beta_quality: None,
    }
}

fn make_checkpoint(task_id: &str) -> TaskCheckpoint {
    TaskCheckpoint {
        task_id: task_id.to_string(),
        phase: "ParallelGeneration".to_string(),
        node_id: "node-rt".to_string(),
        lease_seq: 0,
        proposals: vec!["proposal A".to_string()],
        auditor_survivors: vec![0],
        resolved_output: None,
        manifest_json: "{}".to_string(),
        object_store_ref: None,
        created_at_ms: 1_000_000,
        updated_at_ms: 1_000_000,
        constraint_snapshot: None,
        j_eff: None,
    }
}

// ── snapshot KV ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn snapshot_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let task_id = TaskId::new();
    let snap = TaskSnapshot {
        task_id: task_id.clone(),
        last_sequence: 42,
        task_state_json: r#"{"phase":"Merging"}"#.to_string(),
        taken_at: chrono::Utc::now(),
    };
    client.put_snapshot(&snap).await.expect("put_snapshot");
    let loaded = client
        .get_snapshot(&task_id)
        .await
        .expect("get_snapshot")
        .expect("should be Some");
    assert_eq!(loaded.task_id, task_id);
    assert_eq!(loaded.last_sequence, 42);
}

#[tokio::test]
async fn snapshot_get_returns_none_for_unknown_task() {
    let Some(client) = connect().await else {
        return;
    };
    let result = client
        .get_snapshot(&TaskId::new())
        .await
        .expect("get_snapshot");
    assert!(result.is_none());
}

// ── calibration KV ───────────────────────────────────────────────────────────

#[tokio::test]
async fn calibration_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let event = make_calibration_completed();
    let stored_alpha = event.coefficients.alpha;
    client
        .put_calibration(&event)
        .await
        .expect("put_calibration");
    let loaded = client
        .get_calibration()
        .await
        .expect("get_calibration")
        .expect("should be Some");
    assert!((loaded.coefficients.alpha - stored_alpha).abs() < 1e-9);
}

#[tokio::test]
async fn calibration_record_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let record = CalibrationRecord {
        adapter_profile: "test-adapter-rt".to_string(),
        constraint_id: None,
        alpha: 0.05,
        alpha_measured: 0.06,
        beta_0: 0.02,
        k: 0.8,
        n_useful_history: vec![],
        probe_source: ProbeSource::Same,
        fingerprint: None,
        circuit_state: AuditorCircuitState::Closed,
    };
    client
        .put_calibration_record(&record)
        .await
        .expect("put_calibration_record");
    let loaded = client
        .get_calibration_record("test-adapter-rt")
        .await
        .expect("get_calibration_record")
        .expect("should be Some");
    assert!((loaded.alpha - 0.05_f32).abs() < 1e-6_f32);
    assert_eq!(loaded.adapter_profile, "test-adapter-rt");
}

#[tokio::test]
async fn calibration_record_get_returns_none_for_unknown_profile() {
    let Some(client) = connect().await else {
        return;
    };
    let result = client
        .get_calibration_record("no-such-profile-xyz")
        .await
        .expect("get_calibration_record");
    assert!(result.is_none());
}

// ── auditor health KV ─────────────────────────────────────────────────────────

#[tokio::test]
async fn auditor_health_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let health = AuditorHealth {
        state: AuditorCircuitState::Open,
        last_probe_cg: 0.42,
        tripped_at: Some(1_700_000_000_000),
        recovery_probe_count: 3,
    };
    client
        .put_auditor_health("rt-profile", &health)
        .await
        .expect("put_auditor_health");
    let loaded = client
        .get_auditor_health("rt-profile")
        .await
        .expect("get_auditor_health")
        .expect("should be Some");
    assert_eq!(loaded.state, AuditorCircuitState::Open);
    assert_eq!(loaded.recovery_probe_count, 3);
}

#[tokio::test]
async fn auditor_health_get_returns_none_for_unknown() {
    let Some(client) = connect().await else {
        return;
    };
    let result = client
        .get_auditor_health("totally-unknown-profile-xyz")
        .await
        .expect("get_auditor_health");
    assert!(result.is_none());
}

// ── probe lease ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn probe_lease_acquire_release() {
    let Some(client) = connect().await else {
        return;
    };
    // Release first to clear any stale state from a previous run
    client
        .release_probe_lease("probe-rt-adapter")
        .await
        .expect("release_probe_lease (pre-clean)");

    let acquired = client
        .acquire_probe_lease("probe-rt-adapter", 60)
        .await
        .expect("acquire_probe_lease");
    assert!(acquired, "should acquire lease on first call");

    // Second attempt within TTL should fail
    let second = client
        .acquire_probe_lease("probe-rt-adapter", 60)
        .await
        .expect("acquire_probe_lease second");
    assert!(!second, "lease is already held");

    client
        .release_probe_lease("probe-rt-adapter")
        .await
        .expect("release_probe_lease");

    // After release, can acquire again
    let reacquired = client
        .acquire_probe_lease("probe-rt-adapter", 60)
        .await
        .expect("acquire_probe_lease after release");
    assert!(reacquired, "should acquire after release");

    // Cleanup
    client
        .release_probe_lease("probe-rt-adapter")
        .await
        .expect("release_probe_lease cleanup");
}

// ── shadow promoted domains ───────────────────────────────────────────────────

#[tokio::test]
async fn shadow_promoted_domains_put_get_roundtrip() {
    let _guard = SHADOW_DOMAINS_LOCK.lock().await;
    let Some(client) = connect().await else {
        return;
    };
    let mut domains: HashSet<String> = HashSet::new();
    domains.insert("code".to_string());
    domains.insert("factual".to_string());

    client
        .put_shadow_promoted_domains(&domains)
        .await
        .expect("put_shadow_promoted_domains");
    let loaded = client
        .get_shadow_promoted_domains()
        .await
        .expect("get_shadow_promoted_domains");
    assert!(loaded.contains("code"));
    assert!(loaded.contains("factual"));
    assert_eq!(loaded.len(), 2);
}

#[tokio::test]
async fn shadow_promoted_domains_empty_set() {
    let _guard = SHADOW_DOMAINS_LOCK.lock().await;
    let Some(client) = connect().await else {
        return;
    };
    let empty: HashSet<String> = HashSet::new();
    client
        .put_shadow_promoted_domains(&empty)
        .await
        .expect("put empty");
    let loaded = client
        .get_shadow_promoted_domains()
        .await
        .expect("get_shadow_promoted_domains");
    assert!(loaded.is_empty());
}

// ── TAO estimator state ───────────────────────────────────────────────────────

#[tokio::test]
async fn tao_estimator_state_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::default_tenant();
    client
        .put_tao_estimator_state(&tenant, 1.23, 42)
        .await
        .expect("put_tao_estimator_state");
    let result = client
        .get_tao_estimator_state(&tenant)
        .await
        .expect("get_tao_estimator_state")
        .expect("should be Some");
    assert!((result.0 - 1.23).abs() < 1e-9, "ema should match");
    assert_eq!(result.1, 42, "count should match");
}

#[tokio::test]
async fn tao_estimator_state_returns_none_when_absent() {
    let Some(client) = connect().await else {
        return;
    };
    // Use a tenant that is very unlikely to have existing data
    let tenant = TenantId::from("nonexistent-tao-test-tenant");
    let result = client
        .get_tao_estimator_state(&tenant)
        .await
        .expect("get_tao_estimator_state");
    assert!(result.is_none());
}

// ── SRANI state ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn srani_state_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::default_tenant();
    client
        .put_srani_state(&tenant, 0.55, 10)
        .await
        .expect("put_srani_state");
    let result = client
        .get_srani_state(&tenant)
        .await
        .expect("get_srani_state")
        .expect("should be Some");
    assert!((result.0 - 0.55).abs() < 1e-9, "ema_cfi should match");
    assert_eq!(result.1, 10, "count should match");
}

#[tokio::test]
async fn srani_state_returns_none_when_absent() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::from("nonexistent-srani-test-tenant");
    let result = client
        .get_srani_state(&tenant)
        .await
        .expect("get_srani_state");
    assert!(result.is_none());
}

// ── bandit state ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn bandit_state_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::default_tenant();
    let json = br#"{"arms":[{"variant_id":"v1","alpha":1.0,"beta":1.0}]}"#.to_vec();
    client
        .put_bandit_state(&tenant, json.clone())
        .await
        .expect("put_bandit_state");
    let loaded = client
        .get_bandit_state(&tenant)
        .await
        .expect("get_bandit_state")
        .expect("should be Some");
    assert_eq!(loaded, json);
}

#[tokio::test]
async fn bandit_state_returns_none_when_absent() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::from("nonexistent-bandit-test-tenant");
    let result = client
        .get_bandit_state(&tenant)
        .await
        .expect("get_bandit_state");
    assert!(result.is_none());
}

// ── safety profile snapshot ───────────────────────────────────────────────────

#[tokio::test]
async fn safety_profile_snapshot_put_does_not_error() {
    let Some(client) = connect().await else {
        return;
    };
    let cfg = h2ai_config::SafetyConfig::default();
    client
        .put_safety_profile_snapshot(&cfg)
        .await
        .expect("put_safety_profile_snapshot");
}

// ── prompt variants ───────────────────────────────────────────────────────────

#[tokio::test]
async fn prompt_variant_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let variant = PromptVariant {
        variant_id: "v-rt-001".to_string(),
        adapter_name: "test-adapter-rt".to_string(),
        prompt_key: "system_preamble".to_string(),
        text: "You are a helpful assistant.".to_string(),
        source: PromptVariantSource::Seed,
        created_at: chrono::Utc::now(),
        score: None,
    };
    client
        .put_prompt_variant(&variant)
        .await
        .expect("put_prompt_variant");
    let loaded = client
        .get_prompt_variant("test-adapter-rt", "system_preamble", "v-rt-001")
        .await
        .expect("get_prompt_variant")
        .expect("should be Some");
    assert_eq!(loaded.variant_id, "v-rt-001");
    assert_eq!(loaded.text, "You are a helpful assistant.");
}

#[tokio::test]
async fn prompt_variant_get_returns_none_for_unknown() {
    let Some(client) = connect().await else {
        return;
    };
    let result = client
        .get_prompt_variant("no-adapter", "no-key", "no-variant")
        .await
        .expect("get_prompt_variant");
    assert!(result.is_none());
}

#[tokio::test]
async fn active_variant_ptr_set_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    client
        .set_active_variant_ptr("test-adapter-rt", "system_preamble", "v-rt-001")
        .await
        .expect("set_active_variant_ptr");
    let ptr = client
        .get_active_variant_ptr("test-adapter-rt", "system_preamble")
        .await
        .expect("get_active_variant_ptr")
        .expect("should be Some");
    assert_eq!(ptr, "v-rt-001");
}

#[tokio::test]
async fn active_variant_ptr_returns_none_when_absent() {
    let Some(client) = connect().await else {
        return;
    };
    let result = client
        .get_active_variant_ptr("no-adapter-xyz", "no-key-xyz")
        .await
        .expect("get_active_variant_ptr");
    assert!(result.is_none());
}

// ── OPRO state ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn adapter_opro_state_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let state = AdapterOproState {
        adapter_name: "opro-rt-adapter".to_string(),
        j_eff_ema: 0.75,
        n_tasks_total: 100,
        n_tasks_since_last_opro: 10,
        last_opro_started_at: None,
        suppress_until_n_tasks: 5,
        bandit_arms: HashMap::new(),
    };
    client
        .put_adapter_opro_state(&state)
        .await
        .expect("put_adapter_opro_state");
    let loaded = client
        .get_adapter_opro_state("opro-rt-adapter")
        .await
        .expect("get_adapter_opro_state")
        .expect("should be Some");
    assert!((loaded.j_eff_ema - 0.75).abs() < 1e-9);
    assert_eq!(loaded.n_tasks_total, 100);
}

#[tokio::test]
async fn adapter_opro_state_returns_none_when_absent() {
    let Some(client) = connect().await else {
        return;
    };
    let result = client
        .get_adapter_opro_state("no-such-opro-adapter-xyz")
        .await
        .expect("get_adapter_opro_state");
    assert!(result.is_none());
}

// ── checkpoint list + delete ──────────────────────────────────────────────────

#[tokio::test]
async fn list_task_checkpoints_includes_written_entries() {
    let Some(client) = connect().await else {
        return;
    };
    let c1 = make_checkpoint("rt-list-task-1");
    let c2 = make_checkpoint("rt-list-task-2");
    client.put_task_checkpoint(&c1, None).await.expect("put c1");
    client.put_task_checkpoint(&c2, None).await.expect("put c2");

    let all = client.list_task_checkpoints().await;
    let ids: Vec<&str> = all.iter().map(|c| c.task_id.as_str()).collect();
    assert!(ids.contains(&"rt-list-task-1"));
    assert!(ids.contains(&"rt-list-task-2"));

    // Cleanup
    client.delete_task_checkpoint("rt-list-task-1").await.ok();
    client.delete_task_checkpoint("rt-list-task-2").await.ok();
}

#[tokio::test]
async fn delete_task_checkpoint_removes_entry() {
    let Some(client) = connect().await else {
        return;
    };
    let c = make_checkpoint("rt-delete-task");
    client
        .put_task_checkpoint(&c, None)
        .await
        .expect("put checkpoint");

    // Verify it exists
    let before = client
        .get_task_checkpoint("rt-delete-task")
        .await
        .expect("get before delete");
    assert!(before.is_some(), "checkpoint should exist before delete");

    client
        .delete_task_checkpoint("rt-delete-task")
        .await
        .expect("delete_task_checkpoint");

    let after = client
        .get_task_checkpoint("rt-delete-task")
        .await
        .expect("get after delete");
    assert!(after.is_none(), "checkpoint should be gone after delete");
}

// ── signals stream ────────────────────────────────────────────────────────────

#[tokio::test]
async fn provision_signals_stream_is_idempotent() {
    let Some(client) = connect().await else {
        return;
    };
    // Calling twice should be fine
    client
        .provision_signals_stream()
        .await
        .expect("provision_signals_stream first");
    client
        .provision_signals_stream()
        .await
        .expect("provision_signals_stream second");
}

#[tokio::test]
async fn publish_signal_and_subscribe_roundtrip() {
    use futures::StreamExt;
    use std::time::Duration;

    let Some(client) = connect().await else {
        return;
    };
    client
        .provision_signals_stream()
        .await
        .expect("provision_signals_stream");

    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();

    // Subscribe first to avoid race
    let mut stream = client
        .subscribe_signals(&task_id, &tenant_id)
        .await
        .expect("subscribe_signals");

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let signal = ResumeSignal {
        task_id: task_id.clone(),
        tenant_id: tenant_id.clone(),
        payload: SignalPayload::WaveContinue(WaveContinueSignal {
            grounding: None,
            mandate_override: None,
        }),
        timeout_at_ms: now_ms + 60_000,
        issued_at_ms: now_ms,
    };

    client
        .publish_signal(&signal)
        .await
        .expect("publish_signal");

    let received = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timeout waiting for signal")
        .expect("stream closed")
        .expect("signal deserialization");

    assert_eq!(received.task_id, task_id);

    // Cleanup consumer
    client
        .delete_signal_consumer(&task_id)
        .await
        .expect("delete_signal_consumer");
}

#[tokio::test]
async fn delete_signal_consumer_after_subscribe() {
    let Some(client) = connect().await else {
        return;
    };
    client
        .provision_signals_stream()
        .await
        .expect("provision_signals_stream");

    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();

    // Subscribe to create consumer
    let _stream = client
        .subscribe_signals(&task_id, &tenant_id)
        .await
        .expect("subscribe_signals");

    // Delete should succeed
    client
        .delete_signal_consumer(&task_id)
        .await
        .expect("delete_signal_consumer");
}

// ── delta checkpoints ─────────────────────────────────────────────────────────

#[tokio::test]
async fn put_checkpoint_delta_base_and_get_latest() {
    let Some(client) = connect().await else {
        return;
    };
    let task_id = TaskId::new();
    let task_id_str = task_id.to_string();
    let cp = make_checkpoint(&task_id_str);

    // seq=0 is always a base
    client
        .put_checkpoint_delta(&task_id_str, &cp, 0)
        .await
        .expect("put_checkpoint_delta seq=0");

    let loaded = client
        .get_latest_checkpoint(&task_id_str)
        .await
        .expect("get_latest_checkpoint")
        .expect("should be Some");
    assert_eq!(loaded.task_id, task_id_str);
    assert_eq!(loaded.phase, "ParallelGeneration");

    // Write seq=1 (delta against seq=0)
    let mut cp2 = cp.clone();
    cp2.phase = "AuditorGate".to_string();
    client
        .put_checkpoint_delta(&task_id_str, &cp2, 1)
        .await
        .expect("put_checkpoint_delta seq=1");

    let latest = client
        .get_latest_checkpoint(&task_id_str)
        .await
        .expect("get_latest_checkpoint seq=1")
        .expect("should be Some");
    assert_eq!(latest.phase, "AuditorGate");

    // Cleanup
    client.delete_task_checkpoint(&task_id_str).await.ok();
}

#[tokio::test]
async fn get_latest_checkpoint_returns_none_when_absent() {
    let Some(client) = connect().await else {
        return;
    };
    let task_id = TaskId::new().to_string();
    let result = client
        .get_latest_checkpoint(&task_id)
        .await
        .expect("get_latest_checkpoint");
    assert!(result.is_none());
}

// ── tenant reasoning buckets ──────────────────────────────────────────────────

const TEST_CHECKPOINT_PREFIX: &str = "H2AI_RTCHK";
const TEST_META_PREFIX: &str = "H2AI_RTMETA";

#[tokio::test]
async fn ensure_tenant_reasoning_buckets_is_idempotent() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::default_tenant();
    client
        .ensure_tenant_reasoning_buckets(&tenant, TEST_CHECKPOINT_PREFIX, TEST_META_PREFIX)
        .await
        .expect("ensure_tenant_reasoning_buckets first");
    client
        .ensure_tenant_reasoning_buckets(&tenant, TEST_CHECKPOINT_PREFIX, TEST_META_PREFIX)
        .await
        .expect("ensure_tenant_reasoning_buckets second");
}

#[tokio::test]
async fn reasoning_checkpoint_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::default_tenant();
    client
        .ensure_tenant_reasoning_buckets(&tenant, TEST_CHECKPOINT_PREFIX, TEST_META_PREFIX)
        .await
        .expect("ensure buckets");

    let task_id = TaskId::new();
    let cp = TaskReasoningCheckpoint::new_created(
        task_id.clone(),
        tenant.clone(),
        vec!["safety".to_string()],
        Some("code".to_string()),
    );

    client
        .put_reasoning_checkpoint(&cp, TEST_CHECKPOINT_PREFIX)
        .await
        .expect("put_reasoning_checkpoint");

    let loaded = client
        .get_reasoning_checkpoint(&task_id, &tenant, TEST_CHECKPOINT_PREFIX)
        .await
        .expect("get_reasoning_checkpoint")
        .expect("should be Some");

    assert_eq!(loaded.task_id, task_id);
    assert_eq!(loaded.phase, ReasoningCheckpointPhase::Created);
    assert_eq!(loaded.domain, Some("code".to_string()));
}

#[tokio::test]
async fn reasoning_checkpoint_get_returns_none_when_absent() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::default_tenant();
    client
        .ensure_tenant_reasoning_buckets(&tenant, TEST_CHECKPOINT_PREFIX, TEST_META_PREFIX)
        .await
        .expect("ensure buckets");

    let task_id = TaskId::new();
    let result = client
        .get_reasoning_checkpoint(&task_id, &tenant, TEST_CHECKPOINT_PREFIX)
        .await
        .expect("get_reasoning_checkpoint");
    assert!(result.is_none());
}

// ── task meta state ───────────────────────────────────────────────────────────

fn make_task_meta_state(task_id: TaskId, tenant_id: TenantId) -> TaskMetaState {
    TaskMetaState {
        task_id,
        tenant_id,
        resolved_at: 1_700_000_000,
        constraint_tags: vec!["accuracy".to_string()],
        domain: Some("factual".to_string()),
        task_quadrant: None,
        shared_understanding: "Test understanding.".to_string(),
        tensions: vec!["tension A".to_string()],
        archetype_results: vec![],
        thinking_iterations: 2,
        retry_count: 0,
        retry_context_that_resolved: None,
        tried_topologies: vec![],
        tau_values_that_converged: None,
        system_context_with_rubric_hash: 12345,
        constraint_corpus_fingerprint: 67890,
    }
}

#[tokio::test]
async fn task_meta_state_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::from("test-meta-putget");
    client
        .ensure_tenant_reasoning_buckets(&tenant, TEST_CHECKPOINT_PREFIX, TEST_META_PREFIX)
        .await
        .expect("ensure buckets");

    let task_id = TaskId::new();
    let meta = make_task_meta_state(task_id.clone(), tenant.clone());

    client
        .put_task_meta_state(&meta, TEST_META_PREFIX)
        .await
        .expect("put_task_meta_state");

    let loaded = client
        .get_task_meta_state(&task_id, &tenant, TEST_META_PREFIX)
        .await
        .expect("get_task_meta_state")
        .expect("should be Some");

    assert_eq!(loaded.task_id, task_id);
    assert_eq!(loaded.shared_understanding, "Test understanding.");
    assert_eq!(loaded.thinking_iterations, 2);
}

#[tokio::test]
async fn task_meta_state_get_returns_none_when_absent() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::from("test-meta-absent");
    client
        .ensure_tenant_reasoning_buckets(&tenant, TEST_CHECKPOINT_PREFIX, TEST_META_PREFIX)
        .await
        .expect("ensure buckets");

    let result = client
        .get_task_meta_state(&TaskId::new(), &tenant, TEST_META_PREFIX)
        .await
        .expect("get_task_meta_state");
    assert!(result.is_none());
}

#[tokio::test]
async fn list_task_meta_states_includes_written_entries() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::from("test-meta-list");
    client
        .ensure_tenant_reasoning_buckets(&tenant, TEST_CHECKPOINT_PREFIX, TEST_META_PREFIX)
        .await
        .expect("ensure buckets");

    // Delete and recreate the meta bucket so this test is isolated from
    // state accumulated by previous test runs.
    let meta_bucket = h2ai_state::nats::tenant_bucket_name(TEST_META_PREFIX, &tenant);
    client
        .delete_kv_bucket(&meta_bucket)
        .await
        .expect("delete meta bucket");
    client
        .ensure_tenant_reasoning_buckets(&tenant, TEST_CHECKPOINT_PREFIX, TEST_META_PREFIX)
        .await
        .expect("recreate buckets");

    let task_id = TaskId::new();
    let meta = make_task_meta_state(task_id.clone(), tenant.clone());
    client
        .put_task_meta_state(&meta, TEST_META_PREFIX)
        .await
        .expect("put_task_meta_state");

    let list = client
        .list_task_meta_states(&tenant, TEST_META_PREFIX, 10)
        .await;
    let ids: Vec<String> = list.iter().map(|m| m.task_id.to_string()).collect();
    assert!(ids.contains(&task_id.to_string()));
}

// ── conflict accumulator ──────────────────────────────────────────────────────

const CONFLICT_BUCKET_PREFIX: &str = "H2AI_RTCONFLICT";

#[tokio::test]
async fn conflict_accumulator_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::default_tenant();
    client
        .ensure_tenant_conflict_bucket(&tenant, CONFLICT_BUCKET_PREFIX)
        .await
        .expect("ensure_tenant_conflict_bucket");

    let acc = ConflictRateAccumulator::new(tenant.clone(), 0.1);
    client
        .put_conflict_accumulator(&acc, CONFLICT_BUCKET_PREFIX)
        .await
        .expect("put_conflict_accumulator");

    let loaded = client
        .get_conflict_accumulator(&tenant, CONFLICT_BUCKET_PREFIX)
        .await
        .expect("get_conflict_accumulator")
        .expect("should be Some");
    assert!((loaded.calibration_floor - 0.1).abs() < 1e-9);
}

#[tokio::test]
async fn conflict_accumulator_returns_none_when_absent() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::from("absent-conflict-tenant-xyz");
    client
        .ensure_tenant_conflict_bucket(&tenant, CONFLICT_BUCKET_PREFIX)
        .await
        .expect("ensure_tenant_conflict_bucket");

    let result = client
        .get_conflict_accumulator(&tenant, CONFLICT_BUCKET_PREFIX)
        .await
        .expect("get_conflict_accumulator");
    assert!(result.is_none());
}

#[tokio::test]
async fn ensure_tenant_conflict_bucket_is_idempotent() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant = TenantId::default_tenant();
    client
        .ensure_tenant_conflict_bucket(&tenant, CONFLICT_BUCKET_PREFIX)
        .await
        .expect("first call");
    client
        .ensure_tenant_conflict_bucket(&tenant, CONFLICT_BUCKET_PREFIX)
        .await
        .expect("second call");
}

// ── event publishing ──────────────────────────────────────────────────────────

#[tokio::test]
async fn publish_event_does_not_error() {
    let Some(client) = connect().await else {
        return;
    };
    let task_id = TaskId::new();
    let event = h2ai_types::events::H2AIEvent::CalibrationCompleted(make_calibration_completed());
    client
        .publish_event(&task_id, &event)
        .await
        .expect("publish_event");
}

#[tokio::test]
async fn publish_to_does_not_error() {
    let Some(client) = connect().await else {
        return;
    };
    let task_id = TaskId::new();
    let subject = format!("h2ai.tasks.{task_id}");
    let event = h2ai_types::events::H2AIEvent::CalibrationCompleted(make_calibration_completed());
    client
        .publish_to(&subject, &event)
        .await
        .expect("publish_to");
}

#[tokio::test]
async fn publish_event_seq_returns_nonzero_sequence() {
    let Some(client) = connect().await else {
        return;
    };
    let task_id = TaskId::new();
    let event = h2ai_types::events::H2AIEvent::CalibrationCompleted(make_calibration_completed());
    let seq = client
        .publish_event_seq(&task_id, &event)
        .await
        .expect("publish_event_seq");
    assert!(seq > 0, "JetStream sequence should be positive");
}
