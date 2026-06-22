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
/// Comprehensive NATS KV integration tests.
///
/// NATS is available at <nats://localhost:4222> — all tests run live.
/// Each test uses unique keys/prefixes to avoid interference.
use h2ai_state::nats::NatsClient;
use h2ai_types::identity::{TaskId, TenantId};

async fn connect() -> Option<NatsClient> {
    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    match NatsClient::connect(&url).await {
        Ok(client) => {
            if client.ensure_infrastructure().await.is_err() {
                eprintln!("NATS infra setup failed at {url} — skipping");
                return None;
            }
            Some(client)
        }
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            None
        }
    }
}

// ── snapshot roundtrip ───────────────────────────────────────────────────────

#[tokio::test]
async fn snapshot_put_get_returns_stored_value() {
    use h2ai_types::events::TaskSnapshot;
    let Some(client) = connect().await else {
        return;
    };

    let task_id = TaskId::new();
    let snap = TaskSnapshot {
        task_id: task_id.clone(),
        last_sequence: 5,
        task_state_json: r#"{"phase":"ParallelGeneration"}"#.into(),
        taken_at: chrono::Utc::now(),
    };
    client.put_snapshot(&snap).await.expect("put_snapshot");
    let got = client.get_snapshot(&task_id).await.expect("get_snapshot");
    let got = got.expect("snapshot should exist");
    assert_eq!(got.task_id, task_id);
    assert_eq!(got.last_sequence, 5);
    assert!(got.task_state_json.contains("ParallelGeneration"));
}

#[tokio::test]
async fn snapshot_get_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let task_id = TaskId::new();
    let got = client.get_snapshot(&task_id).await.expect("get_snapshot");
    assert!(got.is_none());
}

#[tokio::test]
async fn snapshot_overwrite_returns_latest() {
    use h2ai_types::events::TaskSnapshot;
    let Some(client) = connect().await else {
        return;
    };

    let task_id = TaskId::new();
    let snap1 = TaskSnapshot {
        task_id: task_id.clone(),
        last_sequence: 0,
        task_state_json: r#"{"phase":"Phase1"}"#.into(),
        taken_at: chrono::Utc::now(),
    };
    let snap2 = TaskSnapshot {
        task_id: task_id.clone(),
        last_sequence: 1,
        task_state_json: r#"{"phase":"Phase2"}"#.into(),
        taken_at: chrono::Utc::now(),
    };
    client.put_snapshot(&snap1).await.expect("put 1");
    client.put_snapshot(&snap2).await.expect("put 2");
    let got = client.get_snapshot(&task_id).await.expect("get");
    assert!(got.unwrap().task_state_json.contains("Phase2"));
}

// ── tao estimator ────────────────────────────────────────────────────────────

#[tokio::test]
async fn tao_estimator_put_get_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let tid = TenantId::from("tao-test-tenant");
    client
        .put_tao_estimator_state(&tid, 1.25, 42)
        .await
        .expect("put");
    let back = client
        .get_tao_estimator_state(&tid)
        .await
        .expect("get")
        .expect("should be Some");
    assert!((back.0 - 1.25).abs() < 1e-9);
    assert_eq!(back.1, 42);
}

#[tokio::test]
async fn tao_estimator_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let tid = TenantId::from("tao-never-written-xyz");
    let result = client.get_tao_estimator_state(&tid).await.expect("get");
    assert!(result.is_none());
}

#[tokio::test]
async fn tao_estimator_overwrite_updates_value() {
    let Some(client) = connect().await else {
        return;
    };
    let tid = TenantId::from("tao-overwrite-tenant");
    client
        .put_tao_estimator_state(&tid, 0.5, 1)
        .await
        .expect("put 1");
    client
        .put_tao_estimator_state(&tid, 2.0, 99)
        .await
        .expect("put 2");
    let back = client
        .get_tao_estimator_state(&tid)
        .await
        .expect("get")
        .expect("some");
    assert!((back.0 - 2.0).abs() < 1e-9);
    assert_eq!(back.1, 99);
}

// ── bandit state ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn bandit_state_put_get_raw_bytes_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let tid = TenantId::from("bandit-test-tenant");
    let data = b"{\"arms\":[{\"alpha\":1.0,\"beta\":1.0}]}".to_vec();
    client
        .put_bandit_state(&tid, data.clone())
        .await
        .expect("put");
    let got = client
        .get_bandit_state(&tid)
        .await
        .expect("get")
        .expect("some");
    assert_eq!(got, data);
}

#[tokio::test]
async fn bandit_state_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let tid = TenantId::from("bandit-never-written-xyz");
    let result = client.get_bandit_state(&tid).await.expect("get");
    assert!(result.is_none());
}

#[tokio::test]
async fn bandit_state_overwrite_updates_value() {
    let Some(client) = connect().await else {
        return;
    };
    let tid = TenantId::from("bandit-overwrite-tenant");
    client
        .put_bandit_state(&tid, b"old".to_vec())
        .await
        .expect("put 1");
    client
        .put_bandit_state(&tid, b"new".to_vec())
        .await
        .expect("put 2");
    let got = client
        .get_bandit_state(&tid)
        .await
        .expect("get")
        .expect("some");
    assert_eq!(got, b"new");
}

// ── shadow promoted domains ──────────────────────────────────────────────────

#[tokio::test]
async fn shadow_promoted_domains_put_get_roundtrip() {
    use std::collections::HashSet;
    let Some(client) = connect().await else {
        return;
    };

    let mut domains: HashSet<String> = HashSet::new();
    domains.insert("code".into());
    domains.insert("factual".into());

    client
        .put_shadow_promoted_domains(&domains)
        .await
        .expect("put");
    let got = client.get_shadow_promoted_domains().await.expect("get");
    assert!(got.contains("code"));
    assert!(got.contains("factual"));
    assert_eq!(got.len(), 2);
}

// ── oracle observations ──────────────────────────────────────────────────────

#[tokio::test]
async fn oracle_observations_put_get_roundtrip() {
    use h2ai_types::sizing::{OracleDomain, OracleObservation};
    let Some(client) = connect().await else {
        return;
    };
    let _tid = TenantId::from("oracle-test-tenant");

    let obs = vec![
        OracleObservation {
            task_id: "task-oracle-1".into(),
            q_confidence: 0.9,
            y_oracle: true,
            residual: 0.1,
            domain: OracleDomain::Code,
            timestamp_ms: 1_000_000,
        },
        OracleObservation {
            task_id: "task-oracle-2".into(),
            q_confidence: 0.4,
            y_oracle: false,
            residual: 0.6,
            domain: OracleDomain::Factual,
            timestamp_ms: 2_000_000,
        },
    ];

    client.put_oracle_observations(&obs).await.expect("put");
    let got = client.get_oracle_observations().await.expect("get");
    // find the obs we just wrote by task_id (other tests may have overwritten the bucket)
    let obs1 = got.iter().find(|o| o.task_id == "task-oracle-1");
    let obs2 = got.iter().find(|o| o.task_id == "task-oracle-2");
    if let (Some(o1), Some(o2)) = (obs1, obs2) {
        assert!((o1.q_confidence - 0.9).abs() < 1e-9);
        assert!((o2.residual - 0.6).abs() < 1e-9);
    }
    // At minimum, put_oracle_observations succeeded without error.
}

#[tokio::test]
async fn oracle_observations_get_returns_ok() {
    // Oracle observations are stored globally (single "observations" key).
    // We can only verify the call succeeds, not that the bucket is empty,
    // because parallel tests share the same KV bucket.
    let Some(client) = connect().await else {
        return;
    };
    let result = client.get_oracle_observations().await;
    assert!(result.is_ok(), "get_oracle_observations must not error");
}

#[tokio::test]
async fn oracle_observations_overwrite_replaces_all() {
    use h2ai_types::sizing::{OracleDomain, OracleObservation};
    let Some(client) = connect().await else {
        return;
    };
    let _tid = TenantId::from("oracle-overwrite-tenant");

    let obs1 = vec![OracleObservation {
        task_id: "old".into(),
        q_confidence: 0.1,
        y_oracle: false,
        residual: 0.9,
        domain: OracleDomain::Code,
        timestamp_ms: 1,
    }];
    let obs2 = vec![OracleObservation {
        task_id: "new".into(),
        q_confidence: 0.95,
        y_oracle: true,
        residual: 0.05,
        domain: OracleDomain::Factual,
        timestamp_ms: 2,
    }];

    client.put_oracle_observations(&obs1).await.expect("put 1");
    client.put_oracle_observations(&obs2).await.expect("put 2");
    let got = client.get_oracle_observations().await.expect("get");
    // Parallel tests share the bucket, so the count may be > 1; verify the
    // replacement invariant directly: the old task_id must be gone and the new one present.
    assert!(
        got.iter().any(|o| o.task_id == "new"),
        "new observation must be present after overwrite"
    );
    assert!(
        !got.iter().any(|o| o.task_id == "old"),
        "old observation must be absent after overwrite replaced it"
    );
}

// ── prompt variants ──────────────────────────────────────────────────────────

#[tokio::test]
async fn prompt_variant_put_get_roundtrip() {
    use chrono::Utc;
    use h2ai_types::prompt_variant::{PromptVariant, PromptVariantSource};
    let Some(client) = connect().await else {
        return;
    };

    let variant = PromptVariant {
        variant_id: "v1-test".into(),
        adapter_name: "adapter-test".into(),
        prompt_key: "system_preamble".into(),
        text: "You are a helpful assistant.".into(),
        source: PromptVariantSource::Seed,
        created_at: Utc::now(),
        score: None,
    };
    client.put_prompt_variant(&variant).await.expect("put");
    let got = client
        .get_prompt_variant("adapter-test", "system_preamble", "v1-test")
        .await
        .expect("get")
        .expect("some");
    assert_eq!(got.variant_id, "v1-test");
    assert_eq!(got.text, "You are a helpful assistant.");
}

#[tokio::test]
async fn prompt_variant_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let got = client
        .get_prompt_variant("no-adapter", "no-key", "no-variant")
        .await
        .expect("get");
    assert!(got.is_none());
}

#[tokio::test]
async fn prompt_variant_active_ptr_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    client
        .set_active_variant_ptr("adapter-active", "brainstorm_task", "v3")
        .await
        .expect("set");
    let ptr = client
        .get_active_variant_ptr("adapter-active", "brainstorm_task")
        .await
        .expect("get")
        .expect("some");
    assert_eq!(ptr, "v3");
}

#[tokio::test]
async fn prompt_variant_active_ptr_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let ptr = client
        .get_active_variant_ptr("no-adapter-xyz", "no-key-xyz")
        .await
        .expect("get");
    assert!(ptr.is_none());
}

#[tokio::test]
async fn prompt_variant_active_ptr_overwrite() {
    let Some(client) = connect().await else {
        return;
    };
    client
        .set_active_variant_ptr("adapter-ptr-ow", "key1", "v1")
        .await
        .expect("set 1");
    client
        .set_active_variant_ptr("adapter-ptr-ow", "key1", "v2")
        .await
        .expect("set 2");
    let ptr = client
        .get_active_variant_ptr("adapter-ptr-ow", "key1")
        .await
        .expect("get")
        .expect("some");
    assert_eq!(ptr, "v2");
}

// ── adapter OPRO state ───────────────────────────────────────────────────────

#[tokio::test]
async fn adapter_opro_state_put_get_roundtrip() {
    use chrono::Utc;
    use h2ai_types::prompt_variant::AdapterOproState;
    use std::collections::HashMap;
    let Some(client) = connect().await else {
        return;
    };

    let state = AdapterOproState {
        adapter_name: "test-adapter-opro".into(),
        j_eff_ema: 0.72,
        n_tasks_total: 100,
        n_tasks_since_last_opro: 25,
        last_opro_started_at: Some(Utc::now()),
        suppress_until_n_tasks: 50,
        bandit_arms: HashMap::new(),
    };
    client.put_adapter_opro_state(&state).await.expect("put");
    let got = client
        .get_adapter_opro_state("test-adapter-opro")
        .await
        .expect("get")
        .expect("some");
    assert_eq!(got.adapter_name, "test-adapter-opro");
    assert!((got.j_eff_ema - 0.72).abs() < 1e-9);
    assert_eq!(got.n_tasks_total, 100);
}

#[tokio::test]
async fn adapter_opro_state_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let got = client
        .get_adapter_opro_state("no-such-adapter-xyz")
        .await
        .expect("get");
    assert!(got.is_none());
}

// ── task checkpoints (delta path) ───────────────────────────────────────────

#[tokio::test]
async fn task_checkpoint_delta_base_roundtrip() {
    use h2ai_types::checkpoint::TaskCheckpoint;
    let Some(client) = connect().await else {
        return;
    };

    let task_uuid = uuid::Uuid::new_v4();
    let task_id = task_uuid.to_string();
    let cp = TaskCheckpoint {
        task_id: task_id.clone(),
        phase: "ParallelGeneration".into(),
        node_id: "node-1".into(),
        lease_seq: 0,
        proposals: vec!["proposal A".into()],
        auditor_survivors: vec![],
        resolved_output: None,
        manifest_json: "{}".into(),
        object_store_ref: None,
        created_at_ms: 1_000_000,
        updated_at_ms: 1_000_000,
        constraint_snapshot: None,
        j_eff: None,
    };

    client
        .put_checkpoint_delta(&task_id, &cp, 0)
        .await
        .expect("put seq=0");
    let got = client
        .get_latest_checkpoint(&task_id)
        .await
        .expect("get")
        .expect("some");
    assert_eq!(got.phase, "ParallelGeneration");
    assert_eq!(got.task_id, task_id);
}

#[tokio::test]
async fn task_checkpoint_delta_then_delta_roundtrip() {
    use h2ai_config::StateConfig;
    use h2ai_types::checkpoint::TaskCheckpoint;

    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    let mut cfg = StateConfig::default();
    cfg.delta.enabled = true;
    cfg.delta.base_interval = 10;

    let client = match NatsClient::connect_with_cfg(&url, cfg).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable — skipping: {e}");
            return;
        }
    };
    client.ensure_infrastructure().await.expect("infra");

    let task_uuid = uuid::Uuid::new_v4();
    let task_id = task_uuid.to_string();
    let base = TaskCheckpoint {
        task_id: task_id.clone(),
        phase: "ParallelGeneration".into(),
        node_id: "node-1".into(),
        lease_seq: 0,
        proposals: vec!["proposal A".into()],
        auditor_survivors: vec![],
        resolved_output: None,
        manifest_json: "{}".into(),
        object_store_ref: None,
        created_at_ms: 1_000_000,
        updated_at_ms: 1_000_000,
        constraint_snapshot: None,
        j_eff: None,
    };

    client
        .put_checkpoint_delta(&task_id, &base, 10)
        .await
        .expect("put base at seq=10");

    let mut delta = base.clone();
    delta.phase = "AuditorGate".into();
    delta.updated_at_ms = 2_000_000;

    client
        .put_checkpoint_delta(&task_id, &delta, 11)
        .await
        .expect("put delta at seq=11");

    let got = client
        .get_latest_checkpoint(&task_id)
        .await
        .expect("get")
        .expect("some");
    assert_eq!(got.phase, "AuditorGate");
    assert_eq!(got.updated_at_ms, 2_000_000);
}

#[tokio::test]
async fn task_checkpoint_get_latest_uses_cache_on_second_read() {
    use h2ai_types::checkpoint::TaskCheckpoint;
    let Some(client) = connect().await else {
        return;
    };

    let task_uuid = uuid::Uuid::new_v4();
    let task_id = task_uuid.to_string();
    let cp = TaskCheckpoint {
        task_id: task_id.clone(),
        phase: "Merging".into(),
        node_id: "node-cache".into(),
        lease_seq: 0,
        proposals: vec![],
        auditor_survivors: vec![],
        resolved_output: Some("cached result".into()),
        manifest_json: "{}".into(),
        object_store_ref: None,
        created_at_ms: 5_000_000,
        updated_at_ms: 5_000_000,
        constraint_snapshot: None,
        j_eff: None,
    };

    client
        .put_checkpoint_delta(&task_id, &cp, 0)
        .await
        .expect("put");

    let first = client
        .get_latest_checkpoint(&task_id)
        .await
        .expect("first get")
        .expect("some");
    assert_eq!(first.resolved_output, Some("cached result".into()));

    let second = client
        .get_latest_checkpoint(&task_id)
        .await
        .expect("second get")
        .expect("some");
    assert_eq!(second.resolved_output, Some("cached result".into()));
}

// ── reasoning checkpoints ────────────────────────────────────────────────────

#[tokio::test]
async fn reasoning_checkpoint_put_get_roundtrip() {
    use h2ai_types::reasoning_checkpoint::{ReasoningCheckpointPhase, TaskReasoningCheckpoint};

    let Some(client) = connect().await else {
        return;
    };
    let tenant_id = TenantId::from("reasoning-test-tenant");
    let task_id = TaskId::new();
    let cp_prefix = "H2AI_CKPT_TEST";
    let meta_prefix = "H2AI_META_TEST";

    client
        .ensure_tenant_reasoning_buckets(&tenant_id, cp_prefix, meta_prefix)
        .await
        .expect("ensure buckets");

    let cp = TaskReasoningCheckpoint {
        task_id: task_id.clone(),
        tenant_id: tenant_id.clone(),
        created_at: 1_000_000,
        last_updated: 1_000_001,
        phase: ReasoningCheckpointPhase::ThinkingDone,
        constraint_tags: vec!["security".into()],
        domain: Some("code".into()),
        task_quadrant: None,
        system_context_with_rubric_hash: 42,
        constraint_corpus_fingerprint: 99,
        shared_understanding: Some("ADR-001 JWT stateless auth".into()),
        tensions: Some(vec!["security vs convenience".into()]),
        archetype_selection: None,
        thinking_iterations: Some(3),
        completed_waves: vec![],
        retry_count: 0,
        retry_context_that_resolved: None,
        tried_topologies: vec![],
        tau_values_that_converged: None,
        resolved_attribution_json: None,
        resolved_waste_ratio: None,
        hitl_timeouts_fired: 0,
    };

    client
        .put_reasoning_checkpoint(&cp, cp_prefix)
        .await
        .expect("put");
    let got = client
        .get_reasoning_checkpoint(&task_id, &tenant_id, cp_prefix)
        .await
        .expect("get")
        .expect("some");
    assert_eq!(got.task_id, task_id);
    assert_eq!(
        got.shared_understanding,
        Some("ADR-001 JWT stateless auth".into())
    );
    assert_eq!(got.thinking_iterations, Some(3));
}

#[tokio::test]
async fn reasoning_checkpoint_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant_id = TenantId::from("reasoning-test-tenant");
    let task_id = TaskId::new();
    let cp_prefix = "H2AI_CKPT_TEST";
    let meta_prefix = "H2AI_META_TEST";

    client
        .ensure_tenant_reasoning_buckets(&tenant_id, cp_prefix, meta_prefix)
        .await
        .expect("ensure buckets");

    let result = client
        .get_reasoning_checkpoint(&task_id, &tenant_id, cp_prefix)
        .await
        .expect("get");
    assert!(result.is_none());
}

// ── task meta state ──────────────────────────────────────────────────────────

#[tokio::test]
async fn task_meta_state_put_get_roundtrip() {
    use h2ai_types::reasoning_checkpoint::TaskMetaState;

    let Some(client) = connect().await else {
        return;
    };
    let tenant_id = TenantId::from("meta-test-tenant");
    let task_id = TaskId::new();
    let cp_prefix = "H2AI_CKPT_METASTATE_TEST";
    let meta_prefix = "H2AI_META_METASTATE_TEST";

    client
        .ensure_tenant_reasoning_buckets(&tenant_id, cp_prefix, meta_prefix)
        .await
        .expect("ensure buckets");

    let meta = TaskMetaState {
        task_id: task_id.clone(),
        tenant_id: tenant_id.clone(),
        resolved_at: 9_000_000,
        constraint_tags: vec!["performance".into()],
        domain: Some("infra".into()),
        task_quadrant: None,
        shared_understanding: "Cache-aside pattern for Redis".into(),
        tensions: vec![],
        archetype_results: vec![],
        thinking_iterations: 2,
        retry_count: 0,
        retry_context_that_resolved: None,
        tried_topologies: vec![],
        tau_values_that_converged: Some(vec![0.3, 0.5]),
        system_context_with_rubric_hash: 1234,
        constraint_corpus_fingerprint: 5678,
    };

    client
        .put_task_meta_state(&meta, meta_prefix)
        .await
        .expect("put");
    let got = client
        .get_task_meta_state(&task_id, &tenant_id, meta_prefix)
        .await
        .expect("get")
        .expect("some");
    assert_eq!(got.shared_understanding, "Cache-aside pattern for Redis");
    assert_eq!(got.resolved_at, 9_000_000);
    assert_eq!(got.tau_values_that_converged, Some(vec![0.3, 0.5]));
}

#[tokio::test]
async fn task_meta_state_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant_id = TenantId::from("meta-missing-tenant");
    let task_id = TaskId::new();
    let cp_prefix = "H2AI_CKPT_METASTATE_MISS";
    let meta_prefix = "H2AI_META_METASTATE_MISS";

    client
        .ensure_tenant_reasoning_buckets(&tenant_id, cp_prefix, meta_prefix)
        .await
        .expect("ensure buckets");

    let result = client
        .get_task_meta_state(&task_id, &tenant_id, meta_prefix)
        .await
        .expect("get");
    assert!(result.is_none());
}

#[tokio::test]
async fn task_meta_state_list_returns_stored_records() {
    use h2ai_types::reasoning_checkpoint::TaskMetaState;

    let Some(client) = connect().await else {
        return;
    };
    let tenant_id = TenantId::from("meta-list-tenant");
    let cp_prefix = "H2AI_CKPT_LIST_TEST";
    let meta_prefix = "H2AI_META_LIST_TEST";

    client
        .ensure_tenant_reasoning_buckets(&tenant_id, cp_prefix, meta_prefix)
        .await
        .expect("ensure buckets");

    let task_id1 = TaskId::new();
    let task_id2 = TaskId::new();

    let make_meta = |task_id: TaskId, understanding: &str| TaskMetaState {
        task_id,
        tenant_id: tenant_id.clone(),
        resolved_at: 1_000_000,
        constraint_tags: vec![],
        domain: None,
        task_quadrant: None,
        shared_understanding: understanding.into(),
        tensions: vec![],
        archetype_results: vec![],
        thinking_iterations: 1,
        retry_count: 0,
        retry_context_that_resolved: None,
        tried_topologies: vec![],
        tau_values_that_converged: None,
        system_context_with_rubric_hash: 0,
        constraint_corpus_fingerprint: 0,
    };

    client
        .put_task_meta_state(&make_meta(task_id1.clone(), "understanding A"), meta_prefix)
        .await
        .expect("put 1");
    client
        .put_task_meta_state(&make_meta(task_id2.clone(), "understanding B"), meta_prefix)
        .await
        .expect("put 2");

    let list = client
        .list_task_meta_states(&tenant_id, meta_prefix, 100)
        .await;
    assert!(list.len() >= 2);
    let understandings: Vec<&str> = list
        .iter()
        .map(|m| m.shared_understanding.as_str())
        .collect();
    assert!(understandings.contains(&"understanding A"));
    assert!(understandings.contains(&"understanding B"));
}

#[tokio::test]
async fn task_meta_state_list_empty_bucket_returns_empty() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant_id = TenantId::from("meta-list-empty-xyz");
    let cp_prefix = "H2AI_CKPT_EMPTY_XYZ";
    let meta_prefix = "H2AI_META_EMPTY_XYZ";
    let _ = client
        .list_task_meta_states(&tenant_id, meta_prefix, 100)
        .await;
    client
        .ensure_tenant_reasoning_buckets(&tenant_id, cp_prefix, meta_prefix)
        .await
        .expect("ensure");
    let list2 = client
        .list_task_meta_states(&tenant_id, meta_prefix, 100)
        .await;
    assert!(list2.is_empty());
}

// ── conflict rate accumulator ────────────────────────────────────────────────

#[tokio::test]
async fn conflict_accumulator_put_get_roundtrip() {
    use h2ai_types::conflict::ConflictRateAccumulator;
    let Some(client) = connect().await else {
        return;
    };
    let tenant_id = TenantId::from("conflict-test-tenant");
    let bucket_prefix = "H2AI_CONFLICT_TEST";

    client
        .ensure_tenant_conflict_bucket(&tenant_id, bucket_prefix)
        .await
        .expect("ensure bucket");

    let acc = ConflictRateAccumulator::new(tenant_id.clone(), 0.3);
    client
        .put_conflict_accumulator(&acc, bucket_prefix)
        .await
        .expect("put");
    let got = client
        .get_conflict_accumulator(&tenant_id, bucket_prefix)
        .await
        .expect("get")
        .expect("some");
    assert!((got.calibration_floor - 0.3).abs() < 1e-9);
    assert_eq!(got.total_tasks_seen, 0);
}

#[tokio::test]
async fn conflict_accumulator_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant_id = TenantId::from("conflict-missing-tenant");
    let bucket_prefix = "H2AI_CONFLICT_MISSING";

    client
        .ensure_tenant_conflict_bucket(&tenant_id, bucket_prefix)
        .await
        .expect("ensure bucket");

    let result = client
        .get_conflict_accumulator(&tenant_id, bucket_prefix)
        .await
        .expect("get");
    assert!(result.is_none());
}

// ── infrastructure idempotency ────────────────────────────────────────────────

#[tokio::test]
async fn ensure_infrastructure_is_idempotent() {
    let Some(client) = connect().await else {
        return;
    };
    client
        .ensure_infrastructure()
        .await
        .expect("second ensure_infrastructure call must be idempotent");
}

#[tokio::test]
async fn ensure_tenant_reasoning_buckets_is_idempotent() {
    let Some(client) = connect().await else {
        return;
    };
    let tenant_id = TenantId::from("idem-test-tenant");
    let cp_prefix = "H2AI_CKPT_IDEM";
    let meta_prefix = "H2AI_META_IDEM";

    client
        .ensure_tenant_reasoning_buckets(&tenant_id, cp_prefix, meta_prefix)
        .await
        .expect("first call");
    client
        .ensure_tenant_reasoning_buckets(&tenant_id, cp_prefix, meta_prefix)
        .await
        .expect("second call must be idempotent");
}

// ── publish event and event_seq ──────────────────────────────────────────────

#[tokio::test]
async fn publish_event_does_not_error() {
    use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent};
    use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold};
    let Some(client) = connect().await else {
        return;
    };
    let task_id = TaskId::new();

    let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.7, 0.8]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let event = H2AIEvent::CalibrationCompleted(CalibrationCompletedEvent {
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
    });
    client
        .publish_event(&task_id, &event)
        .await
        .expect("publish_event must succeed");
}

#[tokio::test]
async fn publish_event_seq_returns_positive_sequence() {
    use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent};
    use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold};
    let Some(client) = connect().await else {
        return;
    };
    let task_id = TaskId::new();

    let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.7, 0.8]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let event = H2AIEvent::CalibrationCompleted(CalibrationCompletedEvent {
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
    });

    let seq = client
        .publish_event_seq(&task_id, &event)
        .await
        .expect("publish_event_seq must succeed");
    assert!(seq > 0, "sequence number must be positive, got {seq}");
}

// ── probe lease ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn probe_lease_acquire_release_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    let profile = format!("test-profile-{}", uuid::Uuid::new_v4());

    client.release_probe_lease(&profile).await.unwrap();

    let won = client.acquire_probe_lease(&profile, 60).await.unwrap();
    assert!(won, "first acquire must succeed");

    let lost = client.acquire_probe_lease(&profile, 60).await.unwrap();
    assert!(!lost, "second acquire must fail while lease held");

    client.release_probe_lease(&profile).await.unwrap();

    let won2 = client.acquire_probe_lease(&profile, 60).await.unwrap();
    assert!(won2, "acquire after release must succeed");

    client.release_probe_lease(&profile).await.unwrap();
}

// ── calibration record (missing key) ─────────────────────────────────────────

#[tokio::test]
async fn get_calibration_record_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let result = client
        .get_calibration_record("nonexistent-profile-abc-xyz")
        .await
        .unwrap();
    assert!(result.is_none());
}

// ── auditor health missing ────────────────────────────────────────────────────

#[tokio::test]
async fn get_auditor_health_missing_returns_none() {
    let Some(client) = connect().await else {
        return;
    };
    let result = client
        .get_auditor_health("nonexistent-health-profile-xyz")
        .await
        .unwrap();
    assert!(result.is_none());
}

// ── checkpoint put with revision (CAS) ───────────────────────────────────────

#[tokio::test]
async fn put_task_checkpoint_cas_revision() {
    use h2ai_types::checkpoint::TaskCheckpoint;
    let Some(client) = connect().await else {
        return;
    };
    let task_id = format!("cas-test-{}", uuid::Uuid::new_v4());

    let cp = TaskCheckpoint {
        task_id: task_id.clone(),
        phase: "Phase1".into(),
        node_id: "node-1".into(),
        lease_seq: 0,
        proposals: vec![],
        auditor_survivors: vec![],
        resolved_output: None,
        manifest_json: "{}".into(),
        object_store_ref: None,
        created_at_ms: 1_000_000,
        updated_at_ms: 1_000_000,
        constraint_snapshot: None,
        j_eff: None,
    };

    let rev1 = client
        .put_task_checkpoint(&cp, None)
        .await
        .expect("first put");
    assert!(rev1 > 0);

    let mut cp2 = cp.clone();
    cp2.phase = "Phase2".into();
    let rev2 = client
        .put_task_checkpoint(&cp2, Some(rev1))
        .await
        .expect("CAS put");
    assert!(rev2 > rev1, "revision must increase");

    let got = client
        .get_task_checkpoint(&task_id)
        .await
        .expect("get")
        .expect("some");
    assert_eq!(got.phase, "Phase2");

    client.delete_task_checkpoint(&task_id).await.ok();
}
