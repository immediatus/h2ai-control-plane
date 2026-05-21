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
use h2ai_state::{
    apply_patches, generate_delta, should_store_base, tenant_bucket_name, CachedCheckpoint,
};
use h2ai_types::checkpoint::TaskCheckpoint;
use h2ai_types::identity::TenantId;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::RwLock;

fn minimal_checkpoint() -> TaskCheckpoint {
    TaskCheckpoint {
        task_id: "task-001".into(),
        phase: "ParallelGeneration".into(),
        node_id: "node-1".into(),
        lease_seq: 1,
        proposals: vec!["proposal A".into()],
        auditor_survivors: vec![],
        resolved_output: None,
        manifest_json: "{}".into(),
        object_store_ref: None,
        created_at_ms: 1_000_000,
        updated_at_ms: 1_000_000,
        constraint_snapshot: None,
        j_eff: None,
    }
}

// ── delta encoding ────────────────────────────────────────────────────────────

#[test]
fn should_store_base_seq_zero() {
    assert!(should_store_base(0, 10));
}

#[test]
fn should_store_base_at_interval() {
    assert!(should_store_base(10, 10));
    assert!(should_store_base(20, 10));
    assert!(should_store_base(100, 10));
}

#[test]
fn should_store_base_not_at_interval() {
    assert!(!should_store_base(5, 10));
    assert!(!should_store_base(1, 10));
    assert!(!should_store_base(9, 10));
}

#[test]
fn generate_delta_no_change() {
    let cp = minimal_checkpoint();
    let patch = generate_delta(&cp, &cp).expect("generate_delta");
    assert_eq!(
        patch.0.len(),
        0,
        "identical checkpoints should produce empty patch"
    );
}

#[test]
fn generate_delta_single_field_changed() {
    let base = minimal_checkpoint();
    let mut modified = base.clone();
    modified.phase = "AuditorGate".into();

    let patch = generate_delta(&base, &modified).expect("generate_delta");
    assert_eq!(patch.0.len(), 1, "one field changed → one patch operation");

    let op = &patch.0[0];
    let op_json = serde_json::to_value(op).unwrap();
    assert_eq!(op_json["op"], "replace");
    assert_eq!(op_json["path"], "/phase");
    assert_eq!(op_json["value"], "AuditorGate");
}

#[test]
fn apply_patches_roundtrip() {
    let base = minimal_checkpoint();
    let mut modified = base.clone();
    modified.phase = "Merging".into();
    modified.resolved_output = Some("final answer".into());
    modified.updated_at_ms = 2_000_000;

    let patch = generate_delta(&base, &modified).expect("generate_delta");
    let reconstructed = apply_patches(&base, &[patch]).expect("apply_patches");

    assert_eq!(reconstructed.phase, "Merging");
    assert_eq!(reconstructed.resolved_output, Some("final answer".into()));
    assert_eq!(reconstructed.updated_at_ms, 2_000_000);
    assert_eq!(reconstructed.task_id, base.task_id);
    assert_eq!(reconstructed.proposals, base.proposals);
}

#[test]
fn apply_patches_empty_patch() {
    let base = minimal_checkpoint();
    let empty_patch = json_patch::Patch(vec![]);
    let result = apply_patches(&base, &[empty_patch]).expect("apply_patches");
    assert_eq!(result, base);
}

// ── delta cache unit tests ────────────────────────────────────────────────────

fn make_checkpoint(task_id: &str) -> TaskCheckpoint {
    TaskCheckpoint {
        task_id: task_id.into(),
        phase: "Merging".into(),
        node_id: "node-1".into(),
        lease_seq: 0,
        proposals: vec!["prop".into()],
        auditor_survivors: vec![],
        resolved_output: None,
        manifest_json: "{}".into(),
        object_store_ref: None,
        created_at_ms: 1000,
        updated_at_ms: 1000,
        constraint_snapshot: None,
        j_eff: None,
    }
}

#[tokio::test]
async fn cache_invalidated_on_write() {
    let cache: Arc<RwLock<LruCache<String, CachedCheckpoint>>> =
        Arc::new(RwLock::new(LruCache::new(NonZeroUsize::new(10).unwrap())));

    let cp = make_checkpoint("task-cache-test");

    cache.write().await.put(
        "task-cache-test".to_string(),
        CachedCheckpoint {
            checkpoint: cp.clone(),
            seq: 5,
            cached_at: std::time::Instant::now(),
        },
    );
    assert!(
        cache.write().await.get("task-cache-test").is_some(),
        "cache should be populated after put"
    );

    cache.write().await.pop("task-cache-test");
    assert!(
        cache.write().await.get("task-cache-test").is_none(),
        "cache should be empty after pop (invalidation)"
    );
}

#[tokio::test]
async fn cache_ttl_expired_entry_treated_as_miss() {
    let cache: Arc<RwLock<LruCache<String, CachedCheckpoint>>> =
        Arc::new(RwLock::new(LruCache::new(NonZeroUsize::new(10).unwrap())));
    let cp = make_checkpoint("task-ttl-test");

    let past = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_hours(1))
        .unwrap_or_else(std::time::Instant::now);
    cache.write().await.put(
        "task-ttl-test".to_string(),
        CachedCheckpoint {
            checkpoint: cp,
            seq: 3,
            cached_at: past,
        },
    );

    let ttl = std::time::Duration::from_mins(1);
    let expired = {
        let mut guard = cache.write().await;
        if let Some(cached) = guard.get("task-ttl-test") {
            cached.cached_at.elapsed() >= ttl
        } else {
            false
        }
    };
    assert!(
        expired,
        "entry older than TTL should be detected as expired"
    );
}

#[test]
fn lru_evicts_oldest_entry_at_capacity() {
    let mut lru: LruCache<String, u32> = LruCache::new(NonZeroUsize::new(2).unwrap());
    lru.put("a".to_string(), 1);
    lru.put("b".to_string(), 2);
    lru.get("a");
    lru.put("c".to_string(), 3);
    assert!(lru.get("a").is_some(), "'a' should survive (recently used)");
    assert!(lru.get("b").is_none(), "'b' should be evicted (LRU)");
    assert!(
        lru.get("c").is_some(),
        "'c' should be present (just inserted)"
    );
}

// ── tenant key helpers ────────────────────────────────────────────────────────

fn kv_key(tenant: &TenantId, suffix: &str) -> String {
    format!("{}/{}", tenant.bucket_safe(), suffix)
}

#[test]
fn hyphen_sanitized_to_underscore() {
    assert_eq!(
        kv_key(&TenantId::from("acme-corp"), "srani"),
        "acme_corp/srani"
    );
}

#[test]
fn default_tenant_key() {
    assert_eq!(
        kv_key(&TenantId::default_tenant(), "bandit"),
        "default/bandit"
    );
}

#[test]
fn approval_key_includes_task_id() {
    let tenant = TenantId::from("acme");
    assert_eq!(kv_key(&tenant, "abc-123"), "acme/abc-123");
}

// ── tenant_bucket_name ────────────────────────────────────────────────────────

#[test]
fn tenant_bucket_name_default() {
    let name = tenant_bucket_name("H2AI_CHECKPOINT", &TenantId::default_tenant());
    assert_eq!(name, "H2AI_CHECKPOINT_default");
}

#[test]
fn tenant_bucket_name_sanitizes_hyphens() {
    let t = TenantId::from("acme-corp");
    let name = tenant_bucket_name("H2AI_META", &t);
    assert_eq!(name, "H2AI_META_acme_corp");
}

#[test]
fn tenant_bucket_name_conflict_default() {
    let name = tenant_bucket_name("H2AI_CONFLICT", &TenantId::default_tenant());
    assert_eq!(name, "H2AI_CONFLICT_default");
}
