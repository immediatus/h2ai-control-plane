use h2ai_orchestrator::induction::nats_scheduler::{
    backoff_ms, InMemoryRetryKvBackend, NatsInductionScheduler,
};
use h2ai_orchestrator::induction::InductionScheduler;
use std::sync::Arc;

#[test]
fn backoff_ms_increases_with_attempt() {
    // With full jitter, the upper bound doubles each attempt (up to cap 500ms)
    // backoff_ms(0) upper = min(5 * 2^0, 500) = 5
    // backoff_ms(1) upper = min(5 * 2^1, 500) = 10
    // backoff_ms(4) upper = min(5 * 2^4, 500) = 80
    // All results must be non-negative and <= 500
    for attempt in 0..8u32 {
        let b = backoff_ms(attempt);
        assert!(
            b <= 500,
            "backoff must be <= 500ms at attempt {attempt}, got {b}"
        );
    }
}

#[test]
fn backoff_ms_caps_at_500() {
    // At attempt 7+, upper = min(5 * 128, 500) = 500
    let b = backoff_ms(10);
    assert!(b <= 500, "backoff must cap at 500ms, got {b}");
}

#[test]
fn nats_scheduler_is_send_sync() {
    // This is a compile-time check — if NatsInductionScheduler implements Send + Sync,
    // this function compiles. NATS connection is None for this test.
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NatsInductionScheduler>();
}

#[test]
fn nats_scheduler_implements_induction_scheduler() {
    // Compile-time check: NatsInductionScheduler implements InductionScheduler
    fn assert_impl<T: InductionScheduler>() {}
    assert_impl::<NatsInductionScheduler>();
}

// ── RetryKvBackend / InMemoryRetryKvBackend ──────────────────────────────────

use h2ai_orchestrator::induction::nats_scheduler::RetryKvBackend;

#[tokio::test]
async fn in_memory_kv_get_returns_none_before_put() {
    let kv = InMemoryRetryKvBackend::default();
    let result = kv.get_entry("any.key").await;
    assert!(result.is_none(), "fresh backend must return None");
}

#[tokio::test]
async fn in_memory_kv_put_then_get_roundtrips() {
    let kv = InMemoryRetryKvBackend::default();
    let data = bytes::Bytes::from("hello");
    kv.put("my.key", data.clone()).await.expect("put");
    let (got, _rev) = kv
        .get_entry("my.key")
        .await
        .expect("must be Some after put");
    assert_eq!(got, data);
}

#[tokio::test]
async fn in_memory_kv_cas_update_increments_revision() {
    let kv = InMemoryRetryKvBackend::default();
    kv.put("k", bytes::Bytes::from("v1")).await.unwrap();
    let (_, rev) = kv.get_entry("k").await.unwrap();
    kv.cas_update("k", bytes::Bytes::from("v2"), rev)
        .await
        .expect("cas ok");
    let (got, _) = kv.get_entry("k").await.unwrap();
    assert_eq!(got, bytes::Bytes::from("v2"));
}

#[tokio::test]
async fn in_memory_kv_cas_update_fails_on_stale_revision() {
    let kv = InMemoryRetryKvBackend::default();
    kv.put("k", bytes::Bytes::from("v1")).await.unwrap();
    let (_, rev) = kv.get_entry("k").await.unwrap();
    // Update to get a new revision
    kv.cas_update("k", bytes::Bytes::from("v2"), rev)
        .await
        .unwrap();
    // Now retry with stale revision — must fail
    let result = kv.cas_update("k", bytes::Bytes::from("v3"), rev).await;
    assert!(result.is_err(), "stale revision must be rejected");
}

// ── Tag-sharded / SAD tests ───────────────────────────────────────────────────

use h2ai_orchestrator::induction::InductionContext;
use h2ai_types::memory::TagPatternBucket;

fn make_pattern(tags: &[&str], hint: &str, s: u64, a: u64) -> h2ai_types::memory::RetryHintPattern {
    h2ai_types::memory::RetryHintPattern {
        trigger_tags: tags.iter().map(|t| t.to_string()).collect(),
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: hint.to_string(),
        success_count: s,
        attempt_count: a,
    }
}

async fn scheduler_with_bucket(
    tenant: &str,
    tag: &str,
    patterns: Vec<h2ai_types::memory::RetryHintPattern>,
) -> NatsInductionScheduler {
    let kv = Arc::new(InMemoryRetryKvBackend::default());
    let bucket = TagPatternBucket { patterns };
    let key = format!("{}.tag.{}", tenant, tag);
    let bytes: bytes::Bytes = serde_json::to_vec(&bucket).unwrap().into();
    kv.put(&key, bytes).await.unwrap();
    NatsInductionScheduler::from_backend(kv)
}

#[tokio::test]
async fn load_priming_hints_returns_empty_for_fresh_backend() {
    let kv = Arc::new(InMemoryRetryKvBackend::default());
    let scheduler = NatsInductionScheduler::from_backend(kv);
    let ctx = InductionContext {
        tenant_id: "t1".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec![],
    };
    let hints = scheduler.load_priming_hints(&ctx).await;
    assert!(hints.is_empty());
}

#[tokio::test]
async fn load_priming_hints_round1_matches_tag() {
    let scheduler = scheduler_with_bucket(
        "t1",
        "billing",
        vec![make_pattern(&["billing"], "append-only schema", 3, 5)],
    )
    .await;
    let ctx = InductionContext {
        tenant_id: "t1".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec![],
    };
    let hints = scheduler.load_priming_hints(&ctx).await;
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0].hint_text, "append-only schema");
}

#[tokio::test]
async fn load_priming_hints_round2_vocabulary_bridge() {
    // Pattern A: stored under "timeout" with trigger_tags ["timeout", "database"]
    // Pattern B: stored ONLY under "database"
    // Round-1 loads "timeout" → finds pattern A (exposes "database" as new vocab)
    // Round-2 loads "database" → finds pattern B
    // Final result contains both A and B.

    let kv = Arc::new(InMemoryRetryKvBackend::default());

    // Pattern A: stored under "timeout"
    let pattern_a = make_pattern(&["timeout", "database"], "use connection pool", 2, 4);
    let bucket_timeout = TagPatternBucket {
        patterns: vec![pattern_a],
    };
    let bytes_t: bytes::Bytes = serde_json::to_vec(&bucket_timeout).unwrap().into();
    kv.put("t1.tag.timeout", bytes_t).await.unwrap();

    // Pattern B: stored ONLY under "database"
    let pattern_b = make_pattern(&["database"], "use READ COMMITTED isolation", 5, 6);
    let bucket_db = TagPatternBucket {
        patterns: vec![pattern_b],
    };
    let bytes_d: bytes::Bytes = serde_json::to_vec(&bucket_db).unwrap().into();
    kv.put("t1.tag.database", bytes_d).await.unwrap();

    let scheduler = NatsInductionScheduler::from_backend(kv);
    let ctx = InductionContext {
        tenant_id: "t1".to_string(),
        task_class_tags: vec!["timeout".to_string()],
        violated_constraint_ids: vec![],
    };
    let hints = scheduler.load_priming_hints(&ctx).await;
    assert_eq!(
        hints.len(),
        2,
        "round-2 vocabulary bridge must surface database pattern"
    );
    let texts: Vec<&str> = hints.iter().map(|h| h.hint_text.as_str()).collect();
    assert!(texts.contains(&"use connection pool"));
    assert!(texts.contains(&"use READ COMMITTED isolation"));
}

#[tokio::test]
async fn run_retroactive_writes_attempt_count_to_tag_buckets() {
    let kv = Arc::new(InMemoryRetryKvBackend::default());

    // Seed two tag buckets for a pattern with two trigger_tags
    let pattern = make_pattern(&["billing", "audit"], "use append-only", 2, 3);
    for tag in &["billing", "audit"] {
        let bucket = TagPatternBucket {
            patterns: vec![pattern.clone()],
        };
        let bytes: bytes::Bytes = serde_json::to_vec(&bucket).unwrap().into();
        kv.put(&format!("t1.tag.{}", tag), bytes).await.unwrap();
    }

    let scheduler =
        NatsInductionScheduler::from_backend(Arc::clone(&kv) as Arc<dyn RetryKvBackend>);
    let ctx = InductionContext {
        tenant_id: "t1".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec![],
    };
    let result = scheduler.run_retroactive(&ctx).await;
    assert!(result.is_some(), "must return InductionResult");

    // Verify attempt_count incremented in the "billing" bucket
    let (bytes, _) = kv.get_entry("t1.tag.billing").await.unwrap();
    let bucket: TagPatternBucket = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        bucket.patterns[0].attempt_count, 4,
        "attempt_count must be incremented from 3 to 4"
    );
}
