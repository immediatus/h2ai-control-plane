use crate::induction::{InductionContext, InductionResult, InductionScheduler};
use async_nats::jetstream::kv::Store as KvStore;
use async_trait::async_trait;
use h2ai_types::memory::TagPatternBucket;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(crate) const MAX_CAS_RETRIES: u32 = 5;
pub(crate) const H2AI_MEMORY_BUCKET: &str = "H2AI_MEMORY";

/// Full-jitter exponential backoff for NATS KV CAS retry.
/// base=5ms, cap=500ms.
pub fn backoff_ms(attempt: u32) -> u64 {
    let cap: u64 = 500;
    let base: u64 = 5;
    let shift = attempt.min(7) as u64;
    let exp = base.saturating_mul(1u64 << shift);
    let upper = exp.min(cap);
    if upper == 0 {
        return 0;
    }
    rand::random::<u64>() % upper
}

/// KV backend abstraction for `NatsInductionScheduler` — enables in-memory testing.
/// Each entry has a monotonically increasing revision number.
#[async_trait]
pub trait RetryKvBackend: Send + Sync {
    /// Returns `(value, revision)` or `None` when key absent.
    async fn get_entry(&self, key: &str) -> Option<(bytes::Bytes, u64)>;
    /// Unconditional put. Creates key if absent, overwrites if present.
    async fn put(&self, key: &str, value: bytes::Bytes) -> Result<(), String>;
    /// CAS update: succeeds only if `revision` matches the stored revision.
    async fn cas_update(&self, key: &str, value: bytes::Bytes, revision: u64)
        -> Result<(), String>;
}

// ── In-memory backend (for tests) ────────────────────────────────────────────

struct Entry {
    value: bytes::Bytes,
    revision: u64,
}

/// In-memory `RetryKvBackend` for unit tests — no NATS, no I/O.
#[derive(Default)]
pub struct InMemoryRetryKvBackend {
    data: Mutex<HashMap<String, Entry>>,
}

#[async_trait]
impl RetryKvBackend for InMemoryRetryKvBackend {
    async fn get_entry(&self, key: &str) -> Option<(bytes::Bytes, u64)> {
        let guard = self.data.lock().unwrap();
        guard.get(key).map(|e| (e.value.clone(), e.revision))
    }

    async fn put(&self, key: &str, value: bytes::Bytes) -> Result<(), String> {
        let mut guard = self.data.lock().unwrap();
        let rev = guard.get(key).map(|e| e.revision + 1).unwrap_or(1);
        guard.insert(
            key.to_string(),
            Entry {
                value,
                revision: rev,
            },
        );
        Ok(())
    }

    async fn cas_update(
        &self,
        key: &str,
        value: bytes::Bytes,
        revision: u64,
    ) -> Result<(), String> {
        let mut guard = self.data.lock().unwrap();
        match guard.get(key) {
            Some(e) if e.revision == revision => {
                let new_rev = revision + 1;
                guard.insert(
                    key.to_string(),
                    Entry {
                        value,
                        revision: new_rev,
                    },
                );
                Ok(())
            }
            Some(_) => Err("revision mismatch".to_string()),
            None => Err("key not found".to_string()),
        }
    }
}

// ── NATS-backed backend ───────────────────────────────────────────────────────

struct NatsRetryKvBackend {
    kv: KvStore,
}

#[async_trait]
impl RetryKvBackend for NatsRetryKvBackend {
    async fn get_entry(&self, key: &str) -> Option<(bytes::Bytes, u64)> {
        match self.kv.entry(key).await {
            Ok(Some(e)) => Some((e.value, e.revision)),
            _ => None,
        }
    }

    async fn put(&self, key: &str, value: bytes::Bytes) -> Result<(), String> {
        self.kv
            .put(key, value)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    async fn cas_update(
        &self,
        key: &str,
        value: bytes::Bytes,
        revision: u64,
    ) -> Result<(), String> {
        self.kv
            .update(key, value, revision)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

// ── Tag normalization ─────────────────────────────────────────────────────────

fn normalize_tag(tag: &str) -> String {
    crate::induction::normalize_for_shingling(tag).replace(' ', "_")
}

// ── NatsInductionScheduler ────────────────────────────────────────────────────

/// Induction scheduler backed by tag-sharded NATS JetStream KV.
///
/// KV key scheme: `{tenant_id}.tag.{normalized_tag}` → `TagPatternBucket`.
/// A pattern with N trigger_tags appears in N buckets.
///
/// Two paths:
/// - `load_priming_hints`: two-round SAD read-only (no G-counter update)
/// - `run_retroactive`: two-round SAD read + CAS `attempt_count` increment
pub struct NatsInductionScheduler {
    kv: Arc<dyn RetryKvBackend>,
}

impl NatsInductionScheduler {
    /// Construct from a real NATS JetStream KV store.
    pub fn new(kv: KvStore) -> Self {
        Self {
            kv: Arc::new(NatsRetryKvBackend { kv }),
        }
    }

    /// Construct from a custom backend (used in tests).
    pub fn from_backend(backend: Arc<dyn RetryKvBackend>) -> Self {
        Self { kv: backend }
    }

    /// Load all patterns from a tag bucket. Returns empty vec when key absent or corrupt.
    async fn load_bucket(&self, key: &str) -> Vec<h2ai_types::memory::RetryHintPattern> {
        let (bytes, _) = match self.kv.get_entry(key).await {
            Some(e) => e,
            None => return vec![],
        };
        match serde_json::from_slice::<TagPatternBucket>(&bytes) {
            Ok(b) => b.patterns,
            Err(_) => vec![],
        }
    }

    /// Core two-round SAD retrieval.
    async fn sad_retrieve(
        &self,
        ctx: &InductionContext,
    ) -> Vec<h2ai_types::memory::RetryHintPattern> {
        use crate::induction::algorithmic::rank_and_filter;

        // Round 1: load buckets for each task_class_tag
        let mut r1: Vec<h2ai_types::memory::RetryHintPattern> = vec![];
        for tag in &ctx.task_class_tags {
            let key = format!("{}.tag.{}", ctx.tenant_id, normalize_tag(tag));
            r1.extend(self.load_bucket(&key).await);
        }

        // Vocabulary bridge: tags in r1 patterns not in original query
        let query_tags: std::collections::HashSet<&str> =
            ctx.task_class_tags.iter().map(|s| s.as_str()).collect();
        let new_tags: Vec<String> = r1
            .iter()
            .flat_map(|p| p.trigger_tags.iter())
            .filter(|t| !query_tags.contains(t.as_str()))
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Round 2: load buckets for vocabulary-expanded tags
        // seen is mut so cross-bucket duplicates within round-2 are also deduplicated
        let mut seen: std::collections::HashSet<(String, String)> = r1
            .iter()
            .map(|p| (p.exit_reason_kind.clone(), p.hint_text.clone()))
            .collect();
        let mut all = r1;
        for tag in &new_tags {
            let key = format!("{}.tag.{}", ctx.tenant_id, normalize_tag(tag));
            for p in self.load_bucket(&key).await {
                let key_pair = (p.exit_reason_kind.clone(), p.hint_text.clone());
                if seen.insert(key_pair) {
                    all.push(p);
                }
            }
        }

        // Build augmented context that includes vocabulary-expanded tags so that
        // round-2 patterns (whose trigger_tags were not in the original query) pass
        // the rank_and_filter overlap check.
        let mut augmented_tags = ctx.task_class_tags.clone();
        augmented_tags.extend(new_tags);
        let augmented_ctx = InductionContext {
            tenant_id: ctx.tenant_id.clone(),
            task_class_tags: augmented_tags,
            violated_constraint_ids: ctx.violated_constraint_ids.clone(),
        };
        rank_and_filter(&all, &augmented_ctx)
    }

    /// CAS-increment `attempt_count` (and optionally `success_count`) for a specific
    /// pattern in a tag bucket.
    async fn cas_increment_in_bucket(&self, key: &str, hint_text: &str, success: bool) {
        for attempt in 0..MAX_CAS_RETRIES {
            let (bytes, rev) = match self.kv.get_entry(key).await {
                Some(e) => e,
                None => return,
            };
            let mut bucket: TagPatternBucket = match serde_json::from_slice(&bytes) {
                Ok(b) => b,
                Err(_) => return,
            };
            let mut found = false;
            for p in bucket.patterns.iter_mut() {
                if p.hint_text == hint_text {
                    p.attempt_count += 1;
                    if success {
                        p.success_count += 1;
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                return;
            }
            let Ok(new_bytes) = serde_json::to_vec(&bucket) else {
                return;
            };
            match self.kv.cas_update(key, new_bytes.into(), rev).await {
                Ok(_) => return,
                Err(_) => {
                    let delay = backoff_ms(attempt);
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                }
            }
        }
    }
}

#[async_trait]
impl InductionScheduler for NatsInductionScheduler {
    async fn load_priming_hints(
        &self,
        ctx: &InductionContext,
    ) -> Vec<h2ai_types::memory::RetryHintPattern> {
        self.sad_retrieve(ctx).await
    }

    async fn run_retroactive(&self, ctx: &InductionContext) -> Option<InductionResult> {
        let matched = self.sad_retrieve(ctx).await;
        if matched.is_empty() {
            return None;
        }
        // Increment attempt_count for each matched pattern in all its tag buckets
        for p in &matched {
            for tag in &p.trigger_tags {
                let key = format!("{}.tag.{}", ctx.tenant_id, normalize_tag(tag));
                self.cas_increment_in_bucket(&key, &p.hint_text, false)
                    .await;
            }
        }
        let context_tags: Vec<String> = ctx
            .task_class_tags
            .iter()
            .chain(ctx.violated_constraint_ids.iter())
            .cloned()
            .collect();
        Some(InductionResult {
            patterns: matched,
            trigger_tags: context_tags,
        })
    }

    async fn record_success(&self, hint_texts: &[String], ctx: &InductionContext) {
        // Intentionally updates only task_class_tags buckets (not all trigger_tags).
        // run_retroactive writes to every trigger_tag bucket because it fires on failure
        // and the full trigger_tags set is what will be queried next time. record_success
        // fires after a task completes and the caller only has task_class_tags — a
        // narrower write is sufficient since success signals are lower-frequency.
        for hint_text in hint_texts {
            for tag in &ctx.task_class_tags {
                let key = format!("{}.tag.{}", ctx.tenant_id, normalize_tag(tag));
                self.cas_increment_in_bucket(&key, hint_text, true).await;
            }
        }
    }
}

// ── Construction helper ───────────────────────────────────────────────────────

/// Construct a `NatsInductionScheduler` from a raw NATS client and tenant ID.
///
/// Creates or opens the `H2AI_MEMORY` bucket. Returns `None` when NATS is unavailable
/// or bucket creation fails — callers fall back to no-priming behaviour.
pub async fn build_induction_scheduler(
    nats: Option<&async_nats::Client>,
    tenant_id: &h2ai_types::identity::TenantId,
) -> Option<std::sync::Arc<dyn InductionScheduler>> {
    let _ = tenant_id; // tenant scoped in KV keys at runtime
    let nats = nats?;
    let js = async_nats::jetstream::new(nats.clone());
    let kv = js
        .create_key_value(async_nats::jetstream::kv::Config {
            bucket: H2AI_MEMORY_BUCKET.to_string(),
            description: "H2AI RetryHintPattern tag-sharded store".to_string(),
            history: 1,
            storage: async_nats::jetstream::stream::StorageType::File,
            ..Default::default()
        })
        .await
        .ok()?;
    Some(std::sync::Arc::new(NatsInductionScheduler::new(kv)))
}
