use crate::induction::{InductionContext, InductionResult, InductionScheduler};
use async_trait::async_trait;
use h2ai_types::memory::{RetryHintPattern, TenantMemoryStore};

/// Filter and rank `patterns` against `ctx`.
///
/// Keeps patterns whose `trigger_tags` overlap with `ctx.task_class_tags ∪ ctx.violated_constraint_ids`.
/// Returns results sorted descending by Beta(2,8) `success_rate()`.
/// Returns empty `Vec` when nothing matches (does NOT return `Option` — callers decide None).
pub fn rank_and_filter(
    patterns: &[RetryHintPattern],
    ctx: &InductionContext,
) -> Vec<RetryHintPattern> {
    let context_tags: Vec<&str> = ctx
        .task_class_tags
        .iter()
        .chain(ctx.violated_constraint_ids.iter())
        .map(|s| s.as_str())
        .collect();

    let mut matching: Vec<RetryHintPattern> = patterns
        .iter()
        .filter(|p| {
            p.trigger_tags
                .iter()
                .any(|t| context_tags.contains(&t.as_str()))
        })
        .cloned()
        .collect();

    matching.sort_by(|a, b| {
        b.success_rate()
            .partial_cmp(&a.success_rate())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    matching
}

/// An induction scheduler that operates purely on an in-memory `TenantMemoryStore`.
///
/// Used in tests and as a fallback when no NATS connection is available.
pub struct AlgorithmicInductionWorker {
    store: TenantMemoryStore,
}

impl AlgorithmicInductionWorker {
    pub fn new(store: TenantMemoryStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl InductionScheduler for AlgorithmicInductionWorker {
    async fn load_priming_hints(&self, ctx: &InductionContext) -> Vec<RetryHintPattern> {
        rank_and_filter(&self.store.retry_hint_patterns, ctx)
    }

    async fn run_retroactive(&self, ctx: &InductionContext) -> Option<InductionResult> {
        let matching = rank_and_filter(&self.store.retry_hint_patterns, ctx);
        if matching.is_empty() {
            return None;
        }
        let context_tags: Vec<String> = ctx
            .task_class_tags
            .iter()
            .chain(ctx.violated_constraint_ids.iter())
            .cloned()
            .collect();
        Some(InductionResult {
            patterns: matching,
            trigger_tags: context_tags,
        })
    }

    async fn record_success(&self, _hint_texts: &[String], _ctx: &InductionContext) {
        // In-memory store is immutable — no G-counter persistence in tests.
    }
}
