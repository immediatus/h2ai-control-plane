use crate::induction::{InductionContext, InductionResult, InductionScheduler};
use async_trait::async_trait;
use h2ai_types::memory::{
    ArchetypePrior, DecompositionTemplate, RetryHintPattern, TenantMemoryStore, TensionPattern,
    MIN_SAMPLE_COUNT_FOR_AVOID,
};
use h2ai_types::TaskMetaState;
use std::collections::{HashMap, HashSet};

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

    async fn run_distillation_cycle(
        &self,
        metas: &[h2ai_types::TaskMetaState],
        _tenant_id: &str,
    ) -> crate::induction::DistillationResult {
        crate::induction::DistillationResult {
            archetype_priors: distill_archetype_priors(metas),
            tension_patterns: distill_tension_patterns(metas),
            decomposition_templates: distill_decomposition_templates(metas),
        }
    }

    async fn load_semantic_memory(
        &self,
        _tenant_id: &str,
    ) -> Option<crate::induction::DistillationResult> {
        let result = crate::induction::DistillationResult {
            archetype_priors: self.store.archetype_priors.clone(),
            tension_patterns: self.store.tension_patterns.clone(),
            decomposition_templates: self.store.decomposition_templates.clone(),
        };
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }
}

// ── GAP-G1 Phase 2: semantic memory distillation constants ───────────────────

/// Jaccard similarity threshold for clustering tension strings.
pub const TENSION_CLUSTER_THRESHOLD: f64 = 0.6;

/// Confidence cutoff below which an archetype appearance contributes to `avoid_for_tags`.
/// Also used as the aggregate `net_confidence` gate. When `net_confidence` is below this
/// value and `sample_count >= MIN_SAMPLE_COUNT_FOR_AVOID`, `avoid_for_tags` is populated.
pub const ARCHETYPE_LOW_CONFIDENCE_THRESHOLD: f64 = 0.4;

// ── GAP-G1 Phase 2: pure distillation functions ───────────────────────────────

/// Distill archetype performance priors from resolved task states.
///
/// Groups `ArchetypeResult` records by `archetype_name`. For each group:
/// - `net_confidence` is the unweighted mean of all per-task confidences.
/// - `domain_tags` is the sorted union of all `constraint_tags` across group members.
/// - `avoid_for_tags` is populated with tags from low-confidence tasks only when
///   `sample_count >= MIN_SAMPLE_COUNT_FOR_AVOID && net_confidence < 0.4`.
pub fn distill_archetype_priors(metas: &[TaskMetaState]) -> Vec<ArchetypePrior> {
    struct Accum {
        confidences: Vec<f64>,
        domain_tags: HashSet<String>,
        low_confidence_tags: HashSet<String>,
    }

    let mut by_name: HashMap<String, Accum> = HashMap::new();

    for meta in metas {
        for result in &meta.archetype_results {
            let acc = by_name.entry(result.name.clone()).or_insert_with(|| Accum {
                confidences: Vec::new(),
                domain_tags: HashSet::new(),
                low_confidence_tags: HashSet::new(),
            });
            acc.confidences.push(result.confidence);
            for tag in &meta.constraint_tags {
                acc.domain_tags.insert(tag.clone());
                if result.confidence < ARCHETYPE_LOW_CONFIDENCE_THRESHOLD {
                    acc.low_confidence_tags.insert(tag.clone());
                }
            }
        }
    }

    by_name
        .into_iter()
        .map(|(archetype_name, acc)| {
            let sample_count = acc.confidences.len() as u32;
            let net_confidence = acc.confidences.iter().sum::<f64>() / f64::from(sample_count);
            let avoid_for_tags = if sample_count >= MIN_SAMPLE_COUNT_FOR_AVOID
                && net_confidence < ARCHETYPE_LOW_CONFIDENCE_THRESHOLD
            {
                let mut tags: Vec<String> = acc.low_confidence_tags.into_iter().collect();
                tags.sort_unstable();
                tags
            } else {
                vec![]
            };
            let mut domain_tags: Vec<String> = acc.domain_tags.into_iter().collect();
            domain_tags.sort_unstable();
            ArchetypePrior {
                archetype_name,
                domain_tags,
                net_confidence,
                sample_count,
                avoid_for_tags,
            }
        })
        .collect()
}

/// Distill tension cluster patterns from resolved task states.
///
/// Collects all `tensions` strings from `metas`, clusters them by trigram Jaccard
/// similarity (threshold = `TENSION_CLUSTER_THRESHOLD`), and produces one
/// `TensionPattern` per cluster. The canonical text is the longest normalized member.
/// `shingles` are pre-computed for fast retrieval. `resolution_hint` is `None` in Phase 2.
pub fn distill_tension_patterns(metas: &[TaskMetaState]) -> Vec<TensionPattern> {
    use crate::induction::{cluster_by_similarity, normalize_for_shingling, trigram_shingles};

    let all_tensions: Vec<String> = metas
        .iter()
        .flat_map(|m| m.tensions.iter().cloned())
        .collect();

    if all_tensions.is_empty() {
        return vec![];
    }

    let labels = cluster_by_similarity(&all_tensions, TENSION_CLUSTER_THRESHOLD);

    let mut clusters: HashMap<usize, Vec<&str>> = HashMap::new();
    for (tension, label) in all_tensions.iter().zip(labels.iter()) {
        clusters.entry(*label).or_default().push(tension.as_str());
    }

    clusters
        .into_values()
        .map(|members| {
            let frequency = members.len() as u32;
            let canonical = members
                .iter()
                .max_by(|a, b| {
                    let len_a = normalize_for_shingling(a).len();
                    let len_b = normalize_for_shingling(b).len();
                    len_a.cmp(&len_b).then_with(|| b.cmp(a))
                })
                .copied()
                .unwrap_or("")
                .to_string();
            let normalized = normalize_for_shingling(&canonical);
            let shingles = trigram_shingles(&normalized);
            TensionPattern {
                canonical_text: canonical,
                frequency,
                resolution_hint: None,
                shingles,
            }
        })
        .collect()
}

/// Distill decomposition seeding templates from resolved task states.
///
/// Groups metas by `(task_quadrant as Debug string, sorted constraint_tags joined by ",")`.
/// Within each group the template `shared_understanding` is taken from the member
/// with the lowest `retry_count`. `success_count` counts members where `retry_count == 0`.
pub fn distill_decomposition_templates(metas: &[TaskMetaState]) -> Vec<DecompositionTemplate> {
    let mut groups: HashMap<(String, String), Vec<&TaskMetaState>> = HashMap::new();

    for meta in metas {
        let quadrant_key = meta
            .task_quadrant
            .as_ref()
            .map(|q| format!("{q:?}"))
            .unwrap_or_default();
        let mut tags = meta.constraint_tags.clone();
        tags.sort_unstable();
        let tags_key = tags.join(",");
        groups
            .entry((quadrant_key, tags_key))
            .or_default()
            .push(meta);
    }

    groups
        .into_iter()
        .map(|((quadrant_str, _), members)| {
            let best = members
                .iter()
                .min_by_key(|m| m.retry_count)
                .expect("groups are non-empty by construction");
            let success_count = members.iter().filter(|m| m.retry_count == 0).count() as u32;
            let mut constraint_tags = best.constraint_tags.clone();
            constraint_tags.sort_unstable();
            DecompositionTemplate {
                quadrant: quadrant_str,
                constraint_tags,
                shared_understanding: best.shared_understanding.clone(),
                success_count: success_count.max(1),
            }
        })
        .collect()
}
