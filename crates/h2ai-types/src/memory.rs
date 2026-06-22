use crate::sizing::n_it_optimal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Consolidation tier of a context chunk in the Atkinson–Shiffrin memory hierarchy.
///
/// Each tier carries two calibrated constants:
/// - `rho` (ρ): per-iteration information-capture probability used by `n_it_optimal`.
///   Higher tier → higher ρ → fewer ensemble agents needed (stable knowledge is
///   reliably captured by a single pass; uncertain working memory needs many).
/// - `decay_halflife_secs`: Ebbinghaus halflife for temporal weighting.
///   Working memory expires in hours; procedural knowledge persists for weeks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryTier {
    /// In-flight observations; valid for minutes to hours. ρ=0.08, halflife=1h.
    Working = 0,
    /// Recent events and session history. ρ=0.20, halflife=24h.
    Episodic = 1,
    /// Consolidated domain concepts and facts. ρ=0.35, halflife=7d.
    Semantic = 2,
    /// Codified rules and stable constraints. ρ=0.50, halflife=30d.
    Procedural = 3,
}

impl MemoryTier {
    /// Per-iteration information-capture probability for `n_it_optimal`.
    ///
    /// Reflects how reliably a single ensemble agent can use knowledge from this
    /// tier: high ρ for stable procedural rules (reliable), low ρ for ephemeral
    /// working memory (uncertain).
    #[must_use]
    pub const fn rho(self) -> f64 {
        match self {
            Self::Working => 0.08,
            Self::Episodic => 0.20,
            Self::Semantic => 0.35,
            Self::Procedural => 0.50,
        }
    }

    /// Exponential decay time constant τ in seconds: at age t=τ, weight = e^−1 ≈ 0.37.
    ///
    /// Uses `exp(-t/τ)` (same convention as `CoherencyCoefficients::beta_eff_temporal`).
    #[must_use]
    pub const fn decay_halflife_secs(self) -> u64 {
        match self {
            Self::Working => 3_600,        // 1 hour
            Self::Episodic => 86_400,      // 24 hours
            Self::Semantic => 604_800,     // 7 days
            Self::Procedural => 2_592_000, // 30 days
        }
    }

    /// Minimum ensemble size for reliable use of chunks at this tier.
    ///
    /// Derived from `n_it_optimal(self.rho())`. Ranges from 9 (Working) to 2
    /// (Procedural) — procedural rules require only two agents to reach
    /// consensus; ephemeral working memory needs the full ensemble.
    #[must_use]
    pub fn n_it_optimal(self) -> usize {
        n_it_optimal(self.rho())
    }
}

// ── GAP-G1: Induction Worker types ───────────────────────────────────────────

/// A single induction-derived retry hint with G-Counter success/attempt tracking.
///
/// G-Counter semantics: merge by addition (commutative, associative, idempotent on
/// replica convergence). Success rate is derived on read using a Beta(2,8) prior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryHintPattern {
    /// Domain tags that characterize the failure context this hint was distilled from.
    pub trigger_tags: Vec<String>,
    /// String representation of the exit reason kind (e.g. "ZeroSurvival").
    pub exit_reason_kind: String,
    /// Human-readable retry hint text to inject into the retry context.
    pub hint_text: String,
    /// G-Counter numerator: number of task attempts where this hint led to success.
    pub success_count: u64,
    /// G-Counter denominator: total task attempts where this hint was applied.
    pub attempt_count: u64,
}

impl RetryHintPattern {
    /// Bayesian success rate with Beta(2,8) prior (conservative ~20% base rate).
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        (self.success_count as f64 + 2.0) / (self.attempt_count as f64 + 10.0)
    }

    /// Merge G-Counter counts from another pattern (addition — commutative).
    pub fn merge_counts(&mut self, other: &Self) {
        self.success_count += other.success_count;
        self.attempt_count += other.attempt_count;
    }
}

/// Persisted memory store for a single tenant, holding all distilled retry hint patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantMemoryStore {
    pub tenant_id: String,
    pub generated_at: DateTime<Utc>,
    pub task_count_seen: u64,
    pub retry_hint_patterns: Vec<RetryHintPattern>,
    /// Distilled archetype performance priors. Empty on cold-start and old stored data.
    #[serde(default)]
    pub archetype_priors: Vec<ArchetypePrior>,
    /// Distilled tension cluster patterns. Empty on cold-start and old stored data.
    #[serde(default)]
    pub tension_patterns: Vec<TensionPattern>,
    /// Distilled decomposition seeding templates. Empty on cold-start and old stored data.
    #[serde(default)]
    pub decomposition_templates: Vec<DecompositionTemplate>,
}

/// Per-tag KV bucket value for tag-sharded `RetryHintPattern` storage.
///
/// Stored at key `{tenant_id}.tag.{normalized_tag}` in the `H2AI_MEMORY` NATS KV bucket.
/// A pattern with N trigger_tags appears in N buckets (one per tag).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TagPatternBucket {
    pub patterns: Vec<RetryHintPattern>,
}

// ── GAP-G1 Phase 2: Semantic memory types ────────────────────────────────────

/// Minimum task sample count before `avoid_for_tags` is populated for an archetype.
pub const MIN_SAMPLE_COUNT_FOR_AVOID: u32 = 3;

/// Induction-derived prior over archetype performance within a domain.
///
/// One record per unique `archetype_name` across all observed tasks.
/// `net_confidence` is the unweighted mean of per-task confidences reported
/// by that archetype. `avoid_for_tags` is populated only when
/// `sample_count >= MIN_SAMPLE_COUNT_FOR_AVOID && net_confidence < 0.4`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchetypePrior {
    pub archetype_name: String,
    /// Union of `constraint_tags` from all tasks where this archetype was observed.
    pub domain_tags: Vec<String>,
    /// Unweighted mean confidence across `sample_count` tasks.
    pub net_confidence: f64,
    /// Number of task appearances contributing to this estimate.
    pub sample_count: u32,
    /// Constraint tags for which this archetype consistently underperformed.
    /// Empty unless `sample_count >= MIN_SAMPLE_COUNT_FOR_AVOID && net_confidence < 0.4`.
    pub avoid_for_tags: Vec<String>,
}

/// Induction-derived pattern from recurring tension strings across tasks.
///
/// Tensions are clustered by trigram Jaccard similarity. One record per cluster;
/// `canonical_text` is the longest normalized member. `shingles` are pre-computed
/// at distillation time for fast similarity retrieval without re-normalization.
/// `resolution_hint` is populated in Phase 3 when outcome data becomes available.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensionPattern {
    /// Representative text chosen as the longest normalized member of the cluster.
    pub canonical_text: String,
    /// Number of tension strings across all tasks that mapped to this cluster.
    pub frequency: u32,
    /// Effective resolution for this tension class (Phase 3 only; `None` in Phase 2).
    pub resolution_hint: Option<String>,
    /// Pre-computed trigram shingles of `canonical_text` after normalization.
    pub shingles: Vec<[u8; 3]>,
}

/// Induction-derived template for task decomposition seeding.
///
/// Groups resolved tasks by `(quadrant, sorted constraint_tags)`. The `shared_understanding`
/// from the member with the lowest `retry_count` serves as the seed for decomposition step 1.
/// `success_count` counts members where `retry_count == 0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecompositionTemplate {
    /// String representation of `TaskQuadrant` (e.g. `"Coverage"`, `"Precision"`).
    /// Uses `format!("{:?}", quadrant)`. `""` when quadrant is `None`.
    pub quadrant: String,
    /// Sorted constraint tags defining the template's domain scope.
    pub constraint_tags: Vec<String>,
    /// `shared_understanding` from the lowest-`retry_count` task in this group.
    pub shared_understanding: String,
    /// Count of tasks in this group with `retry_count == 0`.
    /// Floored at 1 by construction — a group with no zero-retry members still reports 1.
    pub success_count: u32,
}
