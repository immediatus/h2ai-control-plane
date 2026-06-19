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
}

/// Per-tag KV bucket value for tag-sharded `RetryHintPattern` storage.
///
/// Stored at key `{tenant_id}.tag.{normalized_tag}` in the `H2AI_MEMORY` NATS KV bucket.
/// A pattern with N trigger_tags appears in N buckets (one per tag).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TagPatternBucket {
    pub patterns: Vec<RetryHintPattern>,
}
