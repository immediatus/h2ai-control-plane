use crate::sizing::n_it_optimal;
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
