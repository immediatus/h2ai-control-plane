use crate::physics::n_it_optimal;
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
    pub fn n_it_optimal(self) -> usize {
        n_it_optimal(self.rho())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ordering_reflects_consolidation_level() {
        assert!(MemoryTier::Working < MemoryTier::Episodic);
        assert!(MemoryTier::Episodic < MemoryTier::Semantic);
        assert!(MemoryTier::Semantic < MemoryTier::Procedural);
    }

    #[test]
    fn higher_tier_lower_n_it_optimal() {
        // More consolidated memory → fewer ensemble agents needed
        let nw = MemoryTier::Working.n_it_optimal();
        let ne = MemoryTier::Episodic.n_it_optimal();
        let ns = MemoryTier::Semantic.n_it_optimal();
        let np = MemoryTier::Procedural.n_it_optimal();
        assert!(nw >= ne, "Working({nw}) must need ≥ Episodic({ne}) agents");
        assert!(ne >= ns, "Episodic({ne}) must need ≥ Semantic({ns}) agents");
        assert!(
            ns >= np,
            "Semantic({ns}) must need ≥ Procedural({np}) agents"
        );
    }

    #[test]
    fn n_it_optimal_concrete_values() {
        assert_eq!(MemoryTier::Working.n_it_optimal(), 9);
        assert_eq!(MemoryTier::Episodic.n_it_optimal(), 5);
        assert_eq!(MemoryTier::Semantic.n_it_optimal(), 3);
        assert_eq!(MemoryTier::Procedural.n_it_optimal(), 2);
    }

    #[test]
    fn higher_tier_longer_halflife() {
        assert!(
            MemoryTier::Working.decay_halflife_secs() < MemoryTier::Episodic.decay_halflife_secs()
        );
        assert!(
            MemoryTier::Episodic.decay_halflife_secs() < MemoryTier::Semantic.decay_halflife_secs()
        );
        assert!(
            MemoryTier::Semantic.decay_halflife_secs()
                < MemoryTier::Procedural.decay_halflife_secs()
        );
    }

    #[test]
    fn rho_strictly_increases_with_tier() {
        assert!(MemoryTier::Working.rho() < MemoryTier::Episodic.rho());
        assert!(MemoryTier::Episodic.rho() < MemoryTier::Semantic.rho());
        assert!(MemoryTier::Semantic.rho() < MemoryTier::Procedural.rho());
    }
}
