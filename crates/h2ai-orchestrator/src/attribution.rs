use h2ai_types::physics::condorcet_quality;

/// Input parameters for computing harness attribution.
#[derive(Debug, Clone)]
pub struct AttributionInput {
    /// Mean per-adapter estimated accuracy (from EnsembleCalibration.p_mean,
    /// or proxy `0.5 + CG_mean / 2` when EnsembleCalibration unavailable).
    pub p_mean: f64,
    /// Mean pairwise error correlation (from EnsembleCalibration.rho_mean,
    /// or proxy `1 - CG_mean`).
    pub rho_mean: f64,
    /// Number of explorer agents in the ensemble.
    pub n_agents: u32,
    /// Fraction of proposals that survived verification (1.0 = nothing filtered).
    pub verification_filter_ratio: f64,
    /// Mean number of TAO loop turns executed across accepted proposals.
    pub tao_turns_mean: f64,
    /// Multiplicative factor applied per additional TAO turn (from H2AIConfig::tao_per_turn_factor).
    pub tao_per_turn_factor: f64,
}

/// Condorcet-grounded decomposition of total output quality into per-component contributions.
///
/// `total_quality = 1 − (1 − Q(N, p, ρ)) × verification_filter_ratio × tao_multiplier`
/// clamped to `[p_mean, 1.0]`.
#[derive(Debug, Clone)]
pub struct HarnessAttribution {
    /// Single-agent expected quality: p_mean.
    pub baseline_quality: f64,
    /// Quality improvement from N-agent ensemble via Condorcet Jury Theorem.
    /// `topology_gain = Q(N, p_mean, rho_mean) − p_mean`.
    pub topology_gain: f64,
    /// Upper-bound estimate of the quality contribution from the verification phase.
    /// Computed as `Q_ensemble × (1 − verification_filter_ratio)`. Informational only —
    /// not a strict partition of `total_quality`.
    pub verification_gain: f64,
    /// Upper-bound estimate of the quality contribution from TAO loop iterations.
    /// Computed as `Q_ensemble × (1 − tao_multiplier)`. Informational only —
    /// not a strict partition of `total_quality`.
    pub tao_gain: f64,
    /// Total quality, clamped to `[p_mean, 1.0]`.
    pub total_quality: f64,
}

impl HarnessAttribution {
    pub fn compute(input: &AttributionInput) -> Self {
        let p = input.p_mean.clamp(0.0, 1.0);
        let rho = input.rho_mean.clamp(0.0, 1.0);
        let n = input.n_agents.max(1) as usize;

        let baseline_quality = p;
        let q_ensemble = condorcet_quality(n, p, rho);
        let topology_gain = (q_ensemble - p).max(0.0);

        let tpf = input.tao_per_turn_factor.clamp(0.0, 1.0);
        let turns = input.tao_turns_mean.max(1.0);
        let tao_multiplier = tpf.powf(turns - 1.0);
        let tao_gain = (q_ensemble * (1.0 - tao_multiplier)).max(0.0);

        let fr = input.verification_filter_ratio.clamp(0.0, 1.0);
        let verification_gain = (q_ensemble * (1.0 - fr)).max(0.0);

        // Compound all improvements on the residual error
        let error_remaining = (1.0 - q_ensemble) * fr * tao_multiplier;
        let total_quality = (1.0 - error_remaining).clamp(baseline_quality, 1.0);

        Self {
            baseline_quality,
            topology_gain,
            verification_gain,
            tao_gain,
            total_quality,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribution_n1_topology_gain_is_zero() {
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.3,
            n_agents: 1,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.topology_gain.abs() < 1e-10,
            "N=1 topology_gain should be 0, got {}",
            attr.topology_gain
        );
    }

    #[test]
    fn attribution_n3_topology_gain_positive_for_good_p() {
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.2,
            n_agents: 3,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.topology_gain > 0.0,
            "N=3 with p=0.7, rho=0.2 should have positive topology_gain, got {}",
            attr.topology_gain
        );
    }

    #[test]
    fn attribution_total_quality_bounded() {
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 0.3,
            n_agents: 5,
            verification_filter_ratio: 0.8,
            tao_turns_mean: 2.0,
            tao_per_turn_factor: 0.6,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.total_quality >= 0.0 && attr.total_quality <= 1.0,
            "total_quality out of bounds: {}",
            attr.total_quality
        );
    }

    #[test]
    fn attribution_no_topology_gain_at_full_correlation() {
        let input = AttributionInput {
            p_mean: 0.7,
            rho_mean: 1.0,
            n_agents: 5,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.topology_gain.abs() < 1e-10,
            "rho=1 should give zero topology_gain, got {}",
            attr.topology_gain
        );
    }

    #[test]
    fn attribution_total_quality_at_least_baseline() {
        // total_quality must always be >= p_mean (the single-agent baseline)
        let input = AttributionInput {
            p_mean: 0.6,
            rho_mean: 0.4,
            n_agents: 3,
            verification_filter_ratio: 0.7,
            tao_turns_mean: 2.0,
            tao_per_turn_factor: 0.6,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.total_quality >= attr.baseline_quality,
            "total_quality {} < baseline_quality {}",
            attr.total_quality,
            attr.baseline_quality
        );
    }

    #[test]
    fn attribution_below_majority_accuracy_no_topology_gain() {
        // p < 0.5: ensemble is worse than random; topology_gain should be 0 (clamped)
        let input = AttributionInput {
            p_mean: 0.4,
            rho_mean: 0.0,
            n_agents: 5,
            verification_filter_ratio: 1.0,
            tao_turns_mean: 1.0,
            tao_per_turn_factor: 0.6,
        };
        let attr = HarnessAttribution::compute(&input);
        assert!(
            attr.topology_gain == 0.0,
            "p=0.4 < 0.5 should give topology_gain=0 (clamped), got {}",
            attr.topology_gain
        );
    }
}
