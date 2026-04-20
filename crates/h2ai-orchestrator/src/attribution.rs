/// Input parameters for computing harness attribution.
#[derive(Debug, Clone)]
pub struct AttributionInput {
    /// Baseline incoherence coefficient c_i for the raw model (0..=1).
    pub baseline_c_i: f64,
    /// Number of explorer agents in the ensemble.
    pub n_agents: u32,
    /// USL contention parameter α (from `CoherencyCoefficients`).
    pub alpha: f64,
    /// USL coherence parameter κ_eff (from `CoherencyCoefficients::kappa_eff()`).
    pub kappa_eff: f64,
    /// Fraction of proposals that survived verification (1.0 = nothing filtered).
    pub verification_filter_ratio: f64,
    /// Mean number of TAO loop turns executed across accepted proposals.
    pub tao_turns_mean: f64,
}

/// USL-grounded decomposition of total output quality into per-component contributions.
///
/// `total_quality ≈ baseline_quality + topology_gain + verification_gain + tao_gain`
/// (clamped to `[0.0, 1.0]`).
#[derive(Debug, Clone)]
pub struct HarnessAttribution {
    /// Single-agent quality from the raw model: `1.0 - c_i`.
    pub baseline_quality: f64,
    /// Quality improvement from N-agent ensemble via USL N_max scaling.
    pub topology_gain: f64,
    /// Quality improvement from the verification phase filtering low-scoring proposals.
    pub verification_gain: f64,
    /// Quality improvement from TAO loop iterations reducing c_i over multiple turns.
    pub tao_gain: f64,
    /// Sum of all components, clamped to `[0.0, 1.0]`.
    pub total_quality: f64,
}

impl HarnessAttribution {
    /// Compute harness attribution from the given input parameters.
    pub fn compute(input: &AttributionInput) -> Self {
        let c_i = input.baseline_c_i.clamp(0.0, 1.0);
        let baseline_quality = 1.0 - c_i;

        // ── Topology gain via USL throughput ────────────────────────────────
        // G_topology = c_i × (1 − 1/X(N))  where X(N) = N / (1 + α(N-1) + κN(N-1))
        // At N=1: X=1 → gain=0 (single agent = no ensemble benefit).
        let alpha = input.alpha.max(0.0);
        let kappa = input.kappa_eff.max(0.0);
        let n = input.n_agents.max(1) as f64;
        let usl_n = (n / (1.0 + alpha * (n - 1.0) + kappa * n * (n - 1.0))).max(1.0);
        let topology_gain = (c_i * (1.0 - 1.0 / usl_n)).max(0.0);

        // ── TAO gain: each additional turn reduces c_i by 40% ───────────────
        let tao_c_i = c_i * 0.6_f64.powf((input.tao_turns_mean - 1.0).max(0.0));
        let tao_gain = ((1.0 - tao_c_i) - baseline_quality).max(0.0);

        // ── Verification gain: filtering keeps better proposals ──────────────
        // filter_ratio = fraction that passed (1.0 → no filtering → 0 gain).
        // Effective c_i of kept set = c_i * filter_ratio.
        let filter_ratio = input.verification_filter_ratio.clamp(0.0, 1.0);
        let verification_gain = ((1.0 - c_i * filter_ratio) - baseline_quality).max(0.0);

        let total_quality =
            (baseline_quality + topology_gain + verification_gain + tao_gain).clamp(0.0, 1.0);

        Self {
            baseline_quality,
            topology_gain,
            verification_gain,
            tao_gain,
            total_quality,
        }
    }
}
