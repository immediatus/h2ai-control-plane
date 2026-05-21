use std::collections::HashMap;

/// Per-adapter-pair EMA tracker for empirical ρ estimation.
///
/// After each task wave, compute pairwise Pearson score products and call `update`.
/// Once `n_observations >= 30` (CLT threshold) the EMA is in steady state and
/// `rho_mean()` should replace the `rho_mean = 1 − CG_mean` proxy.
#[derive(Default)]
pub struct RhoEmaState {
    ema: HashMap<(String, String), f64>,
    pub n_observations: u32,
}

impl RhoEmaState {
    /// Update EMA with pairwise centered score products from one task wave.
    ///
    /// `pairs`: `(adapter_id_a, adapter_id_b, score_product)` where
    /// `score_product = (score_a − p_mean) × (score_b − p_mean) / variance`,
    /// clamped to [−1, 1].
    /// `alpha = 0.10` gives an effective window of ~10 tasks; steady state after ~30.
    pub fn update(&mut self, pairs: &[(String, String, f64)], alpha: f64) {
        for (a, b, product) in pairs {
            let key = if a <= b {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            };
            let entry = self.ema.entry(key).or_insert(0.45);
            *entry = (1.0 - alpha).mul_add(*entry, alpha * product);
        }
        self.n_observations += 1;
    }

    /// Mean of all per-pair EMA values. Returns 0.45 (conservative prior) if no pairs yet.
    #[must_use]
    pub fn rho_mean(&self) -> f64 {
        if self.ema.is_empty() {
            return 0.45;
        }
        (self.ema.values().sum::<f64>() / self.ema.len() as f64).clamp(0.0, 0.99)
    }
}
