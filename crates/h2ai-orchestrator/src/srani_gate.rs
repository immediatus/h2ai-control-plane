/// Returns the sigmoid injection pressure for a given CFI, EMA midpoint, and temperature.
///
/// pressure = 1 / (1 + exp(-(cfi - mu) / temperature))
///
/// - pressure < 0.20: silent (no event)
/// - 0.20 ≤ pressure < `gate_threshold`: emit `CorrelatedFabricationEvent` (warn only)
/// - pressure ≥ `gate_threshold`: emit event AND inject grounding hint
#[must_use]
pub fn compute_injection_pressure(cfi: f64, mu: f64, temperature: f64) -> f64 {
    1.0 / (1.0 + (-(cfi - mu) / temperature).exp())
}

/// Update the EMA of observed CFI values.
///
/// `ema_new` = alpha * cfi + (1 - alpha) * `old_ema`
#[must_use]
pub fn update_ema(old_ema: f64, cfi: f64, alpha: f64) -> f64 {
    alpha.mul_add(cfi, (1.0 - alpha) * old_ema)
}
