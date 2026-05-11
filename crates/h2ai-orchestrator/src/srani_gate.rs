/// Returns the sigmoid injection pressure for a given CFI, EMA midpoint, and temperature.
///
/// pressure = 1 / (1 + exp(-(cfi - mu) / temperature))
///
/// - pressure < 0.20: silent (no event)
/// - 0.20 ≤ pressure < gate_threshold: emit CorrelatedFabricationEvent (warn only)
/// - pressure ≥ gate_threshold: emit event AND inject grounding hint
pub fn compute_injection_pressure(cfi: f64, mu: f64, temperature: f64) -> f64 {
    1.0 / (1.0 + (-(cfi - mu) / temperature).exp())
}

/// Update the EMA of observed CFI values.
///
/// ema_new = alpha * cfi + (1 - alpha) * old_ema
pub fn update_ema(old_ema: f64, cfi: f64, alpha: f64) -> f64 {
    alpha * cfi + (1.0 - alpha) * old_ema
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn pressure_at_midpoint_is_exactly_half() {
        let p = compute_injection_pressure(0.45, 0.45, 0.15);
        assert!(
            (p - 0.5).abs() < EPSILON,
            "pressure at mu must be 0.5, got {p}"
        );
    }

    #[test]
    fn pressure_well_below_midpoint_is_near_zero() {
        let p = compute_injection_pressure(0.0, 0.45, 0.15);
        assert!(p < 0.10, "pressure at CFI=0 should be < 0.10, got {p}");
    }

    #[test]
    fn pressure_well_above_midpoint_is_near_one() {
        let p = compute_injection_pressure(1.0, 0.45, 0.15);
        assert!(p > 0.90, "pressure at CFI=1 should be > 0.90, got {p}");
    }

    #[test]
    fn pressure_at_mu_plus_0_30_is_above_gate() {
        // mu=0.45, cfi=0.75: well above typical gate_threshold=0.50
        let p = compute_injection_pressure(0.75, 0.45, 0.15);
        assert!(p > 0.80, "pressure at mu+0.30 should be > 0.80, got {p}");
    }

    #[test]
    fn pressure_at_mu_minus_0_30_is_below_warn_floor() {
        let p = compute_injection_pressure(0.15, 0.45, 0.15);
        assert!(
            p < 0.20,
            "pressure at mu-0.30 should be < 0.20 (warn floor), got {p}"
        );
    }

    #[test]
    fn pressure_increases_monotonically_with_cfi() {
        let mu = 0.45;
        let t = 0.15;
        let mut prev = compute_injection_pressure(0.0, mu, t);
        for i in 1..=10 {
            let cfi = i as f64 * 0.1;
            let p = compute_injection_pressure(cfi, mu, t);
            assert!(p > prev, "pressure not monotone at cfi={cfi}");
            prev = p;
        }
    }

    #[test]
    fn higher_temperature_produces_softer_curve() {
        // At CFI=mu+0.3, lower temperature → higher pressure (sharper)
        let p_sharp = compute_injection_pressure(0.75, 0.45, 0.10);
        let p_soft = compute_injection_pressure(0.75, 0.45, 0.30);
        assert!(p_sharp > p_soft, "lower T must produce higher pressure");
    }

    #[test]
    fn ema_update_formula_correct() {
        // 0.20 * 0.70 + 0.80 * 0.45 = 0.14 + 0.36 = 0.50
        let result = update_ema(0.45, 0.70, 0.20);
        assert!(
            (result - 0.50).abs() < EPSILON,
            "ema_update wrong: {result}"
        );
    }

    #[test]
    fn ema_with_alpha_1_equals_new_value() {
        let result = update_ema(0.45, 0.80, 1.0);
        assert!(
            (result - 0.80).abs() < EPSILON,
            "alpha=1 must return new value"
        );
    }

    #[test]
    fn ema_with_alpha_0_returns_old_value() {
        let result = update_ema(0.45, 0.80, 0.0);
        assert!(
            (result - 0.45).abs() < EPSILON,
            "alpha=0 must return old ema"
        );
    }
}
