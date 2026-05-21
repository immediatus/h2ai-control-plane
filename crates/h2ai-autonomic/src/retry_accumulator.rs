#![allow(clippy::cast_precision_loss)]
use h2ai_types::events::ConstraintViolation;
use std::collections::HashMap;

/// Per-task leaky accumulator for per-criterion violation rates across MAPE-K retries.
///
/// Must be owned by the engine's per-task execution context. Never store in global `NatsKv`
/// or any struct shared across task IDs (spec §A.3: state-leakage prevention).
///
/// Update rule: accumulated = λ·old + (`1−λ)·new_rate`.
/// At λ=0.7: half-life ≈ 2 retries; old violations are negligible after 5 retries.
#[derive(Debug, Default)]
pub struct RetryAccumulator {
    rates: HashMap<String, f64>,
}

impl RetryAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update accumulated violation rates with this wave's violations.
    ///
    /// `violations`: flat slice of `ConstraintViolation` from all failed proposals this wave.
    /// `n_f`: number of failed proposals (denominator for rate computation).
    /// `lambda`: decay factor (use `OspConfig::accumulation_decay`, default 0.7).
    pub fn update(&mut self, violations: &[ConstraintViolation], n_f: usize, lambda: f64) {
        if n_f == 0 {
            return;
        }
        let denominator = n_f as f64;
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for v in violations {
            *counts.entry(v.constraint_id.as_str()).or_insert(0) += 1;
        }
        for (&cid, &count) in &counts {
            let new_rate = count as f64 / denominator;
            let old = self.rates.get(cid).copied().unwrap_or(0.0);
            self.rates.insert(
                cid.to_string(),
                lambda.mul_add(old, (1.0 - lambda) * new_rate),
            );
        }
    }

    /// Read-only access to accumulated per-criterion concordance rates.
    #[must_use]
    pub const fn rates(&self) -> &HashMap<String, f64> {
        &self.rates
    }

    /// Reset all rates. Call when synthesis succeeds.
    /// Do NOT call on `ZeroSurvival` — preserve accumulation across retries.
    pub fn reset(&mut self) {
        self.rates.clear();
    }
}
