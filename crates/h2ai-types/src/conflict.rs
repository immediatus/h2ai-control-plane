use crate::identity::TenantId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRateSample {
    pub conflict_rate: f64,
    pub n_adapters: u32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRateAccumulator {
    pub tenant_id: TenantId,
    pub calibration_floor: f64,
    pub samples: Vec<ConflictRateSample>,
    pub beta_quality: f64,
    pub total_tasks_seen: u64,
    pub last_updated: u64,
}

impl ConflictRateAccumulator {
    pub fn new(tenant_id: TenantId, calibration_floor: f64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            tenant_id,
            calibration_floor,
            samples: Vec::new(),
            beta_quality: calibration_floor,
            total_tasks_seen: 0,
            last_updated: now,
        }
    }

    pub fn push_sample(
        &mut self,
        conflict_rate: f64,
        n_adapters: u32,
        max_samples: usize,
        halflife_secs: u64,
        min_samples_for_override: usize,
    ) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.samples.push(ConflictRateSample {
            conflict_rate,
            n_adapters,
            timestamp: now,
        });
        if self.samples.len() > max_samples {
            self.samples.remove(0);
        }
        self.total_tasks_seen += 1;
        self.last_updated = now;
        self.beta_quality = self.compute_beta_quality(halflife_secs, min_samples_for_override, now);
    }

    fn compute_beta_quality(
        &self,
        halflife_secs: u64,
        min_samples_for_override: usize,
        now: u64,
    ) -> f64 {
        if halflife_secs == 0 {
            return self.calibration_floor;
        }
        if self.samples.len() < min_samples_for_override {
            return self.calibration_floor;
        }
        let halflife = halflife_secs as f64;
        let mut weighted_sum = 0.0f64;
        let mut weight_total = 0.0f64;
        for s in &self.samples {
            let age = now.saturating_sub(s.timestamp) as f64;
            let w = (-age / halflife * std::f64::consts::LN_2).exp();
            weighted_sum += w * s.conflict_rate;
            weight_total += w;
        }
        if weight_total < 1e-12 {
            return self.calibration_floor;
        }
        let rolling = (weighted_sum / weight_total).clamp(1e-6, 1.0);
        rolling.max(self.calibration_floor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_acc(floor: f64) -> ConflictRateAccumulator {
        ConflictRateAccumulator::new(TenantId::default_tenant(), floor)
    }

    #[test]
    fn new_accumulator_uses_calibration_floor_as_beta_quality() {
        let acc = make_acc(0.05);
        assert!((acc.beta_quality - 0.05).abs() < 1e-10);
        assert_eq!(acc.total_tasks_seen, 0);
        assert!(acc.samples.is_empty());
    }

    #[test]
    fn push_sample_below_min_threshold_returns_floor() {
        let mut acc = make_acc(0.05);
        // min_samples_for_override=5, push only 3
        for _ in 0..3 {
            acc.push_sample(0.8, 4, 100, 604_800, 5);
        }
        assert!((acc.beta_quality - 0.05).abs() < 1e-10);
        assert_eq!(acc.total_tasks_seen, 3);
    }

    #[test]
    fn push_samples_above_threshold_uses_rolling_estimate() {
        let mut acc = make_acc(0.05);
        for _ in 0..5 {
            acc.push_sample(0.3, 4, 100, 604_800, 5);
        }
        // beta_quality should be ~0.3 (all fresh samples) and > floor (0.05)
        assert!(acc.beta_quality > 0.05);
        assert!((acc.beta_quality - 0.3).abs() < 0.01);
    }

    #[test]
    fn eviction_caps_at_max_samples() {
        let mut acc = make_acc(0.01);
        for _ in 0..15 {
            acc.push_sample(0.2, 4, 10, 604_800, 5);
        }
        assert_eq!(acc.samples.len(), 10);
        assert_eq!(acc.total_tasks_seen, 15);
    }

    #[test]
    fn rolling_never_goes_below_calibration_floor() {
        let mut acc = make_acc(0.4);
        for _ in 0..5 {
            // All samples are 0.1 — below the floor
            acc.push_sample(0.1, 4, 100, 604_800, 5);
        }
        // beta_quality must not drop below calibration_floor
        assert!(acc.beta_quality >= 0.4);
    }

    #[test]
    fn old_samples_decay_toward_floor() {
        let floor = 0.05;
        let mut acc = make_acc(floor);
        // Inject 5 samples with very old timestamps (simulated via direct manipulation)
        let ancient = 0u64; // epoch
        for _ in 0..5 {
            acc.samples.push(ConflictRateSample {
                conflict_rate: 0.9,
                n_adapters: 4,
                timestamp: ancient,
            });
            acc.total_tasks_seen += 1;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Manually trigger recompute
        let result = acc.compute_beta_quality(604_800, 5, now);
        // Ancient samples have near-zero weight → result ≈ floor
        assert!((result - floor).abs() < 0.01);
    }
}
