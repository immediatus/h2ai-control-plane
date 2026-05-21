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
    #[must_use]
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

    #[allow(clippy::cast_precision_loss)]
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
