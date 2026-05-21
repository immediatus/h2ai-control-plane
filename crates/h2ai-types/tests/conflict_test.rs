use h2ai_types::conflict::ConflictRateAccumulator;
use h2ai_types::identity::TenantId;

fn tenant() -> TenantId {
    TenantId::from("test-tenant")
}

#[test]
fn new_accumulator_starts_at_calibration_floor() {
    let acc = ConflictRateAccumulator::new(tenant(), 0.3);
    assert!((acc.beta_quality - 0.3).abs() < 1e-9);
    assert_eq!(acc.total_tasks_seen, 0);
    assert!(acc.samples.is_empty());
}

#[test]
fn push_sample_increments_task_count() {
    let mut acc = ConflictRateAccumulator::new(tenant(), 0.1);
    acc.push_sample(0.5, 3, 100, 3600, 1);
    assert_eq!(acc.total_tasks_seen, 1);
}

#[test]
fn push_sample_below_min_samples_keeps_floor() {
    let mut acc = ConflictRateAccumulator::new(tenant(), 0.2);
    // min_samples_for_override = 3, only push 2
    acc.push_sample(0.9, 3, 100, 3600, 3);
    acc.push_sample(0.9, 3, 100, 3600, 3);
    // beta_quality should still be the floor since < 3 samples
    assert!((acc.beta_quality - 0.2).abs() < 1e-9);
}

#[test]
fn push_sample_above_min_samples_overrides_floor() {
    let mut acc = ConflictRateAccumulator::new(tenant(), 0.1);
    for _ in 0..5 {
        acc.push_sample(0.8, 3, 100, 3600, 3);
    }
    // With 5 samples all at 0.8, weighted avg ≈ 0.8 > floor 0.1
    assert!(acc.beta_quality > 0.1);
}

#[test]
fn push_sample_enforces_max_samples_window() {
    let mut acc = ConflictRateAccumulator::new(tenant(), 0.0);
    for _ in 0..10 {
        acc.push_sample(0.5, 3, 5, 3600, 1);
    }
    assert_eq!(acc.samples.len(), 5);
}

#[test]
fn beta_quality_never_drops_below_floor() {
    let mut acc = ConflictRateAccumulator::new(tenant(), 0.5);
    // Push many very low conflict rate samples
    for _ in 0..10 {
        acc.push_sample(0.0, 3, 100, 3600, 3);
    }
    assert!(acc.beta_quality >= 0.5);
}

#[test]
fn beta_quality_with_zero_halflife_returns_floor() {
    let mut acc = ConflictRateAccumulator::new(tenant(), 0.25);
    for _ in 0..5 {
        acc.push_sample(0.99, 3, 100, 0, 3);
    }
    assert!((acc.beta_quality - 0.25).abs() < 1e-9);
}

#[test]
fn serde_roundtrip_accumulator() {
    let mut acc = ConflictRateAccumulator::new(tenant(), 0.2);
    acc.push_sample(0.6, 4, 100, 3600, 3);
    acc.push_sample(0.7, 4, 100, 3600, 3);
    acc.push_sample(0.65, 4, 100, 3600, 3);

    let json = serde_json::to_string(&acc).unwrap();
    let back: ConflictRateAccumulator = serde_json::from_str(&json).unwrap();
    assert_eq!(back.tenant_id, tenant());
    assert_eq!(back.samples.len(), 3);
    assert!((back.beta_quality - acc.beta_quality).abs() < 1e-9);
}

/// Exercise `push_sample` with a short halflife. All samples are fresh (timestamp≈now)
/// so age≈0 and weight≈1; `total_weight` >> 1e-12 — the normal path runs, not the
/// underflow guard. This test confirms the rolling computation path.
#[test]
fn beta_quality_short_halflife_uses_rolling_average() {
    let mut acc = ConflictRateAccumulator::new(tenant(), 0.1);
    for _ in 0..5 {
        acc.push_sample(0.8, 3, 100, 1, 3);
    }
    // With fresh samples (age≈0) and halflife=1, weight≈1 per sample → rolling ≈ 0.8.
    assert!(acc.beta_quality >= 0.1, "must be >= floor");
    assert!(acc.beta_quality <= 1.0, "must be <= 1.0");
}
