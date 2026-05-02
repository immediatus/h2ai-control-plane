use h2ai_types::physics::{CoherencyCoefficients, CoordinationThreshold};

#[test]
fn calibration_event_has_valid_n_max() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71]).unwrap();
    let n_max = cc.n_max();
    // New formula β_eff = β₀/max(CG,0.05): CG_mean≈0.71, β_eff=0.021/0.71≈0.030 → N_max≈5
    assert!(
        n_max > 1.0 && n_max < 20.0,
        "n_max={n_max} out of expected range"
    );
}

#[test]
fn calibration_theta_coord_bounded() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    assert!(theta.value() <= 0.3);
    assert!(theta.value() >= 0.0);
}
