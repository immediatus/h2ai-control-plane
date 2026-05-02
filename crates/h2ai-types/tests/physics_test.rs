use h2ai_types::physics::{
    n_it_optimal, CoherencyCoefficients, CoordinationThreshold, EigenCalibration,
    EnsembleCalibration, JeffectiveGap, MergeStrategy, MultiplicationCondition,
    MultiplicationConditionFailure, RoleErrorCost, TauValue,
};

#[test]
fn coherency_coefficients_beta_eff_computation() {
    let cc = CoherencyCoefficients::new(0.12, 0.020, vec![0.6, 0.7, 0.65]).unwrap();
    // CG_mean = (0.6+0.7+0.65)/3 = 0.65
    // β_eff = β₀ × (1 − CG_mean) = 0.020 × 0.35 = 0.007
    let beta_eff = cc.beta_eff();
    let expected = 0.020 * (1.0 - 0.65);
    assert!(
        (beta_eff - expected).abs() < 1e-10,
        "β_eff = β₀×(1−CG_mean) = {expected:.6}, got {beta_eff:.6}"
    );
}

#[test]
fn coherency_coefficients_computes_n_max_usl() {
    let cc = CoherencyCoefficients::new(0.12, 0.020, vec![0.65]).unwrap();
    // β_eff = 0.020 × (1 − 0.65) = 0.020 × 0.35 = 0.007
    // N_max = round(√(0.88/0.007)) = round(√125.7) = round(11.21) = 11
    let n_max = cc.n_max();
    let expected = ((1.0_f64 - 0.12) / (0.020 * (1.0 - 0.65))).sqrt().round();
    assert!(
        (n_max - expected).abs() < 1e-9,
        "N_max = round(√((1−α)/β_eff)) = {expected:.3}, got {n_max}"
    );
}

#[test]
fn coordination_threshold_formula_verified() {
    // θ_coord = clamp(CG_mean − CG_std_dev, 0, max)
    // For [0.6, 0.7, 0.65]: mean=0.65, std≈0.0408, spread≈0.6092 → clamped to 0.3
    let cc = CoherencyCoefficients::new(0.12, 0.020, vec![0.6, 0.7, 0.65]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    // spread = 0.65 − 0.0408 ≈ 0.6092 → clamp to 0.3
    assert!(
        (theta.value() - 0.3).abs() < 1e-9,
        "θ_coord must be clamped to max=0.3 when spread > max; got {}",
        theta.value()
    );

    // For low CG samples: spread < max → θ_coord = spread
    let cc_low = CoherencyCoefficients::new(0.12, 0.020, vec![0.1, 0.15, 0.12]).unwrap();
    let theta_low = CoordinationThreshold::from_calibration(&cc_low, 0.3);
    let mean_low = cc_low.cg_mean();
    let std_low = cc_low.cg_std_dev();
    let expected_spread = (mean_low - std_low).clamp(0.0, 0.3);
    assert!(
        (theta_low.value() - expected_spread).abs() < 1e-9,
        "θ_coord = spread = {expected_spread:.6} when spread < max; got {}",
        theta_low.value()
    );
}

#[test]
fn coherency_coefficients_serde_round_trip() {
    let cc = CoherencyCoefficients::new(0.10, 0.015, vec![0.55, 0.70, 0.62]).unwrap();
    let json = serde_json::to_string(&cc).unwrap();
    let back: CoherencyCoefficients = serde_json::from_str(&json).unwrap();
    assert_eq!(cc.alpha, back.alpha);
    assert_eq!(cc.beta_base, back.beta_base);
    assert_eq!(cc.cg_samples.len(), back.cg_samples.len());
}

#[test]
fn coherency_coefficients_kappa_base_alias_loads_as_beta_base() {
    let json = r#"{"alpha":0.12,"kappa_base":0.021,"cg_samples":[0.68,0.74,0.71]}"#;
    let cc: CoherencyCoefficients = serde_json::from_str(json).unwrap();
    assert!((cc.alpha - 0.12).abs() < 1e-10);
    assert!((cc.beta_base - 0.021).abs() < 1e-10);
}

#[test]
fn coherency_coefficients_invalid_when_alpha_out_of_range() {
    assert!(CoherencyCoefficients::new(1.5, 0.02, vec![0.6]).is_err());
    assert!(CoherencyCoefficients::new(-0.1, 0.02, vec![0.6]).is_err());
}

#[test]
fn coherency_coefficients_invalid_when_no_cg_samples() {
    assert!(CoherencyCoefficients::new(0.12, 0.02, vec![]).is_err());
}

#[test]
fn coherency_coefficients_invalid_when_beta_base_negative() {
    assert!(CoherencyCoefficients::new(0.12, -0.01, vec![0.6]).is_err());
}

#[test]
fn coordination_threshold_serde_round_trip() {
    let cc = CoherencyCoefficients::new(0.12, 0.020, vec![0.6, 0.7, 0.65]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let json = serde_json::to_string(&theta).unwrap();
    let back: CoordinationThreshold = serde_json::from_str(&json).unwrap();
    assert_eq!(theta.value(), back.value());
}

#[test]
fn role_error_cost_valid_range() {
    assert!(RoleErrorCost::new(0.0).is_ok());
    assert!(RoleErrorCost::new(1.0).is_ok());
    assert!(RoleErrorCost::new(0.85).is_ok());
}

#[test]
fn role_error_cost_invalid_out_of_range() {
    assert!(RoleErrorCost::new(-0.1).is_err());
    assert!(RoleErrorCost::new(1.1).is_err());
}

#[test]
fn role_error_cost_serde_round_trip() {
    let c = RoleErrorCost::new(0.92).unwrap();
    let json = serde_json::to_string(&c).unwrap();
    let back: RoleErrorCost = serde_json::from_str(&json).unwrap();
    assert_eq!(c.value(), back.value());
}

#[test]
fn merge_strategy_is_score_ordered_when_max_ci_at_or_below_threshold() {
    let costs = vec![
        RoleErrorCost::new(0.3).unwrap(),
        RoleErrorCost::new(0.85).unwrap(),
    ];
    assert_eq!(
        MergeStrategy::from_role_costs(&costs, 0.85, 0.95, 0),
        MergeStrategy::ScoreOrdered
    );
}

#[test]
fn merge_strategy_is_consensus_median_when_max_ci_above_threshold() {
    let costs = vec![
        RoleErrorCost::new(0.3).unwrap(),
        RoleErrorCost::new(0.91).unwrap(),
    ];
    assert_eq!(
        MergeStrategy::from_role_costs(&costs, 0.85, 0.95, 0),
        MergeStrategy::ConsensusMedian
    );
}

#[test]
fn merge_strategy_serde_round_trip() {
    for strategy in [MergeStrategy::ScoreOrdered, MergeStrategy::ConsensusMedian] {
        let json = serde_json::to_string(&strategy).unwrap();
        let back: MergeStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(strategy, back);
    }
}

#[test]
fn j_effective_gap_valid_range() {
    assert!(JeffectiveGap::new(0.0).is_ok());
    assert!(JeffectiveGap::new(1.0).is_ok());
    assert!(JeffectiveGap::new(0.42).is_ok());
}

#[test]
fn j_effective_gap_invalid_out_of_range() {
    assert!(JeffectiveGap::new(-0.1).is_err());
    assert!(JeffectiveGap::new(1.1).is_err());
}

#[test]
fn j_effective_gap_is_below_threshold_when_low() {
    let j = JeffectiveGap::new(0.1).unwrap();
    assert!(j.is_below_threshold(0.3));
}

#[test]
fn j_effective_gap_is_not_below_threshold_when_sufficient() {
    let j = JeffectiveGap::new(0.6).unwrap();
    assert!(!j.is_below_threshold(0.3));
}

#[test]
fn multiplication_condition_passes_when_all_hold() {
    let result = MultiplicationCondition::evaluate(0.7, 0.85, 0.65, 0.3, 0.5, 0.9);
    assert!(result.is_ok());
}

#[test]
fn multiplication_condition_fails_on_low_competence() {
    let result = MultiplicationCondition::evaluate(0.4, 0.85, 0.65, 0.3, 0.5, 0.9);
    assert!(matches!(
        result,
        Err(MultiplicationConditionFailure::InsufficientCompetence { .. })
    ));
}

#[test]
fn multiplication_condition_fails_on_high_correlation() {
    let result = MultiplicationCondition::evaluate(0.7, 0.95, 0.65, 0.3, 0.5, 0.9);
    assert!(matches!(
        result,
        Err(MultiplicationConditionFailure::InsufficientDecorrelation { .. })
    ));
}

#[test]
fn multiplication_condition_fails_when_cg_below_theta() {
    let result = MultiplicationCondition::evaluate(0.7, 0.85, 0.2, 0.3, 0.5, 0.9);
    assert!(matches!(
        result,
        Err(MultiplicationConditionFailure::CommonGroundBelowFloor { .. })
    ));
}

#[test]
fn tau_value_valid_range() {
    assert!(TauValue::new(0.5).is_ok());
    assert_eq!(TauValue::new(0.5).unwrap().value(), 0.5);
}

#[test]
fn tau_value_boundary_one() {
    assert!(TauValue::new(1.0).is_ok());
    assert_eq!(TauValue::new(1.0).unwrap().value(), 1.0);
}

#[test]
fn tau_value_zero_is_valid() {
    assert!(TauValue::new(0.0).is_ok(), "tau=0.0 is a valid lower bound");
}

#[test]
fn tau_value_above_one_invalid() {
    assert!(TauValue::new(1.1).is_err());
}

#[test]
fn merge_strategy_outlier_resistant_serializes_and_deserializes() {
    let s = MergeStrategy::OutlierResistant { f: 1 };
    let json = serde_json::to_string(&s).unwrap();
    let back: MergeStrategy = serde_json::from_str(&json).unwrap();
    assert_eq!(back, s);
}

#[test]
fn merge_strategy_multi_outlier_resistant_serializes_and_deserializes() {
    let s = MergeStrategy::MultiOutlierResistant { f: 2, m: 3 };
    let json = serde_json::to_string(&s).unwrap();
    let back: MergeStrategy = serde_json::from_str(&json).unwrap();
    assert_eq!(back, s);
}

#[test]
fn from_role_costs_selects_outlier_resistant_above_krum_threshold() {
    let costs = vec![RoleErrorCost::new(0.97).unwrap()];
    let strategy = MergeStrategy::from_role_costs(&costs, 0.85, 0.95, 1);
    assert_eq!(strategy, MergeStrategy::OutlierResistant { f: 1 });
}

#[test]
fn from_role_costs_selects_condorcet_in_middle_tier() {
    let costs = vec![RoleErrorCost::new(0.90).unwrap()];
    let strategy = MergeStrategy::from_role_costs(&costs, 0.85, 0.95, 1);
    assert_eq!(strategy, MergeStrategy::ConsensusMedian);
}

#[test]
fn from_role_costs_outlier_resistant_disabled_when_f_is_zero() {
    // krum_f=0 disables OutlierResistant even above krum_threshold
    let costs = vec![RoleErrorCost::new(0.99).unwrap()];
    let strategy = MergeStrategy::from_role_costs(&costs, 0.85, 0.95, 0);
    assert_eq!(strategy, MergeStrategy::ConsensusMedian);
}

#[test]
fn from_role_costs_empty_slice_selects_score_ordered() {
    // Empty costs → max_ci = NEG_INFINITY → below any threshold → ScoreOrdered.
    let strategy = MergeStrategy::from_role_costs(&[], 0.85, 0.95, 1);
    assert_eq!(strategy, MergeStrategy::ScoreOrdered);
}

#[test]
fn multiplication_condition_cg_mean_exactly_at_theta_passes() {
    // cg_mean < theta_coord is strict less-than → equality must pass.
    let result = MultiplicationCondition::evaluate(0.7, 0.85, 0.3, 0.3, 0.5, 0.9);
    assert!(
        result.is_ok(),
        "cg_mean == theta must pass (strict < semantics)"
    );
}

#[test]
fn multiplication_condition_competence_failure_takes_priority_over_others() {
    // All three conditions fail, but InsufficientCompetence is checked first.
    let result = MultiplicationCondition::evaluate(
        0.1,  // competence below min 0.5
        0.99, // correlation above max 0.9
        0.0,  // cg below theta 0.3
        0.3, 0.5, 0.9,
    );
    assert!(
        matches!(
            result,
            Err(MultiplicationConditionFailure::InsufficientCompetence { .. })
        ),
        "competence failure must be reported first"
    );
}

#[test]
fn coherency_coefficients_usl_n_max_ai_agents() {
    // AI-agent tier: α=0.15, β₀=0.039, CG=0.4
    // β_eff = 0.039×(1−0.4) = 0.039×0.6 = 0.02340 → N_max = round(√(0.85/0.02340)) = round(6.02) = 6
    let cc = CoherencyCoefficients::new(0.15, 0.039, vec![0.4]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 6.0).abs() < 1.0,
        "AI-agent tier N_max must be ≈6 with proportional β_eff, got {n_max}"
    );
}

#[test]
fn coherency_coefficients_beta_eff_proportional_formula() {
    // β_eff = β₀ × (1 − CG_mean) = 0.039 × 0.6 = 0.02340
    let cc = CoherencyCoefficients::new(0.15, 0.039, vec![0.4]).unwrap();
    let beta_eff = cc.beta_eff();
    assert!(
        (beta_eff - 0.02340).abs() < 1e-5,
        "β_eff = β₀×(1−CG) = 0.02340, got {beta_eff}"
    );
}

#[test]
fn coherency_coefficients_beta_eff_bounded_at_low_cg() {
    // At CG→0, proportional form: β_eff = β₀×1.0 = β₀. Must not diverge.
    let cc = CoherencyCoefficients::new(0.15, 0.039, vec![0.001]).unwrap();
    let beta_eff = cc.beta_eff();
    assert!(
        beta_eff < 0.04,
        "β_eff must be bounded (≤ β₀) even at CG≈0, got {beta_eff}"
    );
    assert!(beta_eff > 0.0, "β_eff must be positive, got {beta_eff}");
}

#[test]
fn coherency_coefficients_human_team_tier() {
    // Human team tier: α=0.10, β₀=0.0225, CG=0.6
    // β_eff = 0.0225×(1−0.6) = 0.0225×0.4 = 0.009 → N_max = round(√(0.9/0.009)) = round(10.0) = 10
    let cc = CoherencyCoefficients::new(0.10, 0.0225, vec![0.6]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 10.0).abs() < 1.5,
        "Human team N_max must be ≈10 with proportional β_eff, got {n_max}"
    );
}

#[test]
fn coherency_coefficients_cpu_core_tier() {
    // CPU tier: α=0.02, β₀=0.0003, CG=1.0.
    // With proportional formula, β_eff = β₀×(1−1.0) ≈ 0 → retrograde disappears.
    // CPU cores are coherency-free at full alignment; only α limits throughput.
    // N_max with near-zero β_eff is very large (>> 57); assert ≥ 50.
    let cc = CoherencyCoefficients::new(0.02, 0.0003, vec![1.0]).unwrap();
    let n_max = cc.n_max();
    assert!(
        n_max >= 50.0,
        "CPU core tier N_max must be ≥50 with proportional formula at CG=1.0, got {n_max}"
    );
}

#[test]
fn eigen_calibration_full_independence_gives_n_eff_equal_n() {
    use nalgebra::DMatrix;
    // Identity matrix (N=4 fully independent adapters): all eigenvalues = 1
    // N_eff = (4)² / 4 = 4
    let sigma = DMatrix::<f64>::identity(4, 4);
    let ec = EigenCalibration::from_cg_matrix(&sigma);
    assert!(
        (ec.n_effective - 4.0).abs() < 0.1,
        "identity Σ → N_eff = 4, got {}",
        ec.n_effective
    );
    assert!(
        (ec.h_diversity - 1.0).abs() < 0.01,
        "identity Σ → H_norm = 1.0, got {}",
        ec.h_diversity
    );
}

#[test]
fn eigen_calibration_full_correlation_gives_n_eff_one() {
    use nalgebra::DMatrix;
    // Σ = all-ones matrix (rank 1): one eigenvalue = N, rest = 0
    // N_eff = N² / N² = 1
    let n = 4;
    let sigma = DMatrix::<f64>::from_element(n, n, 1.0);
    let ec = EigenCalibration::from_cg_matrix(&sigma);
    assert!(
        (ec.n_effective - 1.0).abs() < 0.5,
        "all-ones Σ → N_eff ≈ 1, got {}",
        ec.n_effective
    );
}

#[test]
fn eigen_calibration_uniform_rho_matches_portfolio_formula() {
    use nalgebra::DMatrix;
    // Uniform ρ=0.5, N=5: Choueifaty formula N_eff = N×(1−ρ)+ρ = 5×0.5+0.5 = 3.0
    let n = 5;
    let rho = 0.5f64;
    let mut sigma = DMatrix::<f64>::identity(n, n);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                sigma[(i, j)] = rho;
            }
        }
    }
    let ec = EigenCalibration::from_cg_matrix(&sigma);
    let expected = 2.5f64; // participation ratio for uniform rho=0.5, N=5
    assert!(
        (ec.n_effective - expected).abs() < 0.3,
        "uniform rho=0.5 -> N_eff approx {expected:.1}, got {:.3}",
        ec.n_effective
    );
}

#[test]
fn n_it_optimal_independent_returns_1() {
    assert_eq!(n_it_optimal(0.0), 1);
}

#[test]
fn n_it_optimal_fully_correlated_returns_9() {
    assert_eq!(n_it_optimal(1.0), 9);
}

#[test]
fn n_it_optimal_typical_rho_values() {
    // ρ=0.3: n = 1 + ln(0.5)/ln(0.7) = 1 + 1.943 → ceil = 3
    assert_eq!(n_it_optimal(0.3), 3);
    // ρ=0.5: n = 1 + ln(0.5)/ln(0.5) = 1 + 1.0 = 2.0 → ceil = 2
    assert_eq!(n_it_optimal(0.5), 2);
    // ρ=0.7: n = 1 + ln(0.5)/ln(0.3) = 1 + 0.576 → ceil = 2
    assert_eq!(n_it_optimal(0.7), 2);
    // ρ=0.9: n = 1 + ln(0.5)/ln(0.1) = 1 + 0.301 → ceil = 2
    assert_eq!(n_it_optimal(0.9), 2);
}

#[test]
fn ensemble_calibration_n_it_optimal_delegates_to_free_function() {
    // from_cg_mean(0.5, 9) → rho_mean = 1.0 - 0.5 = 0.5
    let ec = EnsembleCalibration::from_cg_mean(0.5, 9);
    assert_eq!(ec.n_it_optimal(), n_it_optimal(ec.rho_mean));
}

#[test]
fn ensemble_calibration_from_cg_mean_is_heuristic() {
    use h2ai_types::physics::PredictionBasis;
    let ec = EnsembleCalibration::from_cg_mean(0.7, 9);
    assert_eq!(
        ec.prediction_basis,
        PredictionBasis::Heuristic,
        "from_cg_mean must label prediction as Heuristic"
    );
}

#[test]
fn ensemble_calibration_from_measured_p_is_empirical() {
    use h2ai_types::physics::PredictionBasis;
    let ec = EnsembleCalibration::from_measured_p(0.85, 0.7, 9);
    assert_eq!(
        ec.prediction_basis,
        PredictionBasis::Empirical,
        "from_measured_p must label prediction as Empirical"
    );
}
