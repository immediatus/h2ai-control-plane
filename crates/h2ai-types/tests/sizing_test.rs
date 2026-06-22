use h2ai_types::sizing::{
    condorcet_quality, n_it_optimal, tau_alignment, CoherencyCoefficients, CoordinationThreshold,
    EigenCalibration, EnsembleCalibration, MergeStrategy, MultiplicationCondition,
    MultiplicationConditionFailure, OracleDomain, OracleFamily, PredictionBasis, RoleErrorCost,
    TauValue, CG_HALFLIFE_SECS,
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
fn cg_std_dev_uses_sample_variance() {
    // Three samples: 0.6, 0.7, 0.8 → mean=0.7, sample variance = (0.01+0+0.01)/2 = 0.01
    // sample std = 0.1  (population std would be sqrt(0.01*2/3) ≈ 0.0816)
    let cc = CoherencyCoefficients::new(0.12, 0.020, vec![0.6, 0.7, 0.8]).unwrap();
    let std = cc.cg_std_dev();
    assert!(
        (std - 0.1).abs() < 1e-9,
        "cg_std_dev must use sample variance (n-1): expected 0.1, got {std}"
    );
}

#[test]
#[allow(clippy::float_cmp)]
fn cg_std_dev_single_sample_is_zero() {
    let cc = CoherencyCoefficients::new(0.12, 0.020, vec![0.7]).unwrap();
    assert_eq!(cc.cg_std_dev(), 0.0);
}

#[test]
#[allow(clippy::float_cmp)]
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
#[allow(clippy::float_cmp)]
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
#[allow(clippy::float_cmp)]
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
#[allow(clippy::float_cmp)]
fn tau_value_valid_range() {
    assert!(TauValue::new(0.5).is_ok());
    assert_eq!(TauValue::new(0.5).unwrap().value(), 0.5);
}

#[test]
#[allow(clippy::float_cmp)]
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
    let ec = EigenCalibration::from_cg_matrix(&sigma, 0.05);
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
    let ec = EigenCalibration::from_cg_matrix(&sigma, 0.05);
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
    let ec = EigenCalibration::from_cg_matrix(&sigma, 0.05);
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
    use h2ai_types::sizing::PredictionBasis;
    let ec = EnsembleCalibration::from_cg_mean(0.7, 9);
    assert_eq!(
        ec.prediction_basis,
        PredictionBasis::Heuristic,
        "from_cg_mean must label prediction as Heuristic"
    );
}

#[test]
fn ensemble_calibration_from_measured_p_is_empirical() {
    use h2ai_types::sizing::PredictionBasis;
    let ec = EnsembleCalibration::from_measured_p(0.85, 0.7, 9);
    assert_eq!(
        ec.prediction_basis,
        PredictionBasis::Empirical,
        "from_measured_p must label prediction as Empirical"
    );
}

#[test]
fn n_max_ci_range_contains_point_estimate() {
    // Two samples with spread: mean=0.6, std≈0.1
    let cc = h2ai_types::sizing::CoherencyCoefficients::new(0.1, 0.02, vec![0.5, 0.7]).unwrap();
    let (lo, hi) = cc.n_max_ci();
    let mid = cc.n_max();
    assert!(
        lo <= mid + 1.0,
        "ci_low must be ≤ point estimate (±1 due to rounding): lo={lo}, mid={mid}"
    );
    assert!(
        hi >= mid - 1.0,
        "ci_high must be ≥ point estimate (±1 due to rounding): hi={hi}, mid={mid}"
    );
    assert!(hi >= lo, "ci must be ordered: lo={lo}, hi={hi}");
}

#[test]
#[allow(clippy::float_cmp)]
fn n_max_ci_equal_samples_returns_degenerate_interval() {
    // One sample → std_dev = 0 → both bounds equal point estimate
    let cc = h2ai_types::sizing::CoherencyCoefficients::new(0.1, 0.02, vec![0.6]).unwrap();
    let (lo, hi) = cc.n_max_ci();
    let mid = cc.n_max();
    assert_eq!(lo, mid, "single sample: lo must equal n_max");
    assert_eq!(hi, mid, "single sample: hi must equal n_max");
}

#[test]
fn eigen_calibration_delta_controls_n_pruned() {
    use nalgebra::DMatrix;
    // 4x4 identity: all eigenvalues=1, each increment=1.0
    let sigma = DMatrix::identity(4, 4);
    // With delta=0.05 (default), increment 1.0 >= 0.05 -> n_pruned=4
    let ec_default = h2ai_types::sizing::EigenCalibration::from_cg_matrix(&sigma, 0.05);
    assert_eq!(ec_default.n_pruned, 4, "delta=0.05: all 4 adapters needed");
    // With delta=1.5 (very high), increment 1.0 < 1.5 -> n_pruned=1
    let ec_high = h2ai_types::sizing::EigenCalibration::from_cg_matrix(&sigma, 1.5);
    assert_eq!(ec_high.n_pruned, 1, "delta=1.5: only 1 adapter passes");
}

// ── condorcet_quality and tau_alignment ──────────────────────────────────────

#[test]
#[allow(clippy::float_cmp)]
fn condorcet_quality_n0_returns_zero() {
    assert_eq!(condorcet_quality(0, 0.7, 0.2), 0.0);
}

#[test]
#[allow(clippy::float_cmp)]
fn condorcet_quality_p_zero_returns_zero() {
    assert_eq!(condorcet_quality(3, 0.0, 0.2), 0.0);
}

#[test]
#[allow(clippy::float_cmp)]
fn condorcet_quality_p_one_returns_one() {
    assert_eq!(condorcet_quality(3, 1.0, 0.2), 1.0);
}

#[test]
fn tau_alignment_same_tau_is_one() {
    let a = TauValue::new(0.5).unwrap();
    let b = TauValue::new(0.5).unwrap();
    let result = tau_alignment(a, b);
    assert!(
        (result - 1.0).abs() < 1e-10,
        "same τ → alignment 1.0, got {result}"
    );
}

#[test]
fn tau_alignment_far_apart_is_small() {
    let a = TauValue::new(0.0).unwrap();
    let b = TauValue::new(1.0).unwrap();
    let result = tau_alignment(a, b);
    // exp(-3 * 1.0) ≈ 0.0498
    assert!(
        result < 0.06,
        "τ distance 1.0 → small alignment, got {result}"
    );
    assert!(result > 0.04, "τ distance 1.0 → ~0.05, got {result}");
}

#[test]
fn condorcet_quality_n1_equals_p() {
    for p in [0.3, 0.5, 0.7, 0.9] {
        let q = condorcet_quality(1, p, 0.3);
        assert!((q - p).abs() < 1e-10, "N=1 → Q=p for p={p}, got {q}");
    }
}

#[test]
fn condorcet_quality_full_correlation_equals_p() {
    for n in [3usize, 5, 7] {
        let q = condorcet_quality(n, 0.7, 1.0);
        assert!((q - 0.7).abs() < 1e-10, "ρ=1 → Q=p for N={n}, got {q}");
    }
}

#[test]
fn condorcet_quality_increases_with_n_for_p_above_half() {
    let qs: Vec<f64> = [1usize, 3, 5, 7, 9]
        .iter()
        .map(|&n| condorcet_quality(n, 0.7, 0.2))
        .collect();
    for i in 0..qs.len() - 1 {
        assert!(
            qs[i + 1] >= qs[i],
            "Q should be non-decreasing in N for p=0.7, rho=0.2: {qs:?}"
        );
    }
}

#[test]
fn condorcet_quality_bounded_01() {
    for n in [1usize, 3, 5, 7, 9] {
        for p_int in [30i32, 50, 70, 90] {
            let p = f64::from(p_int) / 100.0;
            for rho_int in [0i32, 20, 50, 80, 100] {
                let rho = f64::from(rho_int) / 100.0;
                let q = condorcet_quality(n, p, rho);
                assert!(
                    (0.0..=1.0).contains(&q),
                    "Q out of [0,1]: N={n} p={p} rho={rho} → {q}"
                );
            }
        }
    }
}

#[test]
fn ensemble_calibration_from_cg_mean_n_optimal_at_least_1() {
    for cg_int in [20i32, 40, 60, 80] {
        let cg = f64::from(cg_int) / 100.0;
        let ec = EnsembleCalibration::from_cg_mean(cg, 9);
        assert!(ec.n_optimal >= 1, "n_optimal >= 1 for cg={cg}");
        assert!(ec.q_optimal >= ec.p_mean, "q_optimal >= p_mean for cg={cg}");
    }
}

#[test]
fn ensemble_calibration_n_optimal_greater_than_1_for_typical_cg() {
    let ec = EnsembleCalibration::from_cg_mean(0.7, 9);
    assert!(
        ec.n_optimal > 1,
        "n_optimal should be >1 for cg=0.7, got {}",
        ec.n_optimal
    );
}

#[test]
fn ensemble_calibration_low_cg_still_recommends_small_ensemble() {
    let ec = EnsembleCalibration::from_cg_mean(0.001, 9);
    assert!(
        ec.n_optimal >= 1,
        "n_optimal must be >= 1, got {}",
        ec.n_optimal
    );
    assert!(
        ec.n_optimal <= 5,
        "very high correlation should give small n_optimal, got {}",
        ec.n_optimal
    );
}

#[test]
fn ensemble_calibration_quality_at_n1_equals_p() {
    let ec = EnsembleCalibration::from_cg_mean(0.7, 9);
    let q = ec.quality_at_n(1);
    assert!(
        (q - ec.p_mean).abs() < 1e-10,
        "quality_at_n(1) == p_mean, got {q} vs {}",
        ec.p_mean
    );
}

#[test]
fn ensemble_calibration_topology_gain_non_negative() {
    for cg_int in [20i32, 50, 80] {
        let cg = f64::from(cg_int) / 100.0;
        let ec = EnsembleCalibration::from_cg_mean(cg, 9);
        assert!(ec.topology_gain() >= 0.0, "topology_gain >= 0 for cg={cg}");
    }
}

#[test]
fn ensemble_calibration_from_measured_p_uses_given_p() {
    let ec = EnsembleCalibration::from_measured_p(0.9, 0.7, 9);
    assert!(
        (ec.p_mean - 0.9).abs() < 1e-10,
        "p_mean should be 0.9, got {}",
        ec.p_mean
    );
}

#[test]
fn from_empirical_sets_empirical_basis_and_exact_rho() {
    let ec = EnsembleCalibration::from_empirical(0.75, 0.35, 9);
    assert_eq!(ec.prediction_basis, PredictionBasis::Empirical);
    assert!(
        (ec.rho_mean - 0.35).abs() < 1e-9,
        "rho_mean must be set directly"
    );
    assert!((ec.p_mean - 0.75).abs() < 1e-9, "p_mean must match input");
    assert!(ec.n_optimal >= 1);
}

#[test]
fn from_empirical_clamps_rho_to_valid_range() {
    let ec_low = EnsembleCalibration::from_empirical(0.7, -0.5, 9);
    let ec_high = EnsembleCalibration::from_empirical(0.7, 2.0, 9);
    assert!(ec_low.rho_mean >= 0.0);
    assert!(ec_high.rho_mean <= 0.99);
}

#[test]
fn beta_eff_temporal_fresh_sample_equals_beta_eff() {
    let now = 1_000_000u64;
    let cc = CoherencyCoefficients::new_with_timestamps(0.1, 0.02, vec![0.6], vec![now]).unwrap();
    let result = cc.beta_eff_temporal(now);
    let expected = cc.beta_eff();
    assert!(
        (result - expected).abs() < 1e-9,
        "fresh sample: {result} vs {expected}"
    );
}

#[test]
fn beta_eff_temporal_stale_sample_approaches_beta_base() {
    let now = CG_HALFLIFE_SECS * 100;
    let cc = CoherencyCoefficients::new_with_timestamps(0.1, 0.05, vec![0.8], vec![0]).unwrap();
    let result = cc.beta_eff_temporal(now);
    assert!(
        (result - cc.beta_base).abs() < 0.001,
        "stale sample must approach beta_base={}, got {result}",
        cc.beta_base
    );
}

#[test]
fn beta_eff_temporal_no_timestamps_falls_back_to_beta_eff() {
    let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.6, 0.7]).unwrap();
    let result = cc.beta_eff_temporal(1_000_000);
    assert!((result - cc.beta_eff()).abs() < 1e-9);
}

#[test]
fn beta_eff_temporal_empty_struct_timestamps_falls_back() {
    let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.6]).unwrap();
    let result = cc.beta_eff_temporal(1_000_000);
    assert!((result - cc.beta_eff()).abs() < 1e-9);
}

#[test]
fn beta_eff_temporal_recent_low_cg_dominates_old_high_cg() {
    let now = CG_HALFLIFE_SECS * 10;
    let cc = CoherencyCoefficients::new_with_timestamps(0.1, 0.05, vec![0.9, 0.2], vec![0u64, now])
        .unwrap();
    let result = cc.beta_eff_temporal(now);
    let fresh_only_beta = cc.beta_base * (1.0 - 0.2_f64);
    assert!(
        (result - fresh_only_beta).abs() < 0.005,
        "recent low-CG sample must dominate: expected ≈{fresh_only_beta:.4}, got {result:.4}"
    );
}

#[test]
fn beta_eff_uses_beta_quality_when_present_no_cg_adjustment() {
    let mut cc = CoherencyCoefficients::new(0.1, 0.05, vec![0.7]).unwrap();
    cc.beta_quality = Some(0.3);
    let eff = cc.beta_eff();
    assert!((eff - 0.3).abs() < 1e-9, "expected 0.3, got {eff}");
}

#[test]
fn beta_eff_falls_back_to_proxy_when_beta_quality_none() {
    let cc = CoherencyCoefficients::new(0.1, 0.05, vec![0.6]).unwrap();
    assert!(cc.beta_quality.is_none());
    let eff = cc.beta_eff();
    assert!((eff - 0.02).abs() < 1e-9, "expected 0.02, got {eff}");
}

#[test]
fn n_max_increases_with_lower_beta_quality() {
    let mut cc_high = CoherencyCoefficients::new(0.1, 0.05, vec![0.5]).unwrap();
    cc_high.beta_quality = Some(0.4);

    let mut cc_low = CoherencyCoefficients::new(0.1, 0.05, vec![0.5]).unwrap();
    cc_low.beta_quality = Some(0.05);

    assert!(cc_low.n_max() > cc_high.n_max());
}

// ── EigenCalibration::from_cosine_matrix ─────────────────────────────────────

#[test]
#[allow(clippy::cast_precision_loss)]
fn from_cosine_matrix_full_collapse_n_eff_is_one() {
    use nalgebra::DMatrix;
    let n = 3usize;
    let k = DMatrix::from_element(n, n, 1.0 / n as f64);
    let ec = EigenCalibration::from_cosine_matrix(&k, 0.05);
    assert!(
        (ec.n_effective - 1.0).abs() < 1e-6,
        "all-same embeddings → N_eff=1, got {}",
        ec.n_effective
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn from_cosine_matrix_full_diversity_n_eff_equals_n() {
    use nalgebra::DMatrix;
    let n = 3usize;
    let k = DMatrix::identity(n, n) / n as f64;
    let ec = EigenCalibration::from_cosine_matrix(&k, 0.05);
    assert!(
        (ec.n_effective - n as f64).abs() < 1e-6,
        "orthogonal embeddings → N_eff={n}, got {}",
        ec.n_effective
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn from_cosine_matrix_partial_collapse_n_eff_approx_1_8() {
    use nalgebra::DMatrix;
    let n = 3usize;
    let raw = DMatrix::from_row_slice(n, n, &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let k = raw / n as f64;
    let ec = EigenCalibration::from_cosine_matrix(&k, 0.05);
    assert!(
        (ec.n_effective - 1.8).abs() < 0.01,
        "partial collapse → N_eff≈1.8, got {}",
        ec.n_effective
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn from_cosine_matrix_n2_full_diversity() {
    use nalgebra::DMatrix;
    let n = 2usize;
    let k = DMatrix::identity(n, n) / n as f64;
    let ec = EigenCalibration::from_cosine_matrix(&k, 0.05);
    assert!(
        (ec.n_effective - 2.0).abs() < 1e-6,
        "N=2 orthogonal → N_eff=2, got {}",
        ec.n_effective
    );
}

// ── n_max_context_aware ───────────────────────────────────────────────────────

#[test]
#[allow(clippy::float_cmp)]
fn n_max_context_aware_returns_n_max_when_tokens_below_one() {
    let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.6]).unwrap();
    // proposal_tokens < 1.0 → fallback to n_max()
    assert_eq!(cc.n_max_context_aware(0.5, 4096.0, 1.0), cc.n_max());
    // max_tokens < 1.0 → fallback to n_max()
    assert_eq!(cc.n_max_context_aware(200.0, 0.0, 1.0), cc.n_max());
}

#[test]
fn n_max_context_aware_reduces_n_max_under_context_pressure() {
    // High context pressure (large gamma) should reduce N_max vs unconstrained.
    let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.6]).unwrap();
    let unconstrained = cc.n_max();
    let constrained = cc.n_max_context_aware(1000.0, 2000.0, 5.0);
    assert!(
        constrained <= unconstrained,
        "context pressure must reduce or equal n_max: constrained={constrained}, unconstrained={unconstrained}"
    );
    assert!(constrained >= 1.0, "n_max_context_aware must be >= 1");
}

#[test]
fn n_max_context_aware_zero_gamma_equals_n_max() {
    // gamma=0: no attention degradation → result equals n_max() (fixed-point at initial value)
    let cc = CoherencyCoefficients::new(0.1, 0.02, vec![0.6]).unwrap();
    let result = cc.n_max_context_aware(200.0, 8192.0, 0.0);
    assert!(
        (result - cc.n_max()).abs() < 1.0,
        "gamma=0 → n_max_context_aware ≈ n_max, got {result} vs {}",
        cc.n_max()
    );
}

#[test]
fn n_max_context_aware_result_at_least_one() {
    // Even with extreme pressure the result is clamped to >= 1
    let cc = CoherencyCoefficients::new(0.5, 0.5, vec![0.01]).unwrap();
    let result = cc.n_max_context_aware(10000.0, 100.0, 100.0);
    assert!(
        result >= 1.0,
        "n_max_context_aware must be >= 1, got {result}"
    );
}

// ── MergeStrategy::min_krum_quorum ────────────────────────────────────────────

#[test]
fn min_krum_quorum_matches_formula() {
    assert_eq!(MergeStrategy::min_krum_quorum(0), 3);
    assert_eq!(MergeStrategy::min_krum_quorum(1), 5);
    assert_eq!(MergeStrategy::min_krum_quorum(2), 7);
}

// ── OspConfig::default ────────────────────────────────────────────────────────

#[test]
fn osp_config_default_values() {
    use h2ai_types::sizing::OspConfig;
    let cfg = OspConfig::default();
    assert!((cfg.t_v - 0.125).abs() < 1e-10);
    assert!((cfg.concordance_alpha - 0.1).abs() < 1e-10);
    assert_eq!(cfg.max_n_v_for_zone3, 4);
    assert!((cfg.accumulation_decay - 0.7).abs() < 1e-10);
}

#[test]
fn osp_config_serde_round_trip() {
    use h2ai_types::sizing::OspConfig;
    let cfg = OspConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let back: OspConfig = serde_json::from_str(&json).unwrap();
    assert!((cfg.t_v - back.t_v).abs() < 1e-10);
    assert_eq!(cfg.max_n_v_for_zone3, back.max_n_v_for_zone3);
}

// ── JeffectiveGap ─────────────────────────────────────────────────────────────

#[test]
fn jeffective_gap_valid_range() {
    use h2ai_types::sizing::JeffectiveGap;
    assert!(JeffectiveGap::new(0.0).is_ok());
    assert!(JeffectiveGap::new(0.5).is_ok());
    assert!(JeffectiveGap::new(1.0).is_ok());
}

#[test]
fn jeffective_gap_invalid_out_of_range() {
    use h2ai_types::sizing::JeffectiveGap;
    assert!(JeffectiveGap::new(-0.1).is_err());
    assert!(JeffectiveGap::new(1.1).is_err());
}

#[test]
fn jeffective_gap_value_accessor() {
    use h2ai_types::sizing::JeffectiveGap;
    let j = JeffectiveGap::new(0.7).unwrap();
    assert!((j.value() - 0.7).abs() < 1e-10);
}

#[test]
fn jeffective_gap_is_below_threshold() {
    use h2ai_types::sizing::JeffectiveGap;
    let j = JeffectiveGap::new(0.4).unwrap();
    assert!(j.is_below_threshold(0.5));
    assert!(!j.is_below_threshold(0.3));
    assert!(!j.is_below_threshold(0.4), "strictly less than, not <=");
}

#[test]
fn jeffective_gap_error_message_contains_value() {
    use h2ai_types::sizing::JeffectiveGap;
    let err = JeffectiveGap::new(1.5).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("1.5"),
        "error message must contain bad value: {msg}"
    );
}

// ── PhysicsError display messages ─────────────────────────────────────────────

#[test]
fn physics_error_display_messages() {
    use h2ai_types::sizing::PhysicsError;
    assert!(PhysicsError::InvalidAlpha(1.5).to_string().contains("1.5"));
    assert!(PhysicsError::InvalidErrorCost(-0.1)
        .to_string()
        .contains("-0.1"));
    assert!(PhysicsError::InvalidJeff(2.0).to_string().contains('2'));
    assert!(PhysicsError::InvalidTau(1.5).to_string().contains("1.5"));
    assert!(PhysicsError::EmptyCgSamples.to_string().contains("empty"));
    assert!(PhysicsError::InvalidBetaBase(-1.0)
        .to_string()
        .contains("-1"));
}

// ── EigenCalibration::rho_eff ─────────────────────────────────────────────────

#[test]
fn eigen_calibration_rho_eff_full_independence() {
    use nalgebra::DMatrix;
    let sigma = DMatrix::<f64>::identity(4, 4);
    let ec = EigenCalibration::from_cg_matrix(&sigma, 0.05);
    // N_eff = 4, n = 4 → rho_eff = 1 - 4/4 = 0.0
    let rho = ec.rho_eff(4);
    assert!(
        (rho - 0.0).abs() < 0.1,
        "full independence → rho_eff ≈ 0, got {rho}"
    );
}

#[test]
fn eigen_calibration_rho_eff_full_correlation() {
    use nalgebra::DMatrix;
    let n = 4usize;
    let sigma = DMatrix::<f64>::from_element(n, n, 1.0);
    let ec = EigenCalibration::from_cg_matrix(&sigma, 0.05);
    // N_eff ≈ 1, n = 4 → rho_eff = 1 - 1/4 = 0.75
    let rho = ec.rho_eff(n);
    assert!(rho > 0.5, "full correlation → high rho_eff, got {rho}");
    assert!(rho <= 1.0, "rho_eff must be in [0,1], got {rho}");
}

#[test]
fn eigen_calibration_rho_eff_clamps_to_zero() {
    use nalgebra::DMatrix;
    // n_effective = 4, n passed = 3 → 1 - 4/3 < 0 → clamp to 0
    let sigma = DMatrix::<f64>::identity(4, 4);
    let ec = EigenCalibration::from_cg_matrix(&sigma, 0.05);
    let rho = ec.rho_eff(3);
    assert!(rho >= 0.0, "rho_eff must clamp to >= 0, got {rho}");
}

// ── from_cg_matrix single-element path ───────────────────────────────────────

#[test]
#[allow(clippy::float_cmp)]
fn eigen_calibration_single_element_matrix() {
    use nalgebra::DMatrix;
    // 1x1 matrix: h_norm takes the `else 0.0` branch (evs.len() == 1)
    let sigma = DMatrix::<f64>::from_element(1, 1, 1.0);
    let ec = EigenCalibration::from_cg_matrix(&sigma, 0.05);
    assert!((ec.n_effective - 1.0).abs() < 1e-6);
    assert_eq!(ec.h_diversity, 0.0);
    assert_eq!(ec.n_pruned, 1);
}

#[test]
#[allow(clippy::float_cmp)]
fn from_cosine_matrix_single_element() {
    use nalgebra::DMatrix;
    // 1x1 cosine matrix: evs.len()==1 → h_norm=0.0
    let k = DMatrix::<f64>::from_element(1, 1, 1.0);
    let ec = EigenCalibration::from_cosine_matrix(&k, 0.05);
    assert!((ec.n_effective - 1.0).abs() < 1e-6);
    assert_eq!(ec.h_diversity, 0.0);
}

// ── from_cosine_matrix delta controls n_pruned ────────────────────────────────

#[test]
#[allow(clippy::cast_precision_loss)]
fn from_cosine_matrix_delta_controls_n_pruned() {
    use nalgebra::DMatrix;
    // 4x4 identity normalized by N: orthogonal → each increment = 1/4 of full
    let n = 4usize;
    let k = DMatrix::identity(n, n) / n as f64;
    // With tiny delta → all adapters pass
    let ec_small = EigenCalibration::from_cosine_matrix(&k, 0.01);
    assert_eq!(
        ec_small.n_pruned, 4,
        "small delta → 4 adapters, got {}",
        ec_small.n_pruned
    );
    // With large delta → pruned to 1
    let ec_large = EigenCalibration::from_cosine_matrix(&k, 10.0);
    assert_eq!(
        ec_large.n_pruned, 1,
        "large delta → 1 adapter, got {}",
        ec_large.n_pruned
    );
}

// ── Zero/near-zero matrix edge cases ─────────────────────────────────────────
// These hit the `else { 1.0 }` branches inside from_cg_matrix / from_cosine_matrix
// when sum_sq (or partial_sum_sq) is <= 1e-12, i.e. all eigenvalues are ~0.

#[test]
fn from_cg_matrix_zero_matrix_n_eff_is_one() {
    use nalgebra::DMatrix;
    // A zero matrix has all zero eigenvalues → sum_sq ≤ 1e-12 → n_eff = 1 (fallback).
    let sigma = DMatrix::<f64>::zeros(4, 4);
    let ec = EigenCalibration::from_cg_matrix(&sigma, 0.05);
    assert!(
        (ec.n_effective - 1.0).abs() < 1e-6,
        "zero matrix → n_eff=1 (underflow branch), got {}",
        ec.n_effective
    );
}

#[test]
fn from_cosine_matrix_zero_matrix_n_eff_is_one() {
    use nalgebra::DMatrix;
    // A zero cosine kernel → all eigenvalues zero → sum_sq ≤ 1e-12 → n_eff = 1.
    let k = DMatrix::<f64>::zeros(4, 4);
    let ec = EigenCalibration::from_cosine_matrix(&k, 0.05);
    assert!(
        (ec.n_effective - 1.0).abs() < 1e-6,
        "zero cosine matrix → n_eff=1 (underflow branch), got {}",
        ec.n_effective
    );
}

// ── Fix A: N_max hard floor (quorum invariant) ────────────────────────────────

#[test]
fn n_max_ci_applies_hard_floor_of_three_when_degraded() {
    // alpha=0.9, beta_base=0.1, cg=[0.4] → n_max ≈ 1 (below BFT quorum)
    let cc = CoherencyCoefficients::new(0.9, 0.1, vec![0.4]).unwrap();
    let (lo, hi) = cc.n_max_ci();
    assert!(lo >= 3.0, "n_max_ci lo must be floored at 3, got {lo}");
    assert!(hi >= 3.0, "n_max_ci hi must be floored at 3, got {hi}");
    assert!(lo <= hi, "lo must not exceed hi");
}

#[test]
fn n_max_ci_does_not_alter_healthy_values() {
    // alpha=0.1, beta_base=0.01, cg=[0.7] → n_max >> 3
    let cc = CoherencyCoefficients::new(0.1, 0.01, vec![0.7]).unwrap();
    let (lo, hi) = cc.n_max_ci();
    assert!(
        lo > 3.0,
        "healthy n_max_ci lo should exceed floor, got {lo}"
    );
    assert!(hi >= lo, "hi must be >= lo");
}

#[test]
fn n_max_degraded_returns_true_when_below_quorum() {
    // alpha=0.9, beta_base=0.1, cg=[0.4] → unclamped n_max ≈ 1 < 3
    let cc = CoherencyCoefficients::new(0.9, 0.1, vec![0.4]).unwrap();
    assert!(
        cc.n_max_degraded(),
        "must detect degradation below quorum floor"
    );
}

#[test]
fn n_max_degraded_returns_false_in_healthy_regime() {
    let cc = CoherencyCoefficients::new(0.1, 0.01, vec![0.7]).unwrap();
    assert!(
        !cc.n_max_degraded(),
        "healthy calibration must not report degradation"
    );
}

#[test]
fn quorum_degraded_multiplication_condition_failure_variant_is_constructible() {
    let failure = MultiplicationConditionFailure::QuorumDegradedBelowMinimum {
        unclamped_n_max: 1.0,
    };
    let msg = failure.to_string();
    assert!(
        msg.contains("1"),
        "error message must include the unclamped value"
    );
}

// ── Fix D: OracleFamily mapping ───────────────────────────────────────────────

#[test]
fn oracle_family_code_maps_to_syntactic() {
    assert_eq!(OracleDomain::Code.family(), OracleFamily::Syntactic);
}

#[test]
fn oracle_family_factual_maps_to_semantic() {
    assert_eq!(OracleDomain::Factual.family(), OracleFamily::Semantic);
}

#[test]
fn oracle_family_reasoning_maps_to_semantic() {
    assert_eq!(OracleDomain::Reasoning.family(), OracleFamily::Semantic);
}

#[test]
fn oracle_family_human_maps_to_human() {
    assert_eq!(OracleDomain::Human.family(), OracleFamily::Human);
}

#[test]
fn oracle_family_unknown_maps_to_semantic() {
    assert_eq!(OracleDomain::Unknown.family(), OracleFamily::Semantic);
}

// ─── BetaCalibrationSource ────────────────────────────────────────────────────

use h2ai_types::sizing::BetaCalibrationSource;

#[test]
fn beta_calibration_theoretical_effective_beta() {
    let src = BetaCalibrationSource::Theoretical {
        assumed_beta: 0.039,
    };
    assert!((src.effective_beta() - 0.039).abs() < 1e-10);
}

#[test]
fn beta_calibration_empirical_effective_beta() {
    let src = BetaCalibrationSource::Empirical {
        fitted_beta: 0.051,
        r_squared: Some(0.94),
    };
    assert!((src.effective_beta() - 0.051).abs() < 1e-10);
}

#[test]
fn beta_calibration_empirical_no_r_squared() {
    let src = BetaCalibrationSource::Empirical {
        fitted_beta: 0.027,
        r_squared: None,
    };
    assert!((src.effective_beta() - 0.027).abs() < 1e-10);
}

#[test]
fn beta_calibration_source_serde_round_trip_theoretical() {
    let src = BetaCalibrationSource::Theoretical {
        assumed_beta: 0.039,
    };
    let json = serde_json::to_string(&src).unwrap();
    let back: BetaCalibrationSource = serde_json::from_str(&json).unwrap();
    assert_eq!(src, back);
}

#[test]
fn beta_calibration_source_serde_round_trip_empirical() {
    let src = BetaCalibrationSource::Empirical {
        fitted_beta: 0.051,
        r_squared: Some(0.94),
    };
    let json = serde_json::to_string(&src).unwrap();
    let back: BetaCalibrationSource = serde_json::from_str(&json).unwrap();
    assert_eq!(src, back);
}
