use h2ai_types::physics::{
    CoherencyCoefficients, CoordinationThreshold, JeffectiveGap, MergeStrategy,
    MultiplicationCondition, MultiplicationConditionFailure, RoleErrorCost, TauValue,
};

#[test]
fn coherency_coefficients_beta_eff_computation() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        beta_base: 0.020,
        cg_samples: vec![0.6, 0.7, 0.65],
    };
    // CG_mean = (0.6+0.7+0.65)/3 = 0.65
    // β_eff = 0.020 / 0.65 ≈ 0.03077
    let beta_eff = cc.beta_eff();
    let expected = 0.020 / 0.65;
    assert!(
        (beta_eff - expected).abs() < 1e-10,
        "β_eff = β₀/CG_mean = {expected:.6}, got {beta_eff:.6}"
    );
}

#[test]
fn coherency_coefficients_computes_n_max_usl() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        beta_base: 0.020,
        cg_samples: vec![0.65],
    };
    // β_eff = 0.020/0.65 ≈ 0.03077
    // N_max = round(√(0.88/0.03077)) = round(√28.6) = round(5.35) = 5
    let n_max = cc.n_max();
    let expected = ((1.0_f64 - 0.12) / (0.020 / 0.65)).sqrt().round();
    assert!(
        (n_max - expected).abs() < 1e-9,
        "N_max = round(√((1−α)/β_eff)) = {expected:.3}, got {n_max}"
    );
}

#[test]
fn coordination_threshold_formula_verified() {
    // θ_coord = clamp(CG_mean − CG_std_dev, 0, max)
    // For [0.6, 0.7, 0.65]: mean=0.65, std≈0.0408, spread≈0.6092 → clamped to 0.3
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        beta_base: 0.020,
        cg_samples: vec![0.6, 0.7, 0.65],
    };
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    // spread = 0.65 − 0.0408 ≈ 0.6092 → clamp to 0.3
    assert!(
        (theta.value() - 0.3).abs() < 1e-9,
        "θ_coord must be clamped to max=0.3 when spread > max; got {}",
        theta.value()
    );

    // For low CG samples: spread < max → θ_coord = spread
    let cc_low = CoherencyCoefficients {
        alpha: 0.12,
        beta_base: 0.020,
        cg_samples: vec![0.1, 0.15, 0.12],
    };
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
    let cc = CoherencyCoefficients {
        alpha: 0.10,
        beta_base: 0.015,
        cg_samples: vec![0.55, 0.70, 0.62],
    };
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
fn coordination_threshold_is_min_of_cg_spread_and_floor() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        beta_base: 0.020,
        cg_samples: vec![0.6, 0.7, 0.65],
    };
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    assert!((theta.value() - 0.3).abs() < 1e-9);
}

#[test]
fn coordination_threshold_uses_spread_when_cg_very_low() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        beta_base: 0.020,
        cg_samples: vec![0.1, 0.15, 0.12],
    };
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    assert!(theta.value() < 0.3);
    assert!(theta.value() > 0.0);
}

#[test]
fn coordination_threshold_serde_round_trip() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        beta_base: 0.020,
        cg_samples: vec![0.6, 0.7, 0.65],
    };
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
fn merge_strategy_krum_serializes_and_deserializes() {
    let s = MergeStrategy::Krum { f: 1 };
    let json = serde_json::to_string(&s).unwrap();
    let back: MergeStrategy = serde_json::from_str(&json).unwrap();
    assert_eq!(back, s);
}

#[test]
fn merge_strategy_multi_krum_serializes_and_deserializes() {
    let s = MergeStrategy::MultiKrum { f: 2, m: 3 };
    let json = serde_json::to_string(&s).unwrap();
    let back: MergeStrategy = serde_json::from_str(&json).unwrap();
    assert_eq!(back, s);
}

#[test]
fn from_role_costs_selects_krum_above_krum_threshold() {
    let costs = vec![RoleErrorCost::new(0.97).unwrap()];
    let strategy = MergeStrategy::from_role_costs(&costs, 0.85, 0.95, 1);
    assert_eq!(strategy, MergeStrategy::Krum { f: 1 });
}

#[test]
fn from_role_costs_selects_condorcet_in_middle_tier() {
    let costs = vec![RoleErrorCost::new(0.90).unwrap()];
    let strategy = MergeStrategy::from_role_costs(&costs, 0.85, 0.95, 1);
    assert_eq!(strategy, MergeStrategy::ConsensusMedian);
}

#[test]
fn from_role_costs_krum_disabled_when_f_is_zero() {
    // krum_f=0 disables Krum even above krum_threshold
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
    assert!(result.is_ok(), "cg_mean == theta must pass (strict < semantics)");
}

#[test]
fn multiplication_condition_competence_failure_takes_priority_over_others() {
    // All three conditions fail, but InsufficientCompetence is checked first.
    let result = MultiplicationCondition::evaluate(
        0.1,  // competence below min 0.5
        0.99, // correlation above max 0.9
        0.0,  // cg below theta 0.3
        0.3,
        0.5,
        0.9,
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
    // AI-agent tier: α=0.15, β₀=0.01, CG_mean=0.4 → β_eff=0.025 → N_max=round(√34)=6
    let cc = CoherencyCoefficients::new(0.15, 0.01, vec![0.4]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 6.0).abs() < 1.0,
        "AI-agent tier N_max must be ≈6, got {n_max}"
    );
}

#[test]
fn coherency_coefficients_beta_eff_divides_by_cg_mean() {
    // β_eff = β₀ / CG_mean = 0.01 / 0.4 = 0.025
    let cc = CoherencyCoefficients::new(0.15, 0.01, vec![0.4]).unwrap();
    let beta_eff = cc.beta_eff();
    assert!(
        (beta_eff - 0.025).abs() < 1e-10,
        "β_eff = β₀/CG_mean = 0.025, got {beta_eff}"
    );
}

#[test]
fn coherency_coefficients_human_team_tier() {
    // Human team tier: α=0.10, β₀=0.005, CG_mean=0.6 → N_max≈10
    let cc = CoherencyCoefficients::new(0.10, 0.005, vec![0.6]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 10.0).abs() < 1.5,
        "Human team N_max must be ≈10, got {n_max}"
    );
}

#[test]
fn coherency_coefficients_cpu_core_tier() {
    // CPU tier: α=0.02, β₀=0.0003, CG_mean=1.0 → N_max≈57
    let cc = CoherencyCoefficients::new(0.02, 0.0003, vec![1.0]).unwrap();
    let n_max = cc.n_max();
    assert!(
        (n_max - 57.0).abs() < 2.0,
        "CPU core tier N_max must be ≈57, got {n_max}"
    );
}
