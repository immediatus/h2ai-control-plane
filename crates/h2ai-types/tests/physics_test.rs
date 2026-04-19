use h2ai_types::physics::{
    CoherencyCoefficients, CoordinationThreshold, JeffectiveGap, MergeStrategy,
    MultiplicationCondition, MultiplicationConditionFailure, RoleErrorCost,
};

#[test]
fn coherency_coefficients_computes_kappa_eff() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        kappa_base: 0.020,
        cg_samples: vec![0.6, 0.7, 0.65],
    };
    let kappa_eff = cc.kappa_eff();
    assert!((kappa_eff - 0.03077).abs() < 0.001);
}

#[test]
fn coherency_coefficients_computes_n_max() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        kappa_base: 0.020,
        cg_samples: vec![0.6, 0.7, 0.65],
    };
    let n_max = cc.n_max();
    assert!(n_max > 5.0 && n_max < 6.0);
}

#[test]
fn coherency_coefficients_serde_round_trip() {
    let cc = CoherencyCoefficients {
        alpha: 0.10,
        kappa_base: 0.015,
        cg_samples: vec![0.55, 0.70, 0.62],
    };
    let json = serde_json::to_string(&cc).unwrap();
    let back: CoherencyCoefficients = serde_json::from_str(&json).unwrap();
    assert_eq!(cc.alpha, back.alpha);
    assert_eq!(cc.cg_samples.len(), back.cg_samples.len());
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
fn coordination_threshold_is_min_of_cg_spread_and_floor() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        kappa_base: 0.020,
        cg_samples: vec![0.6, 0.7, 0.65],
    };
    let theta = CoordinationThreshold::from_calibration(&cc);
    assert!((theta.value() - 0.3).abs() < 1e-9);
}

#[test]
fn coordination_threshold_uses_spread_when_cg_very_low() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        kappa_base: 0.020,
        cg_samples: vec![0.1, 0.15, 0.12],
    };
    let theta = CoordinationThreshold::from_calibration(&cc);
    assert!(theta.value() < 0.3);
    assert!(theta.value() > 0.0);
}

#[test]
fn coordination_threshold_serde_round_trip() {
    let cc = CoherencyCoefficients {
        alpha: 0.12,
        kappa_base: 0.020,
        cg_samples: vec![0.6, 0.7, 0.65],
    };
    let theta = CoordinationThreshold::from_calibration(&cc);
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
fn merge_strategy_is_crdt_when_max_ci_at_or_below_threshold() {
    let costs = vec![
        RoleErrorCost::new(0.3).unwrap(),
        RoleErrorCost::new(0.85).unwrap(),
    ];
    assert_eq!(
        MergeStrategy::from_role_costs(&costs),
        MergeStrategy::CrdtSemilattice
    );
}

#[test]
fn merge_strategy_is_bft_when_max_ci_above_threshold() {
    let costs = vec![
        RoleErrorCost::new(0.3).unwrap(),
        RoleErrorCost::new(0.91).unwrap(),
    ];
    assert_eq!(
        MergeStrategy::from_role_costs(&costs),
        MergeStrategy::BftConsensus
    );
}

#[test]
fn merge_strategy_serde_round_trip() {
    for strategy in [MergeStrategy::CrdtSemilattice, MergeStrategy::BftConsensus] {
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
    let result = MultiplicationCondition::evaluate(0.7, 0.85, 0.65, 0.3);
    assert!(result.is_ok());
}

#[test]
fn multiplication_condition_fails_on_low_competence() {
    let result = MultiplicationCondition::evaluate(0.4, 0.85, 0.65, 0.3);
    assert!(matches!(
        result,
        Err(MultiplicationConditionFailure::InsufficientCompetence { .. })
    ));
}

#[test]
fn multiplication_condition_fails_on_high_correlation() {
    let result = MultiplicationCondition::evaluate(0.7, 0.95, 0.65, 0.3);
    assert!(matches!(
        result,
        Err(MultiplicationConditionFailure::InsufficientDecorrelation { .. })
    ));
}

#[test]
fn multiplication_condition_fails_when_cg_below_theta() {
    let result = MultiplicationCondition::evaluate(0.7, 0.85, 0.2, 0.3);
    assert!(matches!(
        result,
        Err(MultiplicationConditionFailure::CommonGroundBelowFloor { .. })
    ));
}
