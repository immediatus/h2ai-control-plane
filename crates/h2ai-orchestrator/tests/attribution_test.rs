use h2ai_orchestrator::attribution::{AttributionInput, HarnessAttribution};

#[test]
fn attribution_single_agent_no_filter_one_turn_equals_baseline() {
    // Invariant: N=1, filter=1.0, turns=1 → total_quality == baseline_quality exactly.
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.3,
        n_agents: 1,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: h2ai_types::physics::PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        (attr.total_quality - attr.baseline_quality).abs() < 1e-9,
        "N=1, filter=1, turns=1 must give total_quality == baseline_quality, got {} vs {}",
        attr.total_quality,
        attr.baseline_quality
    );
}

#[test]
fn attribution_total_never_exceeds_one_under_high_gain_conditions() {
    // Verify multiplicative model stays ≤ 1.0 in all regimes.
    for &p in &[0.5_f64, 0.7, 0.9] {
        for &n in &[1u32, 4, 8] {
            for &fr in &[0.3_f64, 0.7, 1.0] {
                for &turns in &[1.0_f64, 3.0, 5.0] {
                    let input = AttributionInput {
                        p_mean: p,
                        rho_mean: 0.2,
                        n_agents: n,
                        verification_filter_ratio: fr,
                        tao_turns_mean: turns,
                        tao_per_turn_factor: 0.6,
                        prediction_basis: h2ai_types::physics::PredictionBasis::Heuristic,
                        talagrand_state: None,
                        eigen_calibration: None,
                    };
                    let attr = HarnessAttribution::compute(&input);
                    assert!(
                        attr.total_quality <= 1.0,
                        "total_quality={} > 1.0 for p={p}, N={n}, fr={fr}, turns={turns}",
                        attr.total_quality
                    );
                    assert!(
                        attr.total_quality >= attr.baseline_quality,
                        "total_quality={} < baseline={} for p={p}",
                        attr.total_quality,
                        attr.baseline_quality
                    );
                }
            }
        }
    }
}

#[test]
fn attribution_baseline_quality_is_p_mean() {
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.0,
        n_agents: 1,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: h2ai_types::physics::PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!((attr.baseline_quality - 0.7).abs() < 1e-9);
    assert!(attr.topology_gain >= 0.0);
    assert!(attr.verification_gain >= 0.0);
    assert!(attr.tao_gain >= 0.0);
    assert!(attr.total_quality >= attr.baseline_quality);
}

#[test]
fn attribution_topology_gain_increases_with_more_agents() {
    let base = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.2,
        n_agents: 1,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: h2ai_types::physics::PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let ensemble = AttributionInput {
        n_agents: 4,
        ..base.clone()
    };
    let a1 = HarnessAttribution::compute(&base);
    let a4 = HarnessAttribution::compute(&ensemble);
    assert!(
        a4.topology_gain > a1.topology_gain,
        "ensemble must have higher topology_gain than single agent"
    );
    assert!(a4.total_quality > a1.total_quality);
}

#[test]
fn attribution_tao_gain_increases_with_more_turns() {
    let base = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.2,
        n_agents: 2,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: h2ai_types::physics::PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let multi_turn = AttributionInput {
        tao_turns_mean: 3.0,
        ..base.clone()
    };
    let a1 = HarnessAttribution::compute(&base);
    let a3 = HarnessAttribution::compute(&multi_turn);
    assert!(
        a3.tao_gain > a1.tao_gain,
        "more TAO turns must yield higher tao_gain"
    );
}

#[test]
fn attribution_total_quality_clamped_to_one() {
    let input = AttributionInput {
        p_mean: 0.99,
        rho_mean: 0.01,
        n_agents: 8,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 5.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: h2ai_types::physics::PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.total_quality <= 1.0,
        "total_quality must be clamped to 1.0"
    );
    assert!(attr.total_quality >= 0.0);
}
