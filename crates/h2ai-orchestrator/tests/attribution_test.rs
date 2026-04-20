use h2ai_orchestrator::attribution::{AttributionInput, HarnessAttribution};

#[test]
fn attribution_baseline_quality_is_one_minus_c_i() {
    let input = AttributionInput {
        baseline_c_i: 0.3,
        n_agents: 1,
        alpha: 0.0,
        kappa_eff: 0.0,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
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
        baseline_c_i: 0.3,
        n_agents: 1,
        alpha: 0.1,
        kappa_eff: 0.05,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
    };
    let ensemble = AttributionInput { n_agents: 4, ..base };
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
        baseline_c_i: 0.3,
        n_agents: 2,
        alpha: 0.1,
        kappa_eff: 0.05,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
    };
    let multi_turn = AttributionInput {
        tao_turns_mean: 3.0,
        ..base
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
        baseline_c_i: 0.01, // very good model
        n_agents: 8,
        alpha: 0.01,
        kappa_eff: 0.001,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 5.0,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(attr.total_quality <= 1.0, "total_quality must be clamped to 1.0");
    assert!(attr.total_quality >= 0.0);
}
