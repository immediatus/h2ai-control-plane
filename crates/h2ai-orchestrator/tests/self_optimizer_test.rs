use h2ai_config::H2AIConfig;
use h2ai_orchestrator::self_optimizer::{
    OptimizerParams, QualityMeasurement, SelfOptimizer, SuggestInput,
};

const P_MEAN: f64 = 0.75;
const RHO_MEAN: f64 = 0.25;
const FR: f64 = 1.0; // no filtering initially

fn cfg() -> H2AIConfig {
    H2AIConfig::default()
}

#[test]
fn suggest_raises_tao_turns_before_agents() {
    // When max_turns < 4, raising TAO should be preferred over adding agents
    // (Proposition 8 MAPE-K guidance: first TAO turn gives 22× more gain than last agent)
    let current = OptimizerParams {
        n_agents: 4,
        max_turns: 1,
        verify_threshold: 0.45,
    };
    let cfg = cfg();
    let suggestion = SelfOptimizer::suggest(SuggestInput {
        current: &current,
        history: &[],
        n_max_ceiling: 6,
        n_optimal: None,
        p_mean: P_MEAN,
        rho_mean: RHO_MEAN,
        filter_ratio: FR,
        cfg: &cfg,
    });
    assert_eq!(
        suggestion.max_turns, 2,
        "should raise TAO turns first (max_turns 1→2)"
    );
    assert_eq!(suggestion.n_agents, 4, "should not change n_agents");
}

#[test]
fn suggest_does_not_exceed_n_max_ceiling() {
    let current = OptimizerParams {
        n_agents: 5,
        max_turns: 4,
        verify_threshold: 0.3,
    };
    let cfg = cfg();
    let suggestion = SelfOptimizer::suggest(SuggestInput {
        current: &current,
        history: &[],
        n_max_ceiling: 5,
        n_optimal: None,
        p_mean: P_MEAN,
        rho_mean: RHO_MEAN,
        filter_ratio: FR,
        cfg: &cfg,
    });
    assert_eq!(
        suggestion.n_agents, 5,
        "n_agents must not exceed n_max_ceiling"
    );
}

#[test]
fn suggest_returns_current_when_at_all_ceilings() {
    let current = OptimizerParams {
        n_agents: 5,
        max_turns: 4,
        verify_threshold: 0.3,
    };
    let cfg = cfg();
    let suggestion = SelfOptimizer::suggest(SuggestInput {
        current: &current,
        history: &[],
        n_max_ceiling: 5,
        n_optimal: None,
        p_mean: P_MEAN,
        rho_mean: RHO_MEAN,
        filter_ratio: FR,
        cfg: &cfg,
    });
    assert_eq!(
        suggestion, current,
        "should return current when nothing improves"
    );
}

#[test]
fn suggest_skips_already_tried_params() {
    let current = OptimizerParams {
        n_agents: 4,
        max_turns: 1,
        verify_threshold: 0.45,
    };
    let history = vec![QualityMeasurement {
        params: OptimizerParams {
            max_turns: 2,
            ..current.clone()
        },
        q_total: 0.78,
    }];
    let cfg = cfg();
    let suggestion = SelfOptimizer::suggest(SuggestInput {
        current: &current,
        history: &history,
        n_max_ceiling: 6,
        n_optimal: None,
        p_mean: P_MEAN,
        rho_mean: RHO_MEAN,
        filter_ratio: FR,
        cfg: &cfg,
    });
    assert_ne!(
        suggestion.max_turns, 2,
        "should not re-suggest already-tried params"
    );
}

#[test]
fn suggest_respects_n_optimal_below_n_max_ceiling() {
    // n_optimal=3 means we should not suggest n_agents > 3, even though n_max_ceiling=6
    let current = OptimizerParams {
        n_agents: 3,
        max_turns: 4,
        verify_threshold: 0.3,
    };
    let cfg = cfg();
    let suggestion = SelfOptimizer::suggest(SuggestInput {
        current: &current,
        history: &[],
        n_max_ceiling: 6,
        n_optimal: Some(3),
        p_mean: P_MEAN,
        rho_mean: RHO_MEAN,
        filter_ratio: FR,
        cfg: &cfg,
    });
    assert!(
        suggestion.n_agents <= 3,
        "n_agents must not exceed n_optimal=3 even when n_max_ceiling=6; got {}",
        suggestion.n_agents
    );
}

#[test]
fn suggest_uses_n_optimal_as_target_not_max_ceiling() {
    // Start below n_optimal=3. Optimizer should suggest going to 3, not further.
    let current = OptimizerParams {
        n_agents: 1,
        max_turns: 4,
        verify_threshold: 0.3,
    };
    let cfg = cfg();
    let suggestion = SelfOptimizer::suggest(SuggestInput {
        current: &current,
        history: &[],
        n_max_ceiling: 9,
        n_optimal: Some(3),
        p_mean: P_MEAN,
        rho_mean: RHO_MEAN,
        filter_ratio: FR,
        cfg: &cfg,
    });
    assert!(
        suggestion.n_agents <= 3,
        "optimizer with n_optimal=3 must not suggest n_agents > 3; got {}",
        suggestion.n_agents
    );
}

#[test]
fn quality_is_monotone_in_suggested_direction() {
    let mut current = OptimizerParams {
        n_agents: 1,
        max_turns: 1,
        verify_threshold: 0.9,
    };
    let mut last_q = 0.0_f64;
    let cfg = cfg();
    for _ in 0..8 {
        let next = SelfOptimizer::suggest(SuggestInput {
            current: &current,
            history: &[],
            n_max_ceiling: 6,
            n_optimal: None,
            p_mean: P_MEAN,
            rho_mean: RHO_MEAN,
            filter_ratio: FR,
            cfg: &cfg,
        });
        if next == current {
            break;
        }
        use h2ai_orchestrator::attribution::{AttributionInput, HarnessAttribution};
        let q = HarnessAttribution::compute(&AttributionInput {
            p_mean: P_MEAN,
            rho_mean: RHO_MEAN,
            n_agents: next.n_agents,
            verification_filter_ratio: FR,
            tao_turns_mean: next.max_turns as f64,
            tao_per_turn_factor: 0.6,
        })
        .total_quality;
        assert!(
            q >= last_q - 1e-9,
            "quality regressed: {last_q:.4} → {q:.4}"
        );
        last_q = q;
        current = next;
    }
}
