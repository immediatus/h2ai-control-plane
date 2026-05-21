#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use h2ai_config::H2AIConfig;
use h2ai_orchestrator::self_optimizer::{
    OptimizerParams, QualityMeasurement, SelfOptimizer, SuggestInput, TauSpreadEstimator,
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
        q_confidence: 0.78,
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
fn suggest_lowers_threshold_on_zero_survival() {
    // When filter_ratio == 0 (all proposals failed), the optimizer must lower the
    // verify_threshold even though predict_q returns the same value for both params.
    // This is the ZeroSurvival path: threshold is the only knob to unblock proposals.
    let current = OptimizerParams {
        n_agents: 2,
        max_turns: 4,
        verify_threshold: 0.45,
    };
    let cfg = cfg();
    let suggestion = SelfOptimizer::suggest(SuggestInput {
        current: &current,
        history: &[],
        n_max_ceiling: 4,
        n_optimal: None,
        p_mean: P_MEAN,
        rho_mean: RHO_MEAN,
        filter_ratio: 0.0, // ZeroSurvival
        cfg: &cfg,
    });
    assert!(
        suggestion.verify_threshold < current.verify_threshold,
        "must lower threshold on zero-survival; got {:.3}",
        suggestion.verify_threshold
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
        use h2ai_types::sizing::PredictionBasis;
        let q = HarnessAttribution::compute(&AttributionInput {
            p_mean: P_MEAN,
            rho_mean: RHO_MEAN,
            n_agents: next.n_agents,
            verification_filter_ratio: FR,
            tao_turns_mean: next.max_turns as f64,
            tao_per_turn_factor: 0.6,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        })
        .q_confidence;
        assert!(
            q >= last_q - 1e-9,
            "quality regressed: {last_q:.4} → {q:.4}"
        );
        last_q = q;
        current = next;
    }
}

#[test]
fn tau_spread_estimator_defaults_to_initial_values() {
    let est = TauSpreadEstimator::new(0.2, 0.8);
    assert!((est.tau_min() - 0.2).abs() < 1e-9);
    assert!((est.tau_max() - 0.8).abs() < 1e-9);
}

#[test]
fn tau_spread_estimator_updates_toward_new_value() {
    let mut est = TauSpreadEstimator::new(0.2, 0.8);
    // EMA alpha=0.1: new_min = 0.2*0.9 + 0.3*0.1 = 0.21
    //                new_max = 0.8*0.9 + 0.9*0.1 = 0.81
    est.update(0.3, 0.9);
    assert!((est.tau_min() - 0.21).abs() < 1e-9, "got {}", est.tau_min());
    assert!((est.tau_max() - 0.81).abs() < 1e-9, "got {}", est.tau_max());
}

#[test]
fn tau_spread_estimator_clamps_to_unit_interval() {
    let mut est = TauSpreadEstimator::new(0.0, 1.0);
    est.update(-0.5, 1.5);
    assert!(est.tau_min() >= 0.0);
    assert!(est.tau_max() <= 1.0);
}
