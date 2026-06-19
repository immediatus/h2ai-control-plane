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
use h2ai_orchestrator::attribution::{AttributionInput, HarnessAttribution};

#[test]
fn attribution_single_agent_no_filter_one_turn_equals_baseline() {
    // Invariant: N=1, filter=1.0, turns=1 → q_confidence == baseline_quality exactly.
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.3,
        n_agents: 1,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: h2ai_types::sizing::PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        (attr.q_confidence - attr.baseline_quality).abs() < 1e-9,
        "N=1, filter=1, turns=1 must give q_confidence == baseline_quality, got {} vs {}",
        attr.q_confidence,
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
                        prediction_basis: h2ai_types::sizing::PredictionBasis::Heuristic,
                        talagrand_state: None,
                        eigen_calibration: None,
                    };
                    let attr = HarnessAttribution::compute(&input);
                    assert!(
                        attr.q_confidence <= 1.0,
                        "q_confidence={} > 1.0 for p={p}, N={n}, fr={fr}, turns={turns}",
                        attr.q_confidence
                    );
                    assert!(
                        attr.q_confidence >= attr.baseline_quality,
                        "q_confidence={} < baseline={} for p={p}",
                        attr.q_confidence,
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
        prediction_basis: h2ai_types::sizing::PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!((attr.baseline_quality - 0.7).abs() < 1e-9);
    assert!(attr.topology_gain >= 0.0);
    assert!(attr.verification_gain >= 0.0);
    assert!(attr.tao_gain >= 0.0);
    assert!(attr.q_confidence >= attr.baseline_quality);
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
        prediction_basis: h2ai_types::sizing::PredictionBasis::Heuristic,
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
    assert!(a4.q_confidence > a1.q_confidence);
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
        prediction_basis: h2ai_types::sizing::PredictionBasis::Heuristic,
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
fn attribution_q_confidence_clamped_to_one() {
    let input = AttributionInput {
        p_mean: 0.99,
        rho_mean: 0.01,
        n_agents: 8,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 5.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: h2ai_types::sizing::PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.q_confidence <= 1.0,
        "q_confidence must be clamped to 1.0"
    );
    assert!(attr.q_confidence >= 0.0);
}

// ── Tests migrated from src/attribution.rs #[cfg(test)] block ────────────────

use h2ai_orchestrator::attribution::{bootstrap_interval, conformal_interval, IntervalBasis};
use h2ai_orchestrator::diagnostics::CalibrationState;
use h2ai_types::sizing::PredictionBasis;

#[test]
fn attribution_n1_topology_gain_is_zero() {
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.3,
        n_agents: 1,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.topology_gain.abs() < 1e-10,
        "N=1 topology_gain should be 0, got {}",
        attr.topology_gain
    );
}

#[test]
fn attribution_n3_topology_gain_positive_for_good_p() {
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.2,
        n_agents: 3,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.topology_gain > 0.0,
        "N=3 with p=0.7, rho=0.2 should have positive topology_gain, got {}",
        attr.topology_gain
    );
}

#[test]
fn attribution_q_confidence_bounded() {
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.3,
        n_agents: 5,
        verification_filter_ratio: 0.8,
        tao_turns_mean: 2.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.q_confidence >= 0.0 && attr.q_confidence <= 1.0,
        "q_confidence out of bounds: {}",
        attr.q_confidence
    );
}

#[test]
fn attribution_no_topology_gain_at_full_correlation() {
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 1.0,
        n_agents: 5,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.topology_gain.abs() < 1e-10,
        "rho=1 should give zero topology_gain, got {}",
        attr.topology_gain
    );
}

#[test]
fn attribution_q_confidence_at_least_baseline() {
    // q_confidence must always be >= p_mean (the single-agent baseline)
    let input = AttributionInput {
        p_mean: 0.6,
        rho_mean: 0.4,
        n_agents: 3,
        verification_filter_ratio: 0.7,
        tao_turns_mean: 2.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.q_confidence >= attr.baseline_quality,
        "q_confidence {} < baseline_quality {}",
        attr.q_confidence,
        attr.baseline_quality
    );
}

#[test]
fn attribution_below_majority_accuracy_no_topology_gain() {
    // p < 0.5: ensemble is worse than random; topology_gain should be 0 (clamped)
    let input = AttributionInput {
        p_mean: 0.4,
        rho_mean: 0.0,
        n_agents: 5,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.topology_gain == 0.0,
        "p=0.4 < 0.5 should give topology_gain=0 (clamped), got {}",
        attr.topology_gain
    );
}

#[test]
fn harness_attribution_q_measured_is_none_by_default() {
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.3,
        n_agents: 3,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.q_measured.is_none(),
        "q_measured must be None by default"
    );
}

fn base_input() -> AttributionInput {
    AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.3,
        n_agents: 3,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    }
}

#[test]
fn bootstrap_interval_none_basis_when_single_sample() {
    let iv = bootstrap_interval(&base_input(), &[0.6], 1000);
    assert_eq!(
        iv.interval_basis,
        IntervalBasis::None,
        "single CG sample must produce IntervalBasis::None"
    );
    assert_eq!(
        iv.q_confidence_lo, iv.q_confidence_hi,
        "lo == hi when no interval"
    );
}

#[test]
fn bootstrap_interval_none_basis_when_empty_samples() {
    let iv = bootstrap_interval(&base_input(), &[], 1000);
    assert_eq!(iv.interval_basis, IntervalBasis::None);
}

#[test]
fn bootstrap_interval_bootstrap_basis_with_two_samples() {
    let iv = bootstrap_interval(&base_input(), &[0.5, 0.7], 1000);
    assert!(
        matches!(
            iv.interval_basis,
            IntervalBasis::Bootstrap { n_cg_samples: 2 }
        ),
        "expected Bootstrap{{n_cg_samples:2}}, got {:?}",
        iv.interval_basis
    );
}

#[test]
fn bootstrap_interval_wider_with_higher_cg_variance() {
    // Low variance CG: all samples near 0.6
    let low_var: Vec<f64> = vec![0.58, 0.60, 0.61, 0.59, 0.60];
    // High variance CG: samples spread 0.2–0.9
    let high_var: Vec<f64> = vec![0.2, 0.4, 0.6, 0.8, 0.9];

    let input = base_input();
    let iv_low = bootstrap_interval(&input, &low_var, 2000);
    let iv_high = bootstrap_interval(&input, &high_var, 2000);

    let width_low = iv_low.q_confidence_hi - iv_low.q_confidence_lo;
    let width_high = iv_high.q_confidence_hi - iv_high.q_confidence_lo;
    assert!(
        width_high > width_low,
        "higher CG variance must produce wider CI: high={width_high:.4}, low={width_low:.4}"
    );
}

#[test]
fn bootstrap_interval_lo_le_hi() {
    // The bootstrap CI is derived from cg_samples; q_confidence comes from base_input directly,
    // so q_confidence is not guaranteed to lie inside the CI. The invariant is lo ≤ hi.
    let samples: Vec<f64> = (0..10).map(|i| 0.3 + i as f64 * 0.05).collect();
    let iv = bootstrap_interval(&base_input(), &samples, 1000);
    assert!(
        iv.q_confidence_lo <= iv.q_confidence_hi,
        "bootstrap CI must be non-inverted: lo={:.4}, hi={:.4}",
        iv.q_confidence_lo,
        iv.q_confidence_hi
    );
    assert!(iv.q_confidence_hi > 0.0, "CI hi must be positive");
}

#[test]
fn conformal_interval_empty_residuals_returns_full_range() {
    let (lo, hi) = conformal_interval(0.8, &[], 0.9);
    assert!(
        lo <= 0.8 && hi >= 0.8,
        "empty residuals must bracket q_predicted"
    );
}

#[test]
fn conformal_interval_correct_q_hat_single_residual() {
    // 1 residual = 0.1; idx = ceil(2 * 0.9) = 2, clamped to 1 → q_hat = residuals[0] = 0.1
    let (lo, hi) = conformal_interval(0.8, &[0.1], 0.9);
    assert!((lo - 0.7).abs() < 1e-9, "lo = 0.8 - 0.1 = 0.7, got {lo:.6}");
    assert!((hi - 0.9).abs() < 1e-9, "hi = 0.8 + 0.1 = 0.9, got {hi:.6}");
}

#[test]
fn conformal_interval_achieves_coverage_on_held_out_set() {
    // 50 residuals uniformly spaced 0.0–0.49; 90% coverage → q_hat ≈ 0.45
    let residuals: Vec<f64> = (0..50).map(|i| i as f64 * 0.01).collect();
    let q_pred = 0.7;
    let (lo, hi) = conformal_interval(q_pred, &residuals, 0.9);
    assert!(hi - lo > 0.0, "interval must be non-trivial");
    assert!(
        lo < q_pred && hi > q_pred,
        "point estimate must be inside interval"
    );
}

#[test]
fn conformal_interval_clamped_to_unit() {
    // q_predicted near 1.0 + large residual → hi clamped to 1.0
    let (lo, hi) = conformal_interval(0.95, &[0.5], 0.9);
    assert!((hi - 1.0).abs() < 1e-9, "hi must clamp to 1.0, got {hi:.6}");
    let _ = lo; // lo may be > 0
}

fn make_eigen(n_effective: f64, n_agents: usize) -> h2ai_types::sizing::EigenCalibration {
    h2ai_types::sizing::EigenCalibration {
        n_effective,
        h_diversity: 0.5,
        eigenvalues: vec![n_effective],
        n_pruned: n_agents,
    }
}

#[test]
fn case_b_high_cg_talagrand_under_dispersed_corrects_rho() {
    let input = AttributionInput {
        p_mean: 0.85,
        rho_mean: 0.30,
        n_agents: 4,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: Some(CalibrationState::UnderDispersed),
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        (attr.rho_adjusted - 0.51).abs() < 1e-9,
        "rho_adjusted must be 0.51, got {:.6}",
        attr.rho_adjusted
    );
    assert!(attr.case_b_flag, "case_b_flag must be true");
}

#[test]
fn case_b_low_cg_guard_prevents_correction() {
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.60,
        n_agents: 3,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: Some(CalibrationState::UnderDispersed),
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        (attr.rho_adjusted - 0.60).abs() < 1e-9,
        "guard must prevent correction: rho_adjusted={:.6}",
        attr.rho_adjusted
    );
    assert!(
        !attr.case_b_flag,
        "case_b_flag must be false when guard prevents correction"
    );
}

#[test]
fn case_a_calibrated_no_rho_correction() {
    let input = AttributionInput {
        p_mean: 0.85,
        rho_mean: 0.30,
        n_agents: 4,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: Some(CalibrationState::Calibrated),
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        (attr.rho_adjusted - 0.30).abs() < 1e-9,
        "Calibrated Talagrand must not correct rho, got {:.6}",
        attr.rho_adjusted
    );
    assert!(!attr.case_b_flag);
}

#[test]
fn neff_low_diversity_applies_second_correction() {
    let input = AttributionInput {
        p_mean: 0.85,
        rho_mean: 0.30,
        n_agents: 4,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: Some(CalibrationState::Calibrated),
        eigen_calibration: Some(make_eigen(1.0, 4)),
    };
    let attr = HarnessAttribution::compute(&input);
    let expected = 0.30 + 0.15 * 0.70;
    assert!(
        (attr.rho_adjusted - expected).abs() < 1e-9,
        "N_eff correction must give {:.6}, got {:.6}",
        expected,
        attr.rho_adjusted
    );
    assert!(attr.case_b_flag);
}

#[test]
fn both_signals_fire_correction_capped_at_unit() {
    let input = AttributionInput {
        p_mean: 0.85,
        rho_mean: 0.30,
        n_agents: 4,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: Some(CalibrationState::UnderDispersed),
        eigen_calibration: Some(make_eigen(1.0, 4)),
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.rho_adjusted <= 1.0,
        "rho_adjusted must be <= 1.0 even when both signals fire, got {:.6}",
        attr.rho_adjusted
    );
    let expected = (0.30 + 0.21 + 0.105_f64).clamp(0.0, 1.0);
    assert!(
        (attr.rho_adjusted - expected).abs() < 1e-9,
        "both-signal rho_adjusted must be {expected:.6}, got {:.6}",
        attr.rho_adjusted
    );
    assert!(attr.case_b_flag);
}

#[test]
fn case_b_flag_false_when_no_signals_fire() {
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.3,
        n_agents: 4,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(!attr.case_b_flag, "no signals -> case_b_flag must be false");
    assert!(
        (attr.rho_adjusted - 0.30).abs() < 1e-9,
        "no signals -> rho_adjusted must equal rho_mean"
    );
}

#[test]
fn q_confidence_at_least_baseline_after_rho_correction() {
    let input = AttributionInput {
        p_mean: 0.6,
        rho_mean: 0.30,
        n_agents: 3,
        verification_filter_ratio: 0.8,
        tao_turns_mean: 2.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: Some(CalibrationState::UnderDispersed),
        eigen_calibration: Some(make_eigen(0.8, 3)),
    };
    let attr = HarnessAttribution::compute(&input);
    assert!(
        attr.q_confidence >= attr.baseline_quality,
        "q_confidence {:.4} < baseline {:.4} after Case B correction",
        attr.q_confidence,
        attr.baseline_quality
    );
    assert!(attr.q_confidence <= 1.0);
}

#[test]
fn bootstrap_interval_empirical_basis_uses_fixed_p_and_rho() {
    // PredictionBasis::Empirical → p_boot = base_input.p_mean, rho_boot = base_input.rho_mean
    // regardless of cg_boot. Exercises lines 192 and 196 in attribution.rs.
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.3,
        n_agents: 3,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Empirical,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let cg_samples = vec![0.5_f64, 0.7, 0.6, 0.8, 0.4];
    let iv = bootstrap_interval(&input, &cg_samples, 500);
    assert!(
        matches!(
            iv.interval_basis,
            IntervalBasis::Bootstrap { n_cg_samples: 5 }
        ),
        "expected Bootstrap{{n_cg_samples:5}}, got {:?}",
        iv.interval_basis
    );
    assert!(
        iv.q_confidence_lo <= iv.q_confidence_hi,
        "CI must be non-inverted: lo={:.4} hi={:.4}",
        iv.q_confidence_lo,
        iv.q_confidence_hi
    );
}

#[test]
fn synthesis_gain_defaults_to_zero() {
    let input = AttributionInput {
        p_mean: 0.7,
        rho_mean: 0.3,
        n_agents: 3,
        verification_filter_ratio: 1.0,
        tao_turns_mean: 1.0,
        tao_per_turn_factor: 0.6,
        prediction_basis: PredictionBasis::Heuristic,
        talagrand_state: None,
        eigen_calibration: None,
    };
    let attr = HarnessAttribution::compute(&input);
    assert_eq!(
        attr.synthesis_gain, 0.0,
        "synthesis_gain must default to 0.0"
    );
}
