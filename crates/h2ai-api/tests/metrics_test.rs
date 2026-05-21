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
use h2ai_api::metrics::MetricsState;

#[test]
fn metrics_state_formats_prometheus_text() {
    let m = MetricsState {
        n_eff_prior: 2.5,
        n_eff_actual: 2.1,
        epistemic_yield_ratio: 0.7,
        mapek_mode_collapse_count: 3,
        mapek_constrained_exploration_count: 1,
        phase15_quadrant_precision: 10,
        phase15_quadrant_coverage: 42,
        phase15_quadrant_complex: 5,
        phase15_quadrant_degenerate: 1,
        oracle_ece: 0.0,
        oracle_n_observations: 0,
        oracle_coverage_rate: 0.0,
        oracle_pass_rate: 0.0,
        oracle_residual_p90: 0.0,
        oracle_calibration_basis: 0,
        oracle_tasks_total: 0,
        oracle_tasks_with_spec: 0,
        calibration_source_label: String::new(),
        shadow_audit_total: 0,
        shadow_audit_disagreements: 0,
        shadow_audit_promoted_domains: 0,
        shadow_audit_disagreement_rate: 0.0,
        safety_profile_name: "development".to_string(),
        safety_krum_fault_tolerance: 0,
        safety_diversity_threshold: 0.0,
        safety_shadow_auditor_enabled: 0,
        safety_require_bivariate_cg: 0,
    };
    let text = m.to_prometheus_text();
    assert!(text.contains("h2ai_n_eff_prior 2.5"));
    assert!(text.contains("h2ai_n_eff_actual 2.1"));
    assert!(text.contains("h2ai_epistemic_yield_ratio 0.7"));
    assert!(text.contains(r#"h2ai_mapek_interventions_total{failure_mode="mode_collapse"} 3"#));
    assert!(text
        .contains(r#"h2ai_mapek_interventions_total{failure_mode="constrained_exploration"} 1"#));
    assert!(text.contains(r#"h2ai_phase15_task_quadrant_total{quadrant="precision"} 10"#));
    assert!(text.contains(r#"h2ai_phase15_task_quadrant_total{quadrant="coverage"} 42"#));
    assert!(text.contains(r#"h2ai_phase15_task_quadrant_total{quadrant="complex"} 5"#));
    assert!(text.contains(r#"h2ai_phase15_task_quadrant_total{quadrant="degenerate"} 1"#));
}

#[test]
fn metrics_state_renders_calibration_source_measured() {
    let m = MetricsState {
        calibration_source_label: "measured".to_string(),
        ..Default::default()
    };
    let text = m.to_prometheus_text();
    assert!(text.contains(r#"h2ai_calibration_source{source="measured"} 1"#));
    assert!(text.contains(r#"h2ai_calibration_source{source="partial_fit"} 0"#));
    assert!(text.contains(r#"h2ai_calibration_source{source="synthetic_priors"} 0"#));
}

#[test]
fn metrics_state_renders_calibration_source_synthetic() {
    let m = MetricsState {
        calibration_source_label: "synthetic_priors".to_string(),
        ..Default::default()
    };
    let text = m.to_prometheus_text();
    assert!(text.contains(r#"h2ai_calibration_source{source="measured"} 0"#));
    assert!(text.contains(r#"h2ai_calibration_source{source="synthetic_priors"} 1"#));
}

#[test]
fn metrics_state_renders_calibration_source_partial_fit() {
    let m = MetricsState {
        calibration_source_label: "partial_fit".to_string(),
        ..Default::default()
    };
    let text = m.to_prometheus_text();
    assert!(text.contains(r#"h2ai_calibration_source{source="measured"} 0"#));
    assert!(text.contains(r#"h2ai_calibration_source{source="partial_fit"} 1"#));
    assert!(text.contains(r#"h2ai_calibration_source{source="synthetic_priors"} 0"#));
}

#[test]
fn oracle_metrics_render_in_prometheus_text() {
    let m = MetricsState {
        n_eff_prior: 2.5,
        n_eff_actual: 2.1,
        epistemic_yield_ratio: 0.7,
        mapek_mode_collapse_count: 3,
        mapek_constrained_exploration_count: 1,
        phase15_quadrant_precision: 10,
        phase15_quadrant_coverage: 42,
        phase15_quadrant_complex: 5,
        phase15_quadrant_degenerate: 1,
        oracle_ece: 0.08,
        oracle_n_observations: 45,
        oracle_coverage_rate: 0.6,
        oracle_pass_rate: 0.75,
        oracle_residual_p90: 0.35,
        oracle_calibration_basis: 2,
        oracle_tasks_total: 0,
        oracle_tasks_with_spec: 0,
        calibration_source_label: String::new(),
        shadow_audit_total: 0,
        shadow_audit_disagreements: 0,
        shadow_audit_promoted_domains: 0,
        shadow_audit_disagreement_rate: 0.0,
        safety_profile_name: "development".to_string(),
        safety_krum_fault_tolerance: 0,
        safety_diversity_threshold: 0.0,
        safety_shadow_auditor_enabled: 0,
        safety_require_bivariate_cg: 0,
    };
    let text = m.to_prometheus_text();
    assert!(text.contains("h2ai_oracle_ece_gauge 0.08"));
    assert!(text.contains("h2ai_oracle_n_observations_total 45"));
    assert!(text.contains("h2ai_oracle_coverage_rate 0.6"));
    assert!(text.contains("h2ai_oracle_pass_rate 0.75"));
    assert!(text.contains("h2ai_oracle_residual_p90 0.35"));
    assert!(text.contains("h2ai_calibration_basis 2"));
    assert!(text.contains("h2ai_oracle_tasks_total 0"));
    assert!(text.contains("h2ai_oracle_tasks_with_spec_total 0"));
}

#[test]
fn shadow_audit_metrics_render_in_prometheus_text() {
    let m = MetricsState {
        shadow_audit_total: 100,
        shadow_audit_disagreements: 8,
        shadow_audit_promoted_domains: 2,
        shadow_audit_disagreement_rate: 0.08,
        ..Default::default()
    };
    let text = m.to_prometheus_text();
    assert!(text.contains("h2ai_shadow_audit_total 100"));
    assert!(text.contains("h2ai_shadow_audit_disagreements_total 8"));
    assert!(text.contains("h2ai_shadow_audit_promoted_domains 2"));
    assert!(text.contains("h2ai_shadow_audit_disagreement_rate 0.08"));
}
