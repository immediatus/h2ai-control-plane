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
use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold};

#[test]
fn calibration_event_has_valid_n_max() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71]).unwrap();
    let n_max = cc.n_max();
    // New formula β_eff = β₀/max(CG,0.05): CG_mean≈0.71, β_eff=0.021/0.71≈0.030 → N_max≈5
    assert!(
        n_max > 1.0 && n_max < 20.0,
        "n_max={n_max} out of expected range"
    );
}

#[test]
fn calibration_theta_coord_bounded() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    assert!(theta.value() <= 0.3);
    assert!(theta.value() >= 0.0);
}
