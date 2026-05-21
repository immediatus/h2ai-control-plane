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
use h2ai_api::rho_ema::RhoEmaState;

#[test]
fn update_increments_n_observations() {
    let mut state = RhoEmaState::default();
    state.update(&[("a".into(), "b".into(), 0.4)], 0.1);
    assert_eq!(state.n_observations, 1);
}

#[test]
fn rho_mean_converges_toward_true_rho() {
    let true_rho = 0.40_f64;
    let mut state = RhoEmaState::default();
    for _ in 0..50 {
        state.update(&[("a".into(), "b".into(), true_rho)], 0.10);
    }
    let estimated = state.rho_mean();
    assert!(
        (estimated - true_rho).abs() < 0.05,
        "EMA should converge to ~0.40 after 50 updates, got {estimated:.4}"
    );
}

#[test]
fn rho_mean_returns_conservative_prior_when_empty() {
    let state = RhoEmaState::default();
    assert!((state.rho_mean() - 0.45).abs() < 1e-9);
}
