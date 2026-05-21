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
use h2ai_orchestrator::srani_gate::{compute_injection_pressure, update_ema};

const EPSILON: f64 = 1e-6;

#[test]
fn pressure_at_midpoint_is_exactly_half() {
    let p = compute_injection_pressure(0.45, 0.45, 0.15);
    assert!(
        (p - 0.5).abs() < EPSILON,
        "pressure at mu must be 0.5, got {p}"
    );
}

#[test]
fn pressure_well_below_midpoint_is_near_zero() {
    let p = compute_injection_pressure(0.0, 0.45, 0.15);
    assert!(p < 0.10, "pressure at CFI=0 should be < 0.10, got {p}");
}

#[test]
fn pressure_well_above_midpoint_is_near_one() {
    let p = compute_injection_pressure(1.0, 0.45, 0.15);
    assert!(p > 0.90, "pressure at CFI=1 should be > 0.90, got {p}");
}

#[test]
fn pressure_at_mu_plus_0_30_is_above_gate() {
    // mu=0.45, cfi=0.75: well above typical gate_threshold=0.50
    let p = compute_injection_pressure(0.75, 0.45, 0.15);
    assert!(p > 0.80, "pressure at mu+0.30 should be > 0.80, got {p}");
}

#[test]
fn pressure_at_mu_minus_0_30_is_below_warn_floor() {
    let p = compute_injection_pressure(0.15, 0.45, 0.15);
    assert!(
        p < 0.20,
        "pressure at mu-0.30 should be < 0.20 (warn floor), got {p}"
    );
}

#[test]
fn pressure_increases_monotonically_with_cfi() {
    let mu = 0.45;
    let t = 0.15;
    let mut prev = compute_injection_pressure(0.0, mu, t);
    for i in 1..=10 {
        let cfi = i as f64 * 0.1;
        let p = compute_injection_pressure(cfi, mu, t);
        assert!(p > prev, "pressure not monotone at cfi={cfi}");
        prev = p;
    }
}

#[test]
fn higher_temperature_produces_softer_curve() {
    // At CFI=mu+0.3, lower temperature → higher pressure (sharper)
    let p_sharp = compute_injection_pressure(0.75, 0.45, 0.10);
    let p_soft = compute_injection_pressure(0.75, 0.45, 0.30);
    assert!(p_sharp > p_soft, "lower T must produce higher pressure");
}

#[test]
fn ema_update_formula_correct() {
    // 0.20 * 0.70 + 0.80 * 0.45 = 0.14 + 0.36 = 0.50
    let result = update_ema(0.45, 0.70, 0.20);
    assert!(
        (result - 0.50).abs() < EPSILON,
        "ema_update wrong: {result}"
    );
}

#[test]
fn ema_with_alpha_1_equals_new_value() {
    let result = update_ema(0.45, 0.80, 1.0);
    assert!(
        (result - 0.80).abs() < EPSILON,
        "alpha=1 must return new value"
    );
}

#[test]
fn ema_with_alpha_0_returns_old_value() {
    let result = update_ema(0.45, 0.80, 0.0);
    assert!(
        (result - 0.45).abs() < EPSILON,
        "alpha=0 must return old ema"
    );
}
