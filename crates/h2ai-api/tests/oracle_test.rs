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
use h2ai_api::oracle::{
    determine_calibration_basis, ece_from_observations, pass_rate_from_observations, residual_p90,
};
use h2ai_types::sizing::{OracleDomain, OracleObservation};

fn obs(q: f64, y: bool) -> OracleObservation {
    OracleObservation {
        task_id: "t".into(),
        q_confidence: q,
        y_oracle: y,
        residual: (q - f64::from(u8::from(y))).abs(),
        domain: OracleDomain::Code,
        timestamp_ms: 0,
    }
}

#[test]
fn ece_empty_returns_zero() {
    assert_eq!(ece_from_observations(&[]), 0.0);
}

#[test]
fn ece_perfect_confidence_zero() {
    // q=1.0 and passed=true → residual=0 for each → ECE=0
    let observations = vec![obs(1.0, true), obs(1.0, true), obs(1.0, true)];
    let ece = ece_from_observations(&observations);
    assert!(ece.abs() < 1e-9, "perfect calibration → ECE=0, got {ece}");
}

#[test]
fn ece_formula_mean_residuals() {
    // residuals: |0.8 - 1| = 0.2, |0.4 - 0| = 0.4, |0.6 - 1| = 0.4
    // ECE = (0.2 + 0.4 + 0.4) / 3 = 1.0/3 ≈ 0.333
    let observations = vec![obs(0.8, true), obs(0.4, false), obs(0.6, true)];
    let ece = ece_from_observations(&observations);
    let expected = (0.2 + 0.4 + 0.4) / 3.0;
    assert!(
        (ece - expected).abs() < 1e-9,
        "ECE={ece} expected={expected}"
    );
}

#[test]
fn pass_rate_all_passed() {
    let observations = vec![obs(0.9, true), obs(0.8, true)];
    assert!((pass_rate_from_observations(&observations) - 1.0).abs() < 1e-9);
}

#[test]
fn pass_rate_half() {
    let observations = vec![obs(0.9, true), obs(0.3, false)];
    assert!((pass_rate_from_observations(&observations) - 0.5).abs() < 1e-9);
}

#[test]
fn pass_rate_empty_returns_zero() {
    assert_eq!(pass_rate_from_observations(&[]), 0.0);
}

#[test]
fn residual_p90_sorted() {
    // 10 residuals: 0.1, 0.2, ..., 1.0
    // Angelopoulos-Bates: ⌈(10+1) × 0.9⌉ − 1 = ⌈9.9⌉ − 1 = 9
    // Index 9 (0-based) = 1.0
    let mut observations: Vec<OracleObservation> = (1..=10)
        .map(|i| {
            let r = i as f64 * 0.1;
            OracleObservation {
                task_id: "t".into(),
                q_confidence: 0.5,
                y_oracle: false,
                residual: r,
                domain: OracleDomain::Code,

                timestamp_ms: i,
            }
        })
        .collect();
    // shuffle to verify sorting
    observations.reverse();
    let p90 = residual_p90(&observations);
    assert!(
        (p90 - 1.0).abs() < 1e-9,
        "P90 of 0.1..1.0 (n=10) with Angelopoulos-Bates should be 1.0, got {p90}"
    );
}

#[test]
fn residual_p90_empty_returns_zero() {
    assert_eq!(residual_p90(&[]), 0.0);
}

#[test]
fn basis_heuristic_below_10_obs() {
    let observations: Vec<OracleObservation> = (0..9)
        .map(|i| OracleObservation {
            task_id: format!("t{i}"),
            q_confidence: 0.9,
            y_oracle: true,
            residual: 0.1,
            domain: OracleDomain::Code,

            timestamp_ms: i as u64,
        })
        .collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(status.basis, 0, "n=9 → Heuristic (basis=0)");
}

#[test]
fn basis_bootstrap_at_10_obs() {
    let observations: Vec<OracleObservation> = (0..10)
        .map(|i| OracleObservation {
            task_id: format!("t{i}"),
            q_confidence: 0.9,
            y_oracle: true,
            residual: 0.1,
            domain: OracleDomain::Code,

            timestamp_ms: i as u64,
        })
        .collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(status.basis, 1, "n=10 → Bootstrap (basis=1)");
}

#[test]
fn basis_bootstrap_at_29_obs() {
    let observations: Vec<OracleObservation> = (0..29)
        .map(|i| OracleObservation {
            task_id: format!("t{i}"),
            q_confidence: 0.9,
            y_oracle: true,
            residual: 0.1,
            domain: OracleDomain::Code,

            timestamp_ms: i as u64,
        })
        .collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(status.basis, 1, "n=29 → Bootstrap (basis=1)");
}

#[test]
fn basis_conformal_at_30_obs_low_ece() {
    let observations: Vec<OracleObservation> = (0..30)
        .map(|i| OracleObservation {
            task_id: format!("t{i}"),
            q_confidence: 0.9,
            y_oracle: true,
            residual: 0.1,
            domain: OracleDomain::Code,

            timestamp_ms: i as u64,
        })
        .collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(status.basis, 2, "n=30 ECE=0.1 → Conformal (basis=2)");
}

#[test]
fn basis_heuristic_at_30_obs_high_ece() {
    let observations: Vec<OracleObservation> = (0..30)
        .map(|i| OracleObservation {
            task_id: format!("t{i}"),
            q_confidence: 0.5,
            y_oracle: false,
            residual: 0.5,
            domain: OracleDomain::Code,

            timestamp_ms: i as u64,
        })
        .collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(
        status.basis, 0,
        "n=30 ECE=0.5 → Heuristic (basis=0, quality regression)"
    );
}

#[test]
fn residual_p90_angelopoulos_bates_at_n30() {
    let obs: Vec<OracleObservation> = (0..30)
        .map(|i| OracleObservation {
            task_id: format!("t{i}"),
            q_confidence: 0.5,
            y_oracle: false,
            residual: f64::from(i) / 29.0,
            domain: OracleDomain::Unknown,

            timestamp_ms: 0,
        })
        .collect();
    let p90 = residual_p90(&obs);
    assert!(
        p90 > 0.92,
        "expected Angelopoulos-Bates p90 ≈ 0.931, got {p90}"
    );
}
