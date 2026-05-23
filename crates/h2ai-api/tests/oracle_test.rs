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
    determine_calibration_basis, ece_from_observations, enforce_fifo_cap,
    pass_rate_from_observations, patch_ensemble_p_from_oracle, residual_p90,
};
use h2ai_types::events::CalibrationCompletedEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{
    CoherencyCoefficients, CoordinationThreshold, EnsembleCalibration, OracleDomain,
    OracleObservation,
};
use std::sync::Arc;
use tokio::sync::RwLock;

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

#[test]
fn residual_p90_single_element() {
    // n=1: ⌈(1+1)×0.9⌉−1=1, min(1, 0)=0 → returns element[0]
    let single = vec![OracleObservation {
        task_id: "t".into(),
        q_confidence: 0.5,
        y_oracle: false,
        residual: 0.42,
        domain: OracleDomain::Code,
        timestamp_ms: 0,
    }];
    assert!((residual_p90(&single) - 0.42).abs() < 1e-9);
}

// ── enforce_fifo_cap ───────────────────────────────────────────────────────

#[test]
fn fifo_cap_under_cap_is_noop() {
    let mut v: Vec<OracleObservation> = (0..5).map(|_| obs(0.8, true)).collect();
    enforce_fifo_cap(&mut v, 10);
    assert_eq!(v.len(), 5);
}

#[test]
fn fifo_cap_at_exact_cap_is_noop() {
    let mut v: Vec<OracleObservation> = (0..10).map(|_| obs(0.8, true)).collect();
    enforce_fifo_cap(&mut v, 10);
    assert_eq!(v.len(), 10);
}

#[test]
fn fifo_cap_drops_oldest_entries() {
    let mut v: Vec<OracleObservation> = (0u64..5)
        .map(|i| OracleObservation {
            task_id: format!("t{i}"),
            q_confidence: 0.8,
            y_oracle: true,
            residual: 0.2,
            domain: OracleDomain::Code,
            timestamp_ms: i,
        })
        .collect();
    enforce_fifo_cap(&mut v, 3);
    assert_eq!(v.len(), 3);
    assert_eq!(v[0].task_id, "t2", "oldest two (t0,t1) evicted");
    assert_eq!(v[2].task_id, "t4", "newest retained");
}

#[test]
fn fifo_cap_zero_clears_all() {
    let mut v: Vec<OracleObservation> = (0..5).map(|_| obs(0.5, false)).collect();
    enforce_fifo_cap(&mut v, 0);
    assert!(v.is_empty());
}

// ── patch_ensemble_p_from_oracle ───────────────────────────────────────────

fn make_calibration_event(p_mean: f64) -> CalibrationCompletedEvent {
    let cc = CoherencyCoefficients::new(0.1, 0.01, vec![0.7]).unwrap();
    let ct = CoordinationThreshold::from_calibration(&cc, 1.0);
    let ensemble = EnsembleCalibration::from_measured_p(p_mean, 0.7, 5);
    // Serialize components individually then deserialize the whole event so
    // all #[serde(default)] fields are filled without requiring Default impl.
    let json = serde_json::json!({
        "calibration_id": TaskId::new().to_string(),
        "coefficients": serde_json::to_value(&cc).unwrap(),
        "coordination_threshold": serde_json::to_value(&ct).unwrap(),
        "ensemble": serde_json::to_value(&ensemble).unwrap(),
        "eigen": null,
        "timestamp": "2026-01-01T00:00:00Z"
    });
    serde_json::from_value(json).unwrap()
}

#[tokio::test]
async fn patch_ensemble_below_10_obs_returns_none() {
    let cal = Arc::new(RwLock::new(Some(make_calibration_event(0.75))));
    let result = patch_ensemble_p_from_oracle(&cal, 0.9, 9, 0.7, 5).await;
    assert!(result.is_none(), "n=9 < 10 → must return None");
}

#[tokio::test]
async fn patch_ensemble_no_calibration_returns_none() {
    let cal: Arc<RwLock<Option<CalibrationCompletedEvent>>> = Arc::new(RwLock::new(None));
    let result = patch_ensemble_p_from_oracle(&cal, 0.9, 15, 0.7, 5).await;
    assert!(result.is_none(), "calibration=None → must return None");
}

#[tokio::test]
async fn patch_ensemble_updates_p_mean() {
    let initial_p = 0.7;
    let cal = Arc::new(RwLock::new(Some(make_calibration_event(initial_p))));

    // pass_rate=0.95 clamped to [0.5,1.0] → p_mean should increase toward 0.95
    let result = patch_ensemble_p_from_oracle(&cal, 0.95, 15, 0.7, 5).await;
    assert!(result.is_some(), "n=15, calibration present → Some");
    let (p_before, p_after, _rho) = result.unwrap();
    assert!((p_before - initial_p).abs() < 1e-6, "p_before={p_before}");
    assert!(
        p_after > p_before,
        "higher pass_rate should increase p_mean: p_before={p_before} p_after={p_after}"
    );

    // verify calibration Arc was mutated
    let updated = cal.read().await;
    let new_p = updated.as_ref().unwrap().ensemble.as_ref().unwrap().p_mean;
    assert!((new_p - p_after).abs() < 1e-9, "Arc not updated: {new_p}");
}

#[tokio::test]
async fn patch_ensemble_pass_rate_clamped_to_half() {
    // pass_rate=0.1 should be clamped to 0.5 before computing new p_mean
    let cal = Arc::new(RwLock::new(Some(make_calibration_event(0.8))));
    let result = patch_ensemble_p_from_oracle(&cal, 0.1, 15, 0.7, 5).await;
    assert!(result.is_some());
    let (_p_before, p_after, _rho) = result.unwrap();
    // EnsembleCalibration::from_measured_p clamps pass_rate to [0.5,1.0],
    // so p_after must be ≥ 0.5
    assert!(p_after >= 0.5, "pass_rate clamped: p_after={p_after}");
}

#[tokio::test]
async fn patch_ensemble_no_ensemble_field_returns_none() {
    // Calibration exists but ensemble is None → patch returns None.
    use chrono::Utc;
    use h2ai_types::events::{CalibrationQuality, CalibrationSource, CgMode};
    let cc = CoherencyCoefficients::new(0.1, 0.01, vec![0.7]).unwrap();
    let ct = CoordinationThreshold::from_calibration(&cc, 1.0);
    let event_no_ensemble = CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients: cc,
        coordination_threshold: ct,
        ensemble: None, // deliberately absent
        eigen: None,
        timestamp: Utc::now(),
        pairwise_beta: None,
        cg_mode: CgMode::default(),
        adapter_families: vec![],
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: CalibrationQuality::default(),
        calibration_source: CalibrationSource::Measured,
        beta_quality: None,
    };
    let cal = Arc::new(RwLock::new(Some(event_no_ensemble)));
    let result = patch_ensemble_p_from_oracle(&cal, 0.9, 15, 0.7, 5).await;
    assert!(result.is_none(), "ensemble=None → must return None");
}
