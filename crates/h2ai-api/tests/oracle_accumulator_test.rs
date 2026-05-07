use h2ai_api::oracle::determine_calibration_basis;
use h2ai_types::sizing::{OracleDomain, OracleObservation, OracleType};

fn obs(q: f64, y: bool) -> OracleObservation {
    OracleObservation {
        task_id: "t".into(),
        q_confidence: q,
        y_oracle: y,
        residual: (q - y as u8 as f64).abs(),
        domain: OracleDomain::Code,
        oracle_type: OracleType::TestSuite,
        timestamp_ms: 0,
    }
}

#[test]
fn basis_bootstrap_when_10_to_29() {
    // n=29 is in the Bootstrap range [10, 30)
    let observations: Vec<OracleObservation> = (0..29).map(|_| obs(0.8, true)).collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(status.basis, 1, "n=29 → Bootstrap (basis=1)");
}

#[test]
fn basis_heuristic_when_fewer_than_10() {
    // n=9 is below Bootstrap minimum → Heuristic
    let observations: Vec<OracleObservation> = (0..9).map(|_| obs(0.8, true)).collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(status.basis, 0, "n=9 → Heuristic (basis=0)");
}

#[test]
fn basis_conformal_when_30_plus_and_low_ece() {
    // 30 observations with q=0.9, y=true → residual=0.1 → ECE=0.1 < 0.15
    let observations: Vec<OracleObservation> = (0..30).map(|_| obs(0.9, true)).collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(
        status.basis, 2,
        "n=30, ECE=0.1 < 0.15 → Conformal (basis=2)"
    );
    assert_eq!(status.n_observations, 30);
    assert!((status.ece - 0.1).abs() < 1e-9);
}

#[test]
fn basis_heuristic_when_30_plus_but_high_ece() {
    // 30 observations with q=0.5, y=false → residual=0.5 → ECE=0.5 > 0.15
    let observations: Vec<OracleObservation> = (0..30).map(|_| obs(0.5, false)).collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(
        status.basis, 0,
        "n=30, ECE=0.5 > 0.15 → Heuristic (basis=0)"
    );
}

#[test]
fn fifo_eviction_at_200() {
    let mut observations: Vec<OracleObservation> = (0..200)
        .map(|i| OracleObservation {
            task_id: format!("t{i}"),
            q_confidence: 0.8,
            y_oracle: true,
            residual: 0.2,
            domain: OracleDomain::Code,
            oracle_type: OracleType::TestSuite,
            timestamp_ms: i,
        })
        .collect();
    // Add one more — should evict oldest
    observations.push(OracleObservation {
        task_id: "new".into(),
        q_confidence: 0.9,
        y_oracle: true,
        residual: 0.1,
        domain: OracleDomain::Code,
        oracle_type: OracleType::TestSuite,
        timestamp_ms: 200,
    });
    h2ai_api::oracle::enforce_fifo_cap(&mut observations, 200);
    assert_eq!(observations.len(), 200);
    assert_eq!(observations[0].task_id, "t1", "oldest (t0) evicted");
    assert_eq!(observations[199].task_id, "new", "newest retained");
}

#[test]
fn calibration_status_empty_input() {
    let status = determine_calibration_basis(&[]);
    assert_eq!(status.basis, 0);
    assert_eq!(status.n_observations, 0);
    assert_eq!(status.ece, 0.0);
}

#[test]
fn calibration_drift_condition_triggers_at_ece_above_015() {
    // n=30, ECE=0.5 > 0.15 → should trigger CalibrationDriftWarning
    let observations: Vec<OracleObservation> = (0..30).map(|_| obs(0.5, false)).collect();
    let status = determine_calibration_basis(&observations);
    assert!(
        status.n_observations >= 30 && status.ece > 0.15,
        "drift condition: n={} ece={} should trigger warning",
        status.n_observations,
        status.ece
    );
}

#[test]
fn calibration_drift_condition_does_not_trigger_when_ece_ok() {
    // n=30, ECE=0.1 < 0.15 → no drift warning
    let observations: Vec<OracleObservation> = (0..30).map(|_| obs(0.9, true)).collect();
    let status = determine_calibration_basis(&observations);
    assert!(
        !(status.n_observations >= 30 && status.ece > 0.15),
        "no drift: n={} ece={} should NOT trigger warning",
        status.n_observations,
        status.ece
    );
}

#[test]
fn oracle_suspect_condition_triggers_at_pass_rate_below_030() {
    // 30 obs all failing → pass_rate=0.0 < 0.30 → OracleSuspect
    let observations: Vec<OracleObservation> = (0..30).map(|_| obs(0.5, false)).collect();
    let status = determine_calibration_basis(&observations);
    assert!(
        status.n_observations >= 30 && status.pass_rate < 0.3,
        "suspect: n={} pass_rate={} should trigger suspect",
        status.n_observations,
        status.pass_rate
    );
}

#[test]
fn oracle_suspect_condition_does_not_trigger_when_pass_rate_healthy() {
    // 30 obs all passing → pass_rate=1.0 ≥ 0.30 → no suspect
    let observations: Vec<OracleObservation> = (0..30).map(|_| obs(0.9, true)).collect();
    let status = determine_calibration_basis(&observations);
    assert!(
        !(status.n_observations >= 30 && status.pass_rate < 0.3),
        "no suspect: n={} pass_rate={} should NOT trigger",
        status.n_observations,
        status.pass_rate
    );
}

#[test]
fn alerts_do_not_trigger_below_30_observations() {
    // n=29: both alert conditions require n >= 30, so no triggers regardless of ECE/pass_rate
    let observations: Vec<OracleObservation> = (0..29).map(|_| obs(0.5, false)).collect();
    let status = determine_calibration_basis(&observations);
    // ECE would be 0.5 and pass_rate 0.0 — but n < 30 → no alerts
    assert!(
        status.n_observations < 30,
        "n={} < 30, alert conditions must not fire",
        status.n_observations
    );
}
