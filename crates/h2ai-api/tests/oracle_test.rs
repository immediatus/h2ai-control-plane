use h2ai_api::oracle::{
    determine_calibration_basis, ece_from_observations, pass_rate_from_observations, residual_p90,
};
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
    // 10 residuals: 0.1, 0.2, ..., 1.0 → P90 = index 8 (0-based) = 0.9
    let mut observations: Vec<OracleObservation> = (1..=10)
        .map(|i| {
            let r = i as f64 * 0.1;
            OracleObservation {
                task_id: "t".into(),
                q_confidence: 0.5,
                y_oracle: false,
                residual: r,
                domain: OracleDomain::Code,
                oracle_type: OracleType::TestSuite,
                timestamp_ms: i,
            }
        })
        .collect();
    // shuffle to verify sorting
    observations.reverse();
    let p90 = residual_p90(&observations);
    assert!(
        (p90 - 0.9).abs() < 1e-9,
        "P90 of 0.1..1.0 should be 0.9, got {p90}"
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
            oracle_type: OracleType::TestSuite,
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
            oracle_type: OracleType::TestSuite,
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
            oracle_type: OracleType::TestSuite,
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
            oracle_type: OracleType::TestSuite,
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
            oracle_type: OracleType::TestSuite,
            timestamp_ms: i as u64,
        })
        .collect();
    let status = determine_calibration_basis(&observations);
    assert_eq!(
        status.basis, 0,
        "n=30 ECE=0.5 → Heuristic (basis=0, quality regression)"
    );
}
