use h2ai_autonomic::checker::MultiplicationChecker;
use h2ai_config::H2AIConfig;
use h2ai_types::identity::TaskId;
use h2ai_types::physics::{CoherencyCoefficients, CoordinationThreshold};

fn cc() -> CoherencyCoefficients {
    CoherencyCoefficients::new(0.1, 0.02, vec![0.8, 0.85, 0.9]).unwrap()
}

#[test]
fn checker_passes_when_all_conditions_hold() {
    let cc = cc();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let result = MultiplicationChecker::check(
        &TaskId::new(),
        &cc,
        &theta,
        0.7,
        0.85,
        0,
        &H2AIConfig::default(),
    );
    assert!(result.is_ok());
}

#[test]
fn checker_fails_when_baseline_competence_too_low() {
    let cc = cc();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let result = MultiplicationChecker::check(
        &TaskId::new(),
        &cc,
        &theta,
        0.3,
        0.85,
        0,
        &H2AIConfig::default(),
    );
    let err = result.unwrap_err();
    assert_eq!(err.retry_count, 0);
    assert!(matches!(
        err.failure,
        h2ai_types::physics::MultiplicationConditionFailure::InsufficientCompetence { .. }
    ));
}

#[test]
fn checker_fails_when_error_correlation_too_high() {
    let cc = cc();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let result = MultiplicationChecker::check(
        &TaskId::new(),
        &cc,
        &theta,
        0.7,
        0.95,
        0,
        &H2AIConfig::default(),
    );
    assert!(result.is_err());
}

#[test]
fn checker_embeds_retry_count_in_failure_event() {
    let cc = cc();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let result = MultiplicationChecker::check(
        &TaskId::new(),
        &cc,
        &theta,
        0.3,
        0.85,
        2,
        &H2AIConfig::default(),
    );
    assert_eq!(result.unwrap_err().retry_count, 2);
}

#[test]
fn checker_respects_custom_min_competence_threshold() {
    let cc = cc();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let mut cfg = H2AIConfig::default();
    cfg.min_baseline_competence = 0.8;
    let result = MultiplicationChecker::check(&TaskId::new(), &cc, &theta, 0.75, 0.85, 0, &cfg);
    assert!(result.is_err()); // 0.75 < 0.8
}
