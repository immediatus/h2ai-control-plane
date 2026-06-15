use h2ai_autonomic::epistemic::detect_frozen_verifier;
use h2ai_config::VerifierFreezeConfig;

fn default_cfg() -> VerifierFreezeConfig {
    VerifierFreezeConfig::default()
}

fn flat_scores(v: f64, n: usize) -> Vec<f64> {
    vec![v; n]
}

fn improving_scores() -> Vec<f64> {
    vec![0.25, 0.5, 0.75, 1.0]
}

fn frozen_reasons(n: usize) -> Vec<String> {
    (0..n)
        .map(|_| "idempotency key missing tenant_id prefix cross tenant collision".to_string())
        .collect()
}

fn divergent_reasons() -> Vec<String> {
    vec![
        "missing tenant prefix in redis key".to_string(),
        "kafka consumer group not idempotent".to_string(),
        "ClickHouse TTL partition missing".to_string(),
    ]
}

#[test]
fn frozen_signal_fires_when_all_five_conditions_met() {
    let cfg = default_cfg();
    let wave_scores = flat_scores(0.0, 4);
    let reasons = frozen_reasons(4);
    let improving = improving_scores();
    let other: Vec<&[f64]> = vec![&improving];
    let result = detect_frozen_verifier("C-008", &wave_scores, &reasons, &other, &cfg);
    assert!(result.is_some(), "should fire when all conditions met");
    let sig = result.unwrap();
    assert_eq!(sig.constraint_id, "C-008");
}

#[test]
fn does_not_fire_when_all_constraints_stuck() {
    let cfg = default_cfg();
    let wave_scores = flat_scores(0.0, 4);
    let reasons = frozen_reasons(4);
    let other_flat = flat_scores(0.0, 4);
    let other: Vec<&[f64]> = vec![&other_flat];
    let result = detect_frozen_verifier("C-008", &wave_scores, &reasons, &other, &cfg);
    assert!(
        result.is_none(),
        "should not fire when all constraints stuck"
    );
}

#[test]
fn does_not_fire_when_reasons_diverge() {
    let cfg = default_cfg();
    let wave_scores = flat_scores(0.0, 4);
    let reasons = divergent_reasons();
    let improving = improving_scores();
    let other: Vec<&[f64]> = vec![&improving];
    let result = detect_frozen_verifier("C-008", &wave_scores, &reasons, &other, &cfg);
    assert!(
        result.is_none(),
        "should not fire when reasons diverge (low Jaccard)"
    );
}

#[test]
fn does_not_fire_below_min_waves() {
    let mut cfg = default_cfg();
    cfg.min_waves_to_detect = 4;
    let wave_scores = flat_scores(0.0, 3);
    let reasons = frozen_reasons(3);
    let improving = improving_scores();
    let other: Vec<&[f64]> = vec![&improving];
    let result = detect_frozen_verifier("C-008", &wave_scores, &reasons, &other, &cfg);
    assert!(
        result.is_none(),
        "should not fire with fewer than min_waves_to_detect entries"
    );
}

#[test]
fn fires_at_exactly_min_waves() {
    let mut cfg = default_cfg();
    cfg.min_waves_to_detect = 3;
    let wave_scores = flat_scores(0.0, 3);
    let reasons = frozen_reasons(3);
    let improving = improving_scores();
    let other: Vec<&[f64]> = vec![&improving];
    let result = detect_frozen_verifier("C-008", &wave_scores, &reasons, &other, &cfg);
    assert!(
        result.is_some(),
        "should fire at exactly min_waves_to_detect entries"
    );
}

#[test]
fn does_not_fire_with_empty_other_constraint_trends() {
    let cfg = default_cfg();
    let wave_scores = flat_scores(0.0, 4);
    let reasons = frozen_reasons(4);
    let other: Vec<&[f64]> = vec![];
    let result = detect_frozen_verifier("C-008", &wave_scores, &reasons, &other, &cfg);
    assert!(
        result.is_none(),
        "should not fire when other_constraint_trends is empty"
    );
}

#[test]
fn does_not_fire_when_score_has_variance() {
    let cfg = default_cfg();
    let wave_scores = vec![0.1, 0.1, 0.2];
    let reasons = frozen_reasons(3);
    let improving = improving_scores();
    let other: Vec<&[f64]> = vec![&improving];
    let result = detect_frozen_verifier("C-008", &wave_scores, &reasons, &other, &cfg);
    assert!(
        result.is_none(),
        "should not fire when score shows meaningful variance"
    );
}

#[test]
fn does_not_fire_when_other_constraints_below_success_threshold() {
    let mut cfg = default_cfg();
    cfg.other_constraint_success_threshold = 0.5;
    let wave_scores = flat_scores(0.0, 4);
    let reasons = frozen_reasons(4);
    let weak_improving = vec![0.1_f64, 0.2, 0.2, 0.3];
    let other: Vec<&[f64]> = vec![&weak_improving];
    let result = detect_frozen_verifier("C-008", &wave_scores, &reasons, &other, &cfg);
    assert!(
        result.is_none(),
        "should not fire when other constraints below success threshold"
    );
}
