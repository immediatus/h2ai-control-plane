use h2ai_autonomic::retry_accumulator::RetryAccumulator;
use h2ai_types::events::ConstraintViolation;

fn v(id: &str) -> ConstraintViolation {
    ConstraintViolation {
        constraint_id: id.to_string(),
        score: 0.0,
        severity_label: "Hard".to_string(),
        remediation_hint: Some(format!("Fix {id}")),
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
    }
}

#[test]
fn new_is_empty() {
    assert!(RetryAccumulator::new().rates().is_empty());
}

#[test]
fn update_first_wave_applies_decay() {
    let mut acc = RetryAccumulator::new();
    // n_f=2, both violations are c-001 → new_rate = 1.0
    // accumulated = 0.7*0 + 0.3*1.0 = 0.3
    acc.update(&[v("c-001"), v("c-001")], 2, 0.7);
    let rate = *acc.rates().get("c-001").unwrap();
    assert!((rate - 0.3).abs() < 1e-9, "expected 0.3, got {rate}");
}

#[test]
fn update_accumulates_across_waves() {
    let mut acc = RetryAccumulator::new();
    // Three waves at rate 1.0:
    // after wave 1: 0.3
    // after wave 2: 0.7*0.3 + 0.3 = 0.51
    // after wave 3: 0.7*0.51 + 0.3 = 0.657
    for _ in 0..3 {
        acc.update(&[v("c-001"), v("c-001")], 2, 0.7);
    }
    let rate = *acc.rates().get("c-001").unwrap();
    assert!(
        rate > 0.6 && rate < 0.7,
        "after 3 waves at rate 1.0: {rate}"
    );
}

#[test]
fn update_converges_to_empirical_rate() {
    let mut acc = RetryAccumulator::new();
    // 5 violations out of n_f=10 → new_rate = 0.5
    let viols: Vec<ConstraintViolation> = (0..5).map(|_| v("c-001")).collect();
    for _ in 0..25 {
        acc.update(&viols, 10, 0.7);
    }
    let rate = *acc.rates().get("c-001").unwrap();
    assert!(
        (rate - 0.5).abs() < 0.02,
        "should converge to 0.5, got {rate}"
    );
}

#[test]
fn reset_clears_all() {
    let mut acc = RetryAccumulator::new();
    acc.update(&[v("c-001")], 1, 0.7);
    acc.reset();
    assert!(acc.rates().is_empty());
}

#[test]
fn two_accumulators_independent() {
    let mut a = RetryAccumulator::new();
    let b = RetryAccumulator::new();
    a.update(&[v("c-001")], 1, 0.7);
    assert!(b.rates().is_empty(), "separate instances share no state");
}

#[test]
fn update_nf_zero_is_noop() {
    let mut acc = RetryAccumulator::new();
    acc.update(&[v("c-001")], 0, 0.7);
    assert!(acc.rates().is_empty(), "n_f=0 must be a no-op");
}
