use h2ai_autonomic::repair::{assess_gap_quality, GapQualityVerdict};
use h2ai_config::GapQualityConfig;
use h2ai_types::gap_i1::DomainSynthesis;

fn ds(pre: f64, post: Vec<f64>, injected_at: u32) -> DomainSynthesis {
    DomainSynthesis {
        check_id: ("C-008".to_string(), 2),
        incorrect_pattern: "wrong".to_string(),
        correct_pattern: "right".to_string(),
        mechanistic_reason: "reason".to_string(),
        source: None,
        confidence: 0.5,
        injected_at_wave: Some(injected_at),
        pre_injection_pass_rate: Some(pre),
        post_injection_pass_rates: post,
    }
}

fn cfg() -> GapQualityConfig {
    GapQualityConfig::default() // min_improvement = 0.1, min_post_injection_waves = 2
}

#[test]
fn pending_when_fewer_than_min_post_injection_waves() {
    let d = ds(0.3, vec![0.5], 1); // only 1 post-injection wave, need 2
    assert!(matches!(
        assess_gap_quality(&d, &cfg()),
        GapQualityVerdict::Pending
    ));
}

#[test]
fn pending_when_no_injected_at_wave() {
    let mut d = ds(0.3, vec![0.5, 0.6], 1);
    d.injected_at_wave = None;
    assert!(matches!(
        assess_gap_quality(&d, &cfg()),
        GapQualityVerdict::Pending
    ));
}

#[test]
fn effective_when_improvement_meets_threshold() {
    let d = ds(0.3, vec![0.4, 0.5], 1); // improvement = 0.2 >= 0.1
    assert!(matches!(
        assess_gap_quality(&d, &cfg()),
        GapQualityVerdict::Effective
    ));
}

#[test]
fn ineffective_when_improvement_below_threshold_after_sufficient_waves() {
    let d = ds(0.3, vec![0.32, 0.31], 1); // improvement = 0.01 < 0.1
    assert!(matches!(
        assess_gap_quality(&d, &cfg()),
        GapQualityVerdict::Ineffective
    ));
}

#[test]
fn ineffective_verdict_used_for_eviction() {
    let d = ds(0.3, vec![0.3, 0.3], 1); // 0.0 delta
    let v = assess_gap_quality(&d, &cfg());
    assert!(matches!(v, GapQualityVerdict::Ineffective));
}
