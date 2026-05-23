use h2ai_autonomic::drift::BocpdDetector;
use h2ai_autonomic::drift::DdmDetector;
use h2ai_autonomic::drift::{DriftEvent, DriftMonitor};
use h2ai_config::H2AIConfig;

fn make_monitor() -> DriftMonitor {
    let cfg = H2AIConfig::default();
    DriftMonitor::from_config(&cfg)
}

#[test]
fn drift_monitor_no_events_on_stable_stream() {
    let mut dm = make_monitor();
    for _ in 0..50 {
        let events = dm.observe(0.85);
        assert!(events.is_empty(), "stable stream must produce no events");
    }
}

#[test]
fn drift_monitor_warning_before_changepoint_on_injected_drift() {
    let mut dm = DriftMonitor::new(
        DdmDetector::new(10, 2.5),
        BocpdDetector::new(0.05, 0.90),
        0.05,
        false,
        3600,
    );
    for _ in 0..20 {
        dm.observe(0.85);
    }
    let mut saw_warning = false;
    let mut saw_changepoint = false;
    for _ in 0..60 {
        for event in dm.observe(0.10) {
            match event {
                DriftEvent::Warning(_) => saw_warning = true,
                DriftEvent::Changepoint(_) => saw_changepoint = true,
            }
        }
    }
    assert!(saw_warning, "DDM must fire a warning under injected drift");
    assert!(
        saw_changepoint,
        "BOCPD must fire a changepoint under injected drift"
    );
}

#[test]
fn drift_monitor_conformal_margin_active_after_changepoint() {
    let mut dm = DriftMonitor::new(
        DdmDetector::new(5, 2.5),
        BocpdDetector::new(0.1, 0.90),
        0.05,
        false,
        3600,
    );
    for _ in 0..5 {
        dm.observe(0.85);
    }
    for _ in 0..50 {
        dm.observe(0.05);
    }
    assert!(
        dm.active_conformal_margin() > 0.0,
        "conformal margin must be positive after changepoint"
    );
    assert!(
        (dm.active_conformal_margin() - 0.05).abs() < 1e-9,
        "conformal margin must equal drift_conformal_margin config"
    );
}

#[test]
fn drift_monitor_conformal_margin_zero_before_changepoint() {
    let dm = make_monitor();
    assert_eq!(
        dm.active_conformal_margin(),
        0.0,
        "no margin before any changepoint"
    );
}

#[test]
fn drift_monitor_reset_clears_changepoint_state() {
    let mut dm = DriftMonitor::new(
        DdmDetector::new(5, 2.5),
        BocpdDetector::new(0.1, 0.90),
        0.05,
        false,
        3600,
    );
    for _ in 0..5 {
        dm.observe(0.85);
    }
    for _ in 0..50 {
        dm.observe(0.05);
    }
    assert!(dm.active_conformal_margin() > 0.0);
    dm.reset_after_recalibration();
    assert_eq!(
        dm.active_conformal_margin(),
        0.0,
        "margin must be zero after reset"
    );
}

#[test]
fn ddm_does_not_fire_during_warmup() {
    let mut ddm = DdmDetector::new(20, 2.5);
    for _ in 0..19 {
        assert!(
            ddm.observe(0.8).is_none(),
            "must not fire before reference is established"
        );
    }
}

#[test]
fn ddm_does_not_fire_on_stable_stream() {
    let mut ddm = DdmDetector::new(10, 2.5);
    for _ in 0..10 {
        ddm.observe(0.8);
    }
    for i in 0..20 {
        let result = ddm.observe(0.8 + (i as f64 * 0.001));
        assert!(result.is_none(), "stable stream must not fire DDM");
    }
}

#[test]
fn ddm_fires_on_large_shift() {
    let mut ddm = DdmDetector::new(10, 2.5);
    for _ in 0..10 {
        ddm.observe(0.8);
    }
    for _ in 0..9 {
        ddm.observe(0.2);
    }
    let warning = ddm.observe(0.2);
    assert!(
        warning.is_some(),
        "shift from 0.8 to 0.2 must fire DDM warning"
    );
    let w = warning.unwrap();
    assert!((w.reference_mean - 0.8).abs() < 1e-6);
    assert!(w.deviation_sigmas > 2.5);
    assert_eq!(w.metric, "consensus_agreement_rate");
}

#[test]
fn ddm_reset_clears_reference() {
    let mut ddm = DdmDetector::new(5, 2.5);
    for _ in 0..5 {
        ddm.observe(0.8);
    }
    ddm.reset();
    for _ in 0..5 {
        ddm.observe(0.2);
    }
    for _ in 0..5 {
        let result = ddm.observe(0.21);
        assert!(result.is_none(), "post-reset stable stream must not fire");
    }
}

#[test]
fn bocpd_does_not_fire_on_stable_stream() {
    let mut bocpd = BocpdDetector::new(0.01, 0.90);
    for _ in 0..50 {
        let result = bocpd.observe(0.85);
        assert!(
            result.is_none(),
            "stable stream must not trigger BOCPD changepoint"
        );
    }
}

#[test]
fn bocpd_fires_after_abrupt_shift() {
    let mut bocpd = BocpdDetector::new(0.01, 0.90);
    for _ in 0..40 {
        bocpd.observe(0.85);
    }
    let mut fired = false;
    for _ in 0..30 {
        if bocpd.observe(0.20).is_some() {
            fired = true;
            break;
        }
    }
    assert!(
        fired,
        "BOCPD must detect abrupt shift from 0.85 to 0.20 within 30 observations"
    );
}

#[test]
fn bocpd_run_states_bounded_by_max() {
    let mut bocpd = BocpdDetector::new(0.01, 0.90);
    for i in 0..600 {
        bocpd.observe(0.5 + 0.001 * (i % 10) as f64);
    }
    assert!(
        bocpd.run_states_len() <= 500,
        "run_states must be capped at MAX_RUN_LENGTH=500"
    );
}
