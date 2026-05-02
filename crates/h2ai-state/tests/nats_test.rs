use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent};
use h2ai_types::identity::TaskId;
use h2ai_types::physics::{CoherencyCoefficients, CoordinationThreshold};

#[test]
fn h2ai_event_serialises_to_tagged_json() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let event = H2AIEvent::CalibrationCompleted(CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients: cc,
        coordination_threshold: theta,
        ensemble: None,
        eigen: None,
        timestamp: chrono::Utc::now(),
        pairwise_beta: None,
        cg_mode: Default::default(),
        adapter_families: Vec::new(),
        explorer_verification_family_match: false,
        single_family_warning: false,
    });

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"event_type\":\"CalibrationCompleted\""));
    assert!(json.contains("\"payload\""));
}

#[test]
fn calibration_event_roundtrip() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let original = CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients: cc,
        coordination_threshold: theta,
        ensemble: None,
        eigen: None,
        timestamp: chrono::Utc::now(),
        pairwise_beta: None,
        cg_mode: Default::default(),
        adapter_families: Vec::new(),
        explorer_verification_family_match: false,
        single_family_warning: false,
    };
    let bytes = serde_json::to_vec(&original).unwrap();
    let back: CalibrationCompletedEvent = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back.coefficients.alpha, original.coefficients.alpha);
}
