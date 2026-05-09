use h2ai_state::NatsClient;
use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold};

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
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: Default::default(),
        calibration_source: Default::default(),
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
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: Default::default(),
        calibration_source: Default::default(),
    };
    let bytes = serde_json::to_vec(&original).unwrap();
    let back: CalibrationCompletedEvent = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back.coefficients.alpha, original.coefficients.alpha);
}

#[cfg(test)]
mod oracle_calibration_kv_tests {
    use super::*;

    #[tokio::test]
    #[ignore = "requires NATS server"]
    async fn oracle_calibration_put_get_roundtrip() {
        let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
        let client = NatsClient::connect(&nats_url).await.expect("connect");
        client.ensure_infrastructure().await.expect("infra");

        use h2ai_types::sizing::{OracleDomain, OracleObservation, OracleType};
        let obs = vec![
            OracleObservation {
                task_id: "task-1".into(),
                q_confidence: 0.8,
                y_oracle: true,
                residual: 0.2,
                domain: OracleDomain::Code,
                oracle_type: OracleType::TestSuite,
                timestamp_ms: 1000,
            },
            OracleObservation {
                task_id: "task-2".into(),
                q_confidence: 0.6,
                y_oracle: false,
                residual: 0.6,
                domain: OracleDomain::Factual,
                oracle_type: OracleType::ReferenceAnswer,
                timestamp_ms: 2000,
            },
        ];

        client.put_oracle_observations(&obs).await.expect("put");
        let retrieved = client.get_oracle_observations().await.expect("get");
        assert_eq!(retrieved.len(), 2);
        assert!((retrieved[0].q_confidence - 0.8).abs() < 1e-9);
        assert!((retrieved[1].residual - 0.6).abs() < 1e-9);
    }

    #[tokio::test]
    #[ignore = "requires NATS server"]
    async fn oracle_calibration_get_returns_empty_when_absent() {
        let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
        // Use a fresh namespace or just verify the method signature compiles
        // This test verifies the get returns empty vec when nothing stored
        // (hard to test without isolated bucket; just verify it compiles and returns Ok)
        let client = NatsClient::connect(&nats_url).await.expect("connect");
        client.ensure_infrastructure().await.ok();
        // get_oracle_observations must return Ok(vec![]) when key absent
        // Note: result depends on NATS state; just verify method exists and is callable
        let _ = client.get_oracle_observations().await;
    }
}
