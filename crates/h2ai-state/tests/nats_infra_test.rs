// Requires a live NATS server. Run with:
//   NATS_URL=nats://localhost:4222 cargo nextest run -p h2ai-state --test nats_infra_test

use h2ai_state::nats::NatsClient;

#[tokio::test]
async fn ensure_infrastructure_is_idempotent() {
    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    let client = NatsClient::connect(&url).await.expect("connect");
    client.ensure_infrastructure().await.expect("first call");
    client
        .ensure_infrastructure()
        .await
        .expect("second call must be idempotent");
}

#[tokio::test]
async fn put_and_get_calibration_roundtrip() {
    use chrono::Utc;
    use h2ai_types::events::CalibrationCompletedEvent;
    use h2ai_types::identity::TaskId;
    use h2ai_types::physics::{CoherencyCoefficients, CoordinationThreshold};

    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    let client = NatsClient::connect(&url).await.expect("connect");
    client.ensure_infrastructure().await.expect("infra");

    let event = CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients: CoherencyCoefficients::new(0.12, 0.02, vec![0.8, 0.75]).unwrap(),
        coordination_threshold: CoordinationThreshold::from_calibration(
            &CoherencyCoefficients::new(0.12, 0.02, vec![0.8, 0.75]).unwrap(),
            0.3,
        ),
        ensemble: None,
        eigen: None,
        timestamp: Utc::now(),
        pairwise_beta: None,
    };

    client.put_calibration(&event).await.expect("put");
    let back = client.get_calibration().await.expect("get");
    assert!(back.is_some());
    assert_eq!(back.unwrap().coefficients.alpha, 0.12);
}
