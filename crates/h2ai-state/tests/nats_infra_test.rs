// Requires a live NATS server. Run with:
//   NATS_URL=nats://localhost:4222 cargo nextest run -p h2ai-state --test nats_infra_test

use h2ai_state::nats::NatsClient;

#[tokio::test]
#[ignore]
async fn ensure_infrastructure_is_idempotent() {
    let url =
        std::env::var("NATS_URL").unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
    let client = match NatsClient::connect(&url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            return;
        }
    };
    client.ensure_infrastructure().await.expect("first call");
    client
        .ensure_infrastructure()
        .await
        .expect("second call must be idempotent");
}

#[tokio::test]
#[ignore]
async fn put_and_get_calibration_roundtrip() {
    use chrono::Utc;
    use h2ai_types::events::CalibrationCompletedEvent;
    use h2ai_types::identity::TaskId;
    use h2ai_types::physics::{CoherencyCoefficients, CoordinationThreshold};

    let url =
        std::env::var("NATS_URL").unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
    let client = match NatsClient::connect(&url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            return;
        }
    };
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
        cg_mode: Default::default(),
        adapter_families: Vec::new(),
        explorer_verification_family_match: false,
        single_family_warning: false,
    };

    client.put_calibration(&event).await.expect("put");
    let back = client.get_calibration().await.expect("get");
    assert!(back.is_some());
    assert_eq!(back.unwrap().coefficients.alpha, 0.12);
}
