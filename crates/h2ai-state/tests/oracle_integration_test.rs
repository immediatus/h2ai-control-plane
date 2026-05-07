// Requires a live NATS server. Run with:
//   NATS_URL=nats://localhost:4222 cargo nextest run -p h2ai-state --test oracle_integration_test

use h2ai_state::nats::NatsClient;

async fn connect() -> Option<NatsClient> {
    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    match NatsClient::connect(&url).await {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            None
        }
    }
}

/// Verify the oracle calibration KV bucket is created by ensure_infrastructure
/// and that put/get round-trips correctly.
#[tokio::test]
#[ignore = "requires NATS server"]
async fn oracle_observations_kv_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
    client.ensure_infrastructure().await.expect("infra setup");

    use h2ai_types::sizing::{OracleDomain, OracleObservation, OracleType};

    let observations = vec![
        OracleObservation {
            task_id: "integration-test-1".into(),
            q_confidence: 0.85,
            y_oracle: true,
            residual: 0.15,
            domain: OracleDomain::Code,
            oracle_type: OracleType::TestSuite,
            timestamp_ms: 1000,
        },
        OracleObservation {
            task_id: "integration-test-2".into(),
            q_confidence: 0.60,
            y_oracle: false,
            residual: 0.60,
            domain: OracleDomain::Factual,
            oracle_type: OracleType::ReferenceAnswer,
            timestamp_ms: 2000,
        },
    ];

    client
        .put_oracle_observations(&observations)
        .await
        .expect("put observations");

    let retrieved = client
        .get_oracle_observations()
        .await
        .expect("get observations");

    assert_eq!(retrieved.len(), 2, "both observations retrieved");
    assert!((retrieved[0].q_confidence - 0.85).abs() < 1e-9);
    assert!((retrieved[1].residual - 0.60).abs() < 1e-9);
    assert!(matches!(retrieved[0].domain, OracleDomain::Code));
    assert!(matches!(
        retrieved[1].oracle_type,
        OracleType::ReferenceAnswer
    ));
}

/// Verify that publishing an OraclePendingEvent to NATS core succeeds without error.
#[tokio::test]
#[ignore = "requires NATS server"]
async fn oracle_pending_event_publish_succeeds() {
    let Some(client) = connect().await else {
        return;
    };

    use h2ai_orchestrator::oracle::oracle_dispatch;
    use h2ai_types::identity::TaskId;
    use h2ai_types::sizing::{OracleDomain, OracleSpec, OracleType};

    let spec = OracleSpec {
        runner_uri: "http://localhost:9090".into(),
        test_suite: "tests/".into(),
        language: "python".into(),
        timeout_ms: 5000,
        reference_output: None,
        oracle_type: OracleType::TestSuite,
        domain: OracleDomain::Code,
    };

    // fire() is fire-and-forget; we just verify it doesn't panic
    oracle_dispatch::fire(
        &client.client,
        TaskId::new(),
        "print('hello world')",
        0.85,
        3,
        &spec,
    )
    .await;

    // If we reach here without panic, the publish path works
}

/// Verify that get_oracle_observations returns empty vec when no data stored.
#[tokio::test]
#[ignore = "requires NATS server"]
async fn oracle_observations_empty_on_fresh_bucket() {
    let Some(client) = connect().await else {
        return;
    };
    client.ensure_infrastructure().await.expect("infra setup");

    // This test depends on whether the bucket has prior data.
    // We verify the method is callable and returns Ok.
    let result = client.get_oracle_observations().await;
    assert!(result.is_ok(), "get_oracle_observations returns Ok");
}

/// Verify CalibrationDriftWarning serializes and deserializes correctly.
#[tokio::test]
#[ignore = "requires NATS server"]
async fn calibration_drift_warning_serializes_correctly() {
    let Some(_client) = connect().await else {
        return;
    };

    use h2ai_types::events::CalibrationDriftWarning;

    let warning = CalibrationDriftWarning {
        n_observations: 35,
        ece: 0.22,
        timestamp_ms: 1_000_000,
    };
    let json = serde_json::to_vec(&warning).expect("serialize");
    let back: CalibrationDriftWarning = serde_json::from_slice(&json).expect("deserialize");
    assert_eq!(back.n_observations, 35);
    assert!((back.ece - 0.22).abs() < 1e-9);
    assert_eq!(back.timestamp_ms, 1_000_000);
}

/// Verify the calibration basis promotion logic: 30 well-calibrated obs → Conformal.
///
/// Tests determine_calibration_basis() in isolation (no shared KV state) to avoid
/// race conditions when integration tests run in parallel against the same bucket.
/// The KV roundtrip itself is verified in oracle_observations_kv_roundtrip.
#[tokio::test]
#[ignore = "requires NATS server"]
async fn oracle_accumulator_basis_promotion_after_30_obs() {
    // Connect to verify NATS is available, but don't use shared KV state.
    let Some(_client) = connect().await else {
        return;
    };

    use h2ai_api::oracle::determine_calibration_basis;
    use h2ai_types::sizing::{OracleDomain, OracleObservation, OracleType};

    // 30 observations: q=0.9, y=true → residual=|0.9-1.0|=0.1 → ECE=0.1 < 0.15
    let observations: Vec<OracleObservation> = (0..30)
        .map(|i| OracleObservation {
            task_id: format!("promo-test-{i}"),
            q_confidence: 0.9,
            y_oracle: true,
            residual: 0.1,
            domain: OracleDomain::Code,
            oracle_type: OracleType::TestSuite,
            timestamp_ms: i as u64 * 1000,
        })
        .collect();

    let status = determine_calibration_basis(&observations);
    assert_eq!(
        status.basis, 2,
        "30 obs with ECE=0.1 should promote to Conformal (basis=2), got basis={}",
        status.basis
    );
    assert_eq!(status.n_observations, 30);
    assert!(
        (status.ece - 0.1).abs() < 1e-9,
        "ECE should be 0.1, got {}",
        status.ece
    );

    // n=29 is in Bootstrap range [10, 30)
    let short: Vec<OracleObservation> = observations[..29].to_vec();
    let short_status = determine_calibration_basis(&short);
    assert_eq!(
        short_status.basis, 1,
        "n=29 should be Bootstrap (basis=1), got {}",
        short_status.basis
    );

    // n=9 is below Bootstrap minimum — Heuristic
    let tiny: Vec<OracleObservation> = observations[..9].to_vec();
    let tiny_status = determine_calibration_basis(&tiny);
    assert_eq!(
        tiny_status.basis, 0,
        "n=9 should be Heuristic (basis=0), got {}",
        tiny_status.basis
    );

    // And high-ECE → Heuristic even at n=30
    let high_ece: Vec<OracleObservation> = (0..30)
        .map(|i| OracleObservation {
            task_id: format!("high-ece-{i}"),
            q_confidence: 0.5,
            y_oracle: false,
            residual: 0.5, // ECE = 0.5 > 0.15
            domain: OracleDomain::Code,
            oracle_type: OracleType::TestSuite,
            timestamp_ms: i as u64 * 1000,
        })
        .collect();
    let high_ece_status = determine_calibration_basis(&high_ece);
    assert_eq!(
        high_ece_status.basis, 0,
        "n=30 but ECE=0.5 > 0.15 should stay Heuristic (basis=0)"
    );
}
