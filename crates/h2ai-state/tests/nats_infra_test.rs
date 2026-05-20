// Requires a live NATS server. Run with:
//   NATS_URL=nats://localhost:4222 cargo nextest run -p h2ai-state --test nats_infra_test

use h2ai_state::nats::NatsClient;
use h2ai_types::identity::TenantId;

async fn make_client() -> Option<NatsClient> {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    match NatsClient::connect(&url).await {
        Ok(c) => {
            c.ensure_infrastructure().await.ok()?;
            Some(c)
        }
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            None
        }
    }
}

#[tokio::test]
#[ignore]
async fn ensure_infrastructure_is_idempotent() {
    let url = h2ai_config::H2AIConfig::default().nats_url;
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
    use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold};

    let url = h2ai_config::H2AIConfig::default().nats_url;
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
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: Default::default(),
        calibration_source: Default::default(),
        beta_quality: None,
    };

    client.put_calibration(&event).await.expect("put");
    let back = client.get_calibration().await.expect("get");
    assert!(back.is_some());
    assert_eq!(back.unwrap().coefficients.alpha, 0.12);
}

#[tokio::test]
#[ignore]
async fn put_and_get_tao_estimator_roundtrip() {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    let client = match NatsClient::connect(&url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            return;
        }
    };
    client.ensure_infrastructure().await.expect("infra");

    let tid = TenantId::default_tenant();
    client
        .put_tao_estimator_state(&tid, 0.75, 25)
        .await
        .expect("put");
    let back = client.get_tao_estimator_state(&tid).await.expect("get");
    assert!(back.is_some());
    let (ema, count) = back.unwrap();
    assert_eq!(ema, 0.75);
    assert_eq!(count, 25);
}

#[tokio::test]
#[ignore = "requires NATS"]
async fn signal_round_trips_via_jetstream() {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    let client = match NatsClient::connect(&url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            return;
        }
    };
    client.provision_signals_stream().await.unwrap();

    let task_id = h2ai_types::identity::TaskId::from_uuid(uuid::Uuid::new_v4());
    let tenant_id = h2ai_types::identity::TenantId::from("test");

    let signal = h2ai_types::signal::ResumeSignal {
        task_id: task_id.clone(),
        tenant_id: tenant_id.clone(),
        payload: h2ai_types::signal::SignalPayload::Approve(h2ai_types::signal::ApproveSignal {
            approved: true,
            reviewer_note: None,
            operator_id: "test-operator".into(),
        }),
        timeout_at_ms: u64::MAX,
        issued_at_ms: 0,
    };

    client.publish_signal(&signal).await.unwrap();

    let mut sub = client
        .subscribe_signals(&task_id, &tenant_id)
        .await
        .unwrap();
    let received = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        futures::StreamExt::next(&mut sub),
    )
    .await
    .unwrap()
    .unwrap()
    .unwrap();

    assert!(matches!(
        received.payload,
        h2ai_types::signal::SignalPayload::Approve(_)
    ));

    client.delete_signal_consumer(&task_id).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn calibration_record_roundtrip() {
    use h2ai_types::calibration::{AuditorCircuitState, CalibrationRecord, ProbeSource};
    let Some(client) = make_client().await else {
        return;
    };
    let record = CalibrationRecord {
        adapter_profile: "test-profile".to_string(),
        constraint_id: None,
        alpha: 0.12,
        alpha_measured: 0.10,
        beta_0: 0.039,
        k: 2.0,
        n_useful_history: vec![(2, 3, 1000)],
        probe_source: ProbeSource::Same,
        fingerprint: None,
        circuit_state: AuditorCircuitState::Closed,
    };
    client.put_calibration_record(&record).await.unwrap();
    let got = client
        .get_calibration_record("test-profile")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(got, record);
}

#[tokio::test]
#[ignore]
async fn auditor_health_roundtrip() {
    use h2ai_types::calibration::{AuditorCircuitState, AuditorHealth};
    let Some(client) = make_client().await else {
        return;
    };
    let health = AuditorHealth {
        state: AuditorCircuitState::Open,
        last_probe_cg: 0.0,
        tripped_at: Some(1_700_000_000_000),
        recovery_probe_count: 3,
    };
    client
        .put_auditor_health("test-profile", &health)
        .await
        .unwrap();
    let got = client
        .get_auditor_health("test-profile")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(got, health);
}

#[tokio::test]
#[ignore]
async fn probe_lease_cas_exclusive() {
    let Some(client) = make_client().await else {
        return;
    };
    let profile = "cas-test-profile";
    // Release any stale lease first
    client.release_probe_lease(profile).await.unwrap();
    // First acquire should succeed
    let won = client.acquire_probe_lease(profile, 60).await.unwrap();
    assert!(won, "first acquire should win");
    // Second acquire (same profile, within TTL) should fail
    let won2 = client.acquire_probe_lease(profile, 60).await.unwrap();
    assert!(!won2, "second acquire should lose");
    // Release, then third acquire should succeed
    client.release_probe_lease(profile).await.unwrap();
    let won3 = client.acquire_probe_lease(profile, 60).await.unwrap();
    assert!(won3, "acquire after release should win");
    // Cleanup
    client.release_probe_lease(profile).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn get_calibration_record_missing_returns_none() {
    let Some(client) = make_client().await else {
        return;
    };
    let got = client
        .get_calibration_record("nonexistent-profile-xyz")
        .await
        .unwrap();
    assert!(got.is_none());
}
