// Requires a live NATS server. Run with:
//   NATS_URL=nats://localhost:4222 cargo nextest run -p h2ai-state --test nats_infra_test

use h2ai_state::nats::NatsClient;
use h2ai_types::identity::TenantId;

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
#[ignore = "requires NATS server"]
async fn wiki_kv_roundtrip() {
    use h2ai_constraints::types::{ConstraintMeta, ConstraintSeverity, PredicateKind};
    use h2ai_constraints::wiki::WikiCache;

    let url = h2ai_config::H2AIConfig::default().nats_url;
    let nats = match NatsClient::connect(&url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable — skipping: {e}");
            return;
        }
    };
    nats.ensure_infrastructure().await.expect("infra");

    let mut cache = WikiCache::default();
    cache.metas.insert(
        "ADR-W1".into(),
        ConstraintMeta {
            id: "ADR-W1".into(),
            summary: "Wiki roundtrip test.".into(),
            severity: ConstraintSeverity::Advisory,
            predicate_kind: PredicateKind::Static,
            domains: vec!["test".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            payload_version: "v1".into(),
            inline_predicate: None,
            source: None,
            last_updated_ms: None,
        },
    );

    let rev = nats.put_wiki_cache(&cache, None).await.unwrap();
    let (loaded, loaded_rev) = nats.get_wiki_cache().await.unwrap().unwrap();
    assert_eq!(loaded_rev, rev);
    assert!(loaded.metas.contains_key("ADR-W1"));
}

#[tokio::test]
#[ignore = "requires NATS server"]
async fn constraint_payload_object_store_roundtrip() {
    use h2ai_constraints::types::{ConstraintPayload, ConstraintPredicate};

    let url = h2ai_config::H2AIConfig::default().nats_url;
    let nats = match NatsClient::connect(&url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable — skipping: {e}");
            return;
        }
    };
    nats.ensure_infrastructure().await.expect("infra");

    let payload = ConstraintPayload {
        id: "GDPR-TEST".into(),
        version: "v1".into(),
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "Verify data minimization is demonstrated.".into(),
        },
    };

    nats.put_constraint_payload(&payload).await.unwrap();
    let loaded = nats
        .get_constraint_payload("GDPR-TEST", "v1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded.id, "GDPR-TEST");
    assert!(matches!(
        loaded.predicate,
        ConstraintPredicate::LlmJudge { .. }
    ));
}
