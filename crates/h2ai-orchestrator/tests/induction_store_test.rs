//! Requires NATS — skipped when NATS_URL env is absent.

use h2ai_orchestrator::induction_store::InductionStore;
use h2ai_types::config::AgentRole;

async fn maybe_connect() -> Option<async_nats::Client> {
    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    async_nats::connect(&url).await.ok()
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn cold_start_returns_empty() {
    let Some(nats) = maybe_connect().await else {
        return;
    };
    let bucket = format!("H2AI_MEMORY_test_{}", uuid::Uuid::new_v4().simple());
    let store = InductionStore::create(&nats, &bucket).await.unwrap();
    let patterns = store
        .load_patterns(&["fintech".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    assert!(patterns.is_empty());
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn record_and_load_round_trip() {
    let Some(nats) = maybe_connect().await else {
        return;
    };
    let bucket = format!("H2AI_MEMORY_test_{}", uuid::Uuid::new_v4().simple());
    let store = InductionStore::create(&nats, &bucket).await.unwrap();

    store
        .record(
            &["ofac-sdncheck".to_string(), "wire-transfer".to_string()],
            &AgentRole::Executor,
            &["fintech".to_string()],
        )
        .await
        .unwrap();

    let patterns = store
        .load_patterns(&["fintech".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    assert!(!patterns.is_empty());
    assert!(patterns.iter().any(|p| p.node_id == "ofac-sdncheck"));
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn load_filters_by_role() {
    let Some(nats) = maybe_connect().await else {
        return;
    };
    let bucket = format!("H2AI_MEMORY_test_{}", uuid::Uuid::new_v4().simple());
    let store = InductionStore::create(&nats, &bucket).await.unwrap();

    store
        .record(
            &["gdpr-consent".to_string()],
            &AgentRole::Evaluator,
            &["compliance".to_string()],
        )
        .await
        .unwrap();

    // Executor query should not return Evaluator patterns
    let patterns = store
        .load_patterns(&["compliance".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    assert!(!patterns.iter().any(|p| p.node_id == "gdpr-consent"));
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn hit_rate_accumulates_on_repeated_record() {
    let Some(nats) = maybe_connect().await else {
        return;
    };
    let bucket = format!("H2AI_MEMORY_test_{}", uuid::Uuid::new_v4().simple());
    let store = InductionStore::create(&nats, &bucket).await.unwrap();

    store
        .record(
            &["node-a".to_string()],
            &AgentRole::Executor,
            &["domain-x".to_string()],
        )
        .await
        .unwrap();
    store
        .record(
            &["node-a".to_string()],
            &AgentRole::Executor,
            &["domain-x".to_string()],
        )
        .await
        .unwrap();

    let patterns = store
        .load_patterns(&["domain-x".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    let node_a = patterns.iter().find(|p| p.node_id == "node-a").unwrap();
    assert!(
        node_a.hit_rate > 1.0,
        "hit_rate should accumulate: {}",
        node_a.hit_rate
    );
}
