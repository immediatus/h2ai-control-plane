// requires NATS: NATS_URL=nats://localhost:4222 cargo nextest run -p h2ai-memory --test nats_kv_test

use h2ai_memory::nats_kv::NatsKvStore;
use h2ai_memory::provider::MemoryProvider;

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn session_history_survives_provider_restart() {
    let url =
        std::env::var("NATS_URL").unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
    let nats = async_nats::connect(&url).await.expect("connect");
    let session_id = format!("test-{}", uuid::Uuid::new_v4());

    // First provider — write two entries
    let p1 = NatsKvStore::create(&nats, "H2AI_SESSIONS_TEST")
        .await
        .expect("create p1");
    p1.commit_new_memories(
        &session_id,
        vec![
            serde_json::json!({"role": "user",      "content": "first"}),
            serde_json::json!({"role": "assistant", "content": "second"}),
        ],
    )
    .await
    .expect("commit");

    // Second provider instance — simulates restart
    let p2 = NatsKvStore::create(&nats, "H2AI_SESSIONS_TEST")
        .await
        .expect("create p2");
    let history = p2
        .get_recent_history(&session_id, 10)
        .await
        .expect("history");

    assert_eq!(history.len(), 2, "both entries must survive restart");
    assert_eq!(
        history[0]["content"], "first",
        "oldest entry must come first"
    );
    assert_eq!(
        history[1]["content"], "second",
        "newest entry must come last"
    );
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn get_recent_history_respects_limit_and_preserves_order() {
    let url =
        std::env::var("NATS_URL").unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
    let nats = async_nats::connect(&url).await.expect("connect");
    let session_id = format!("test-limit-{}", uuid::Uuid::new_v4());

    let p = NatsKvStore::create(&nats, "H2AI_SESSIONS_TEST")
        .await
        .expect("create");
    let entries: Vec<_> = (1..=5)
        .map(|i| serde_json::json!({"content": format!("msg-{i}")}))
        .collect();
    p.commit_new_memories(&session_id, entries)
        .await
        .expect("commit");

    let recent = p.get_recent_history(&session_id, 3).await.expect("history");

    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0]["content"], "msg-3"); // oldest of the last 3
    assert_eq!(recent[1]["content"], "msg-4");
    assert_eq!(recent[2]["content"], "msg-5"); // newest last
}
