#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::significant_drop_tightening
)]

use h2ai_orchestrator::induction_store::{InMemoryKvBackend, InductionStore, KvBackend};
use h2ai_types::config::AgentRole;
use h2ai_types::sizing::TauValue;
use std::sync::Arc;

fn mock_store() -> InductionStore {
    InductionStore::from_backend(Arc::new(InMemoryKvBackend::default()))
}

// ── cold start ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn cold_start_returns_empty() {
    let store = mock_store();
    let patterns = store
        .load_patterns(&["fintech".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    assert!(patterns.is_empty());
}

// ── record + load round-trip ──────────────────────────────────────────────────

#[tokio::test]
async fn record_and_load_round_trip() {
    let store = mock_store();
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
    assert!(patterns.iter().any(|p| p.node_id == "wire-transfer"));
}

// ── load_patterns filters by role ─────────────────────────────────────────────

#[tokio::test]
async fn load_filters_by_role() {
    let store = mock_store();
    store
        .record(
            &["gdpr-consent".to_string()],
            &AgentRole::Evaluator,
            &["compliance".to_string()],
        )
        .await
        .unwrap();

    let patterns = store
        .load_patterns(&["compliance".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    assert!(!patterns.iter().any(|p| p.node_id == "gdpr-consent"));
}

// ── hit_rate accumulates ──────────────────────────────────────────────────────

#[tokio::test]
async fn hit_rate_accumulates_on_repeated_record() {
    let store = mock_store();
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

// ── Synthesizer role ──────────────────────────────────────────────────────────

#[tokio::test]
async fn record_with_synthesizer_role_is_found_by_load_patterns() {
    let store = mock_store();
    store
        .record(
            &["synth-node".to_string()],
            &AgentRole::Synthesizer,
            &["synthesis-domain".to_string()],
        )
        .await
        .unwrap();

    let patterns = store
        .load_patterns(&["synthesis-domain".to_string()], &AgentRole::Synthesizer)
        .await
        .unwrap();
    assert!(patterns.iter().any(|p| p.node_id == "synth-node"));
}

// ── Coordinator role ──────────────────────────────────────────────────────────

#[tokio::test]
async fn record_with_coordinator_role_is_found_by_load_patterns() {
    let store = mock_store();
    store
        .record(
            &["coord-node".to_string()],
            &AgentRole::Coordinator,
            &["coord-domain".to_string()],
        )
        .await
        .unwrap();

    let patterns = store
        .load_patterns(&["coord-domain".to_string()], &AgentRole::Coordinator)
        .await
        .unwrap();
    assert!(patterns.iter().any(|p| p.node_id == "coord-node"));
}

// ── Custom role maps to "executor" key ───────────────────────────────────────

#[tokio::test]
async fn record_with_custom_role_is_stored_under_executor_key() {
    let store = mock_store();
    let custom_role = AgentRole::Custom {
        name: "my-role".into(),
        tau: TauValue::new(0.5).unwrap(),
        role_error_cost: 0.1,
    };
    store
        .record(
            &["custom-node".to_string()],
            &custom_role,
            &["domain-c".to_string()],
        )
        .await
        .unwrap();

    let patterns = store
        .load_patterns(&["domain-c".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    assert!(
        patterns.iter().any(|p| p.node_id == "custom-node"),
        "Custom role maps to executor suffix"
    );
}

// ── domain tag merging ────────────────────────────────────────────────────────

#[tokio::test]
async fn record_merges_new_domain_tags_on_existing_pattern() {
    let store = mock_store();
    store
        .record(
            &["tag-node".to_string()],
            &AgentRole::Evaluator,
            &["tag-a".to_string()],
        )
        .await
        .unwrap();
    store
        .record(
            &["tag-node".to_string()],
            &AgentRole::Evaluator,
            &["tag-b".to_string()],
        )
        .await
        .unwrap();

    let patterns = store
        .load_patterns(&["tag-b".to_string()], &AgentRole::Evaluator)
        .await
        .unwrap();
    assert!(
        patterns.iter().any(|p| p.node_id == "tag-node"),
        "merged domain tag must be queryable"
    );
}

// ── node_id with slash is sanitized ──────────────────────────────────────────

#[tokio::test]
async fn node_id_slash_is_replaced_with_hyphen_in_key() {
    let store = mock_store();
    store
        .record(
            &["some/node/path".to_string()],
            &AgentRole::Executor,
            &["dom".to_string()],
        )
        .await
        .unwrap();

    let patterns = store
        .load_patterns(&["dom".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    assert!(patterns.iter().any(|p| p.node_id == "some/node/path"));
}

// ── corrupt value in backend → get_pattern returns None, load_patterns skips ─

#[tokio::test]
async fn corrupt_backend_value_is_silently_skipped() {
    let kv = Arc::new(InMemoryKvBackend::default());
    kv.put(
        "knowledge.bad-node.executor",
        bytes::Bytes::from_static(b"not valid json"),
    )
    .await
    .unwrap();
    let store = InductionStore::from_backend(kv);

    let patterns = store
        .load_patterns(&["any".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    assert!(
        !patterns.iter().any(|p| p.node_id == "bad-node"),
        "corrupt pattern must be silently skipped"
    );
}

// ── top-10 cap on load_patterns ───────────────────────────────────────────────

#[tokio::test]
async fn load_patterns_returns_at_most_ten_sorted_by_hit_rate() {
    let store = mock_store();
    for i in 0..12_u32 {
        let node = format!("node-{i:02}");
        for _ in 0..i {
            store
                .record(
                    std::slice::from_ref(&node),
                    &AgentRole::Executor,
                    &["d".to_string()],
                )
                .await
                .unwrap();
        }
    }

    let patterns = store
        .load_patterns(&["d".to_string()], &AgentRole::Executor)
        .await
        .unwrap();
    assert!(patterns.len() <= 10, "must be capped at 10");
    for w in patterns.windows(2) {
        assert!(w[0].hit_rate >= w[1].hit_rate, "must be sorted desc");
    }
}
