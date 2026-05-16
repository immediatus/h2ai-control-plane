// Integration tests for task checkpoint KV operations.
// Require a live NATS server — run with:
//   cargo test -p h2ai-state --test checkpoint_integration_test -- --include-ignored

use h2ai_state::nats::NatsClient;
use h2ai_types::checkpoint::TaskCheckpoint;

fn make_checkpoint(task_id: &str) -> TaskCheckpoint {
    TaskCheckpoint {
        task_id: task_id.to_string(),
        phase: "Merging".to_string(),
        node_id: "test-node".to_string(),
        lease_seq: 0,
        proposals: vec!["prop A".to_string()],
        auditor_survivors: vec![0],
        resolved_output: Some("the answer".to_string()),
        manifest_json: "{}".to_string(),
        object_store_ref: None,
        created_at_ms: 1000,
        updated_at_ms: 1000,
        constraint_snapshot: None,
        j_eff: None,
    }
}

#[tokio::test]
#[ignore = "requires NATS server"]
async fn checkpoint_kv_put_get_delete_roundtrip() {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    let client = NatsClient::connect(&url).await.expect("connect");
    client.ensure_infrastructure().await.expect("infra");

    let c = make_checkpoint("roundtrip-task");
    let rev = client.put_task_checkpoint(&c, None).await.expect("put");
    assert!(rev > 0);
    let loaded = client
        .get_task_checkpoint("roundtrip-task")
        .await
        .expect("get");
    let loaded = loaded.expect("should be Some");
    assert_eq!(loaded.task_id, "roundtrip-task");
    assert_eq!(loaded.phase, "Merging");
    client
        .delete_task_checkpoint("roundtrip-task")
        .await
        .expect("delete");
    let after = client
        .get_task_checkpoint("roundtrip-task")
        .await
        .expect("get after delete");
    assert!(after.is_none());
}

#[tokio::test]
#[ignore = "requires NATS server"]
async fn list_checkpoints_returns_all_in_flight() {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    let client = NatsClient::connect(&url).await.expect("connect");
    client.ensure_infrastructure().await.expect("infra");

    let c1 = make_checkpoint("list-task-1");
    let c2 = make_checkpoint("list-task-2");
    client.put_task_checkpoint(&c1, None).await.expect("put 1");
    client.put_task_checkpoint(&c2, None).await.expect("put 2");
    let all = client.list_task_checkpoints().await;
    let ids: Vec<&str> = all.iter().map(|c| c.task_id.as_str()).collect();
    assert!(ids.contains(&"list-task-1"));
    assert!(ids.contains(&"list-task-2"));
    client.delete_task_checkpoint("list-task-1").await.ok();
    client.delete_task_checkpoint("list-task-2").await.ok();
}
