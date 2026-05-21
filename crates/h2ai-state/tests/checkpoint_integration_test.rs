#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use h2ai_state::nats::NatsClient;
use h2ai_types::checkpoint::TaskCheckpoint;

async fn connect() -> Option<NatsClient> {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    match NatsClient::connect(&url).await {
        Ok(c) => {
            if c.ensure_infrastructure().await.is_err() {
                return None;
            }
            Some(c)
        }
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            None
        }
    }
}

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
async fn checkpoint_kv_put_get_delete_roundtrip() {
    let Some(client) = connect().await else {
        return;
    };
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
async fn list_checkpoints_returns_all_in_flight() {
    let Some(client) = connect().await else {
        return;
    };
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
