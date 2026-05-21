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
//! Requires NATS — skipped when NATS_URL env is absent.

use h2ai_orchestrator::induction_store::InductionStore;
use h2ai_types::config::AgentRole;

async fn maybe_connect() -> Option<async_nats::Client> {
    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
    async_nats::connect(&url).await.ok()
}

#[tokio::test]
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
