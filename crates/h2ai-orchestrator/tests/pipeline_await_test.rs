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
use async_nats::Client;
use futures::StreamExt;
use h2ai_memory::in_memory::InMemoryCache;
use h2ai_nats::subjects::task_result_subject;
use h2ai_orchestrator::error::OrchestratorError;
use h2ai_orchestrator::pipeline::OrchestratorPipeline;
use h2ai_provisioner::static_provider::StaticProvider;
use h2ai_telemetry::direct_log::DirectLogProvider;
use h2ai_types::agent::{
    AgentDescriptor, AgentTelemetryEvent, AgentTool, CostTier, TaskPayload, TaskResult,
};
use h2ai_types::sizing::TauValue;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

// Serialize pipeline tests to prevent cross-test NATS message contamination.
// Both tests subscribe to the same ephemeral subject wildcard; parallel execution
// causes test 1's fake agent to intercept test 2's task.
static PIPELINE_TEST_LOCK: std::sync::LazyLock<Arc<Mutex<()>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(())));

async fn connect() -> Option<(h2ai_state::nats::NatsClient, async_nats::Client)> {
    let url = h2ai_config::H2AIConfig::default().nats_url;
    let state_client = match h2ai_state::nats::NatsClient::connect(&url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {url} — skipping: {e}");
            return None;
        }
    };
    state_client.ensure_infrastructure().await.expect("infra");
    let nats = async_nats::connect(&url).await.expect("nats connect");
    Some((state_client, nats))
}

async fn spawn_fake_agent(nats: Client) {
    // Create the subscription before spawning so it is registered before we return
    let mut sub = nats
        .subscribe(format!("{}.*", "h2ai.tasks.ephemeral"))
        .await
        .expect("subscribe");
    tokio::spawn(async move {
        if let Some(msg) = sub.next().await {
            let payload: TaskPayload = serde_json::from_slice(&msg.payload).expect("parse payload");
            let agent_id = payload.agent_id.clone();

            let telemetry = AgentTelemetryEvent::LlmPromptSent {
                task_id: payload.task_id.clone(),
                agent_id: agent_id.clone(),
                prompt: "design a rate limiter".into(),
                timestamp: chrono::Utc::now(),
            };
            let _ = nats
                .publish(
                    format!("h2ai.telemetry.{}", agent_id),
                    serde_json::to_vec(&telemetry).unwrap().into(),
                )
                .await;

            let result = TaskResult {
                task_id: payload.task_id.clone(),
                agent_id: agent_id.clone(),
                output: "design approved".into(),
                token_cost: 150,
                error: None,
                tool_calls: vec![],
            };
            let _ = nats
                .publish(
                    task_result_subject(&payload.task_id),
                    serde_json::to_vec(&result).unwrap().into(),
                )
                .await;
        }
    });
}

#[tokio::test]
async fn execute_and_await_returns_task_result() {
    let _guard = PIPELINE_TEST_LOCK.lock().await;
    let Some((_state, nats)) = connect().await else {
        return;
    };
    let pipeline = OrchestratorPipeline::new(
        InMemoryCache::new(),
        StaticProvider::new(10),
        DirectLogProvider,
        nats.clone(),
    );

    spawn_fake_agent(nats).await;

    let agent = AgentDescriptor {
        model: "gpt-4o".into(),
        tools: vec![AgentTool::Shell],
        cost_tier: CostTier::Mid,
    };
    let result = pipeline
        .execute_and_await(
            "session-test-1",
            "design a rate limiter",
            agent,
            TauValue::new(0.5).unwrap(),
            1024,
            Duration::from_secs(5),
        )
        .await
        .expect("execute_and_await");

    assert_eq!(result.output, "design approved");
    assert_eq!(result.token_cost, 150);
    assert!(result.error.is_none());
}

#[tokio::test]
async fn execute_and_await_times_out_without_agent() {
    let _guard = PIPELINE_TEST_LOCK.lock().await;
    let Some((_state, nats)) = connect().await else {
        return;
    };
    let pipeline = OrchestratorPipeline::new(
        InMemoryCache::new(),
        StaticProvider::new(10),
        DirectLogProvider,
        nats,
    );

    let agent = AgentDescriptor {
        model: "gpt-4o".into(),
        tools: vec![],
        cost_tier: CostTier::Mid,
    };
    let err = pipeline
        .execute_and_await(
            "session-timeout",
            "this will time out",
            agent,
            TauValue::new(0.3).unwrap(),
            512,
            Duration::from_millis(200),
        )
        .await;

    assert!(matches!(err, Err(OrchestratorError::Timeout { .. })));
}

/// Spawn a fake agent that sends several telemetry bursts, then sleeps long
/// enough for the `execute_and_await` loop to drain the telemetry buffer
/// (exercising lines 213-219), and only afterwards publishes the result.
async fn spawn_fake_agent_telemetry_burst(nats: Client) {
    let mut sub = nats
        .subscribe(format!("{}.*", "h2ai.tasks.ephemeral"))
        .await
        .expect("subscribe");
    tokio::spawn(async move {
        if let Some(msg) = sub.next().await {
            let payload: TaskPayload = serde_json::from_slice(&msg.payload).expect("parse");
            let agent_id = payload.agent_id.clone();
            let task_id = payload.task_id.clone();
            let telemetry_subject = format!("h2ai.telemetry.{agent_id}");

            // Publish several telemetry events so the `select!` loop
            // is guaranteed to process at least one through the telemetry arm.
            for i in 0u64..5 {
                let event = AgentTelemetryEvent::LlmResponseReceived {
                    task_id: task_id.clone(),
                    agent_id: agent_id.clone(),
                    response: format!("thinking step {i}"),
                    token_cost: i,
                    timestamp: chrono::Utc::now(),
                };
                let _ = nats
                    .publish(
                        telemetry_subject.clone(),
                        serde_json::to_vec(&event).unwrap().into(),
                    )
                    .await;
            }

            // Wait long enough for the buffered telemetry to be consumed
            // before the result subject has any message.
            tokio::time::sleep(Duration::from_millis(50)).await;

            let result = TaskResult {
                task_id: task_id.clone(),
                agent_id: agent_id.clone(),
                output: "burst approved".into(),
                token_cost: 77,
                error: None,
                tool_calls: vec![],
            };
            let _ = nats
                .publish(
                    task_result_subject(&task_id),
                    serde_json::to_vec(&result).unwrap().into(),
                )
                .await;
        }
    });
}

/// Exercises the telemetry arm (lines 213-219) of the `execute_and_await` select loop.
///
/// The fake agent sends 5 telemetry events before publishing the result with a
/// short delay, ensuring the buffered telemetry messages are processed in the
/// telemetry arm before the result arm fires.
#[tokio::test]
async fn execute_and_await_processes_telemetry_before_result() {
    let _guard = PIPELINE_TEST_LOCK.lock().await;
    let Some((_state, nats)) = connect().await else {
        return;
    };
    let pipeline = OrchestratorPipeline::new(
        InMemoryCache::new(),
        StaticProvider::new(10),
        DirectLogProvider,
        nats.clone(),
    );

    spawn_fake_agent_telemetry_burst(nats).await;

    let agent = AgentDescriptor {
        model: "gpt-4o".into(),
        tools: vec![AgentTool::Shell],
        cost_tier: CostTier::Mid,
    };
    let result = pipeline
        .execute_and_await(
            "session-telemetry-burst",
            "process with telemetry burst",
            agent,
            TauValue::new(0.5).unwrap(),
            512,
            Duration::from_secs(5),
        )
        .await
        .expect("execute_and_await");

    assert_eq!(result.output, "burst approved");
    assert_eq!(result.token_cost, 77);
}
