// Integration test: needs NATS. Start with:
//   NATS_URL=nats://localhost:4222 cargo nextest run -p h2ai-orchestrator --test pipeline_await_test

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
use h2ai_types::physics::TauValue;
use std::time::Duration;

async fn spawn_fake_agent(nats: Client) {
    tokio::spawn(async move {
        let mut sub = nats
            .subscribe(format!("{}.*", "h2ai.tasks.ephemeral"))
            .await
            .expect("subscribe");

        if let Some(msg) = sub.next().await {
            let payload: TaskPayload = serde_json::from_slice(&msg.payload).expect("parse payload");
            // Use the agent_id the control plane assigned in the payload.
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
#[ignore = "requires live NATS at localhost:4222"]
async fn execute_and_await_returns_task_result() {
    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());

    let state_client = h2ai_state::nats::NatsClient::connect(&url)
        .await
        .expect("state connect");
    state_client.ensure_infrastructure().await.expect("infra");

    let nats = async_nats::connect(&url).await.expect("nats connect");
    let pipeline = OrchestratorPipeline::new(
        InMemoryCache::new(),
        StaticProvider::new(10),
        DirectLogProvider,
        nats.clone(),
    );

    spawn_fake_agent(nats).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

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
#[ignore = "requires live NATS at localhost:4222"]
async fn execute_and_await_times_out_without_agent() {
    let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());

    let state_client = h2ai_state::nats::NatsClient::connect(&url)
        .await
        .expect("state connect");
    state_client.ensure_infrastructure().await.expect("infra");

    let nats = async_nats::connect(&url).await.expect("nats connect");
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
