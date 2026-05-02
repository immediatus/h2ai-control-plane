use futures::StreamExt;
use h2ai_agent::dispatch::DispatchLoop;
use h2ai_types::agent::{AgentDescriptor, ContextPayload, CostTier, TaskPayload};
use h2ai_types::identity::{AgentId, TaskId};
use h2ai_types::physics::TauValue;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
#[ignore]
async fn dispatch_executes_addressed_task_and_publishes_result() {
    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
    let nats = match h2ai_state::NatsClient::connect(&nats_url).await {
        Ok(c) => Arc::new(c),
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return;
        }
    };
    nats.ensure_infrastructure().await.unwrap();

    let agent_id = AgentId::from(uuid::Uuid::new_v4().to_string());
    let task_id = TaskId::new();
    let descriptor = AgentDescriptor {
        model: "mock".into(),
        tools: vec![],
        cost_tier: CostTier::Low,
    };
    let adapter: Arc<dyn h2ai_types::adapter::IComputeAdapter> =
        Arc::new(h2ai_adapters::mock::MockAdapter::new("mock output".into()));
    let active_tasks = Arc::new(AtomicU32::new(0));

    // Subscribe to the result subject BEFORE starting dispatch
    let result_subject = h2ai_nats::subjects::task_result_subject(&task_id);
    let js = async_nats::jetstream::new(nats.client.clone());
    let stream = js.get_stream("H2AI_RESULTS").await.unwrap();
    let consumer = stream
        .create_consumer(async_nats::jetstream::consumer::pull::OrderedConfig {
            filter_subject: result_subject.clone(),
            ..Default::default()
        })
        .await
        .unwrap();
    let mut messages = consumer.messages().await.unwrap();

    // Spawn the dispatch loop
    let dispatch_client = nats.client.clone();
    let dispatch_agent_id = agent_id.clone();
    let dispatch_adapter = adapter.clone();
    let dispatch_active = active_tasks.clone();
    let _dispatch_handle = tokio::spawn(async move {
        DispatchLoop::new(
            dispatch_client,
            dispatch_agent_id,
            dispatch_adapter,
            dispatch_active,
        )
        .run()
        .await
        .unwrap();
    });

    // Small delay for the dispatch loop to set up its subscription
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Publish a task addressed to this agent
    let payload = TaskPayload {
        task_id: task_id.clone(),
        agent_id: agent_id.clone(),
        agent: descriptor,
        instructions: "test task".into(),
        context: ContextPayload::Inline("test context".into()),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 64,
    };
    nats.client
        .publish(
            h2ai_nats::subjects::ephemeral_task_subject(&task_id),
            serde_json::to_vec(&payload).unwrap().into(),
        )
        .await
        .unwrap();

    // Wait for the result on JetStream
    let msg = tokio::time::timeout(Duration::from_secs(5), messages.next())
        .await
        .expect("timeout waiting for result")
        .expect("stream closed")
        .expect("message error");

    let result: h2ai_types::agent::TaskResult = serde_json::from_slice(&msg.payload).unwrap();
    assert_eq!(result.task_id, task_id);
    assert_eq!(result.agent_id, agent_id);
    assert!(
        result.error.is_none(),
        "expected no error: {:?}",
        result.error
    );
}
