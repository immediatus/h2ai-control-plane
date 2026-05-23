use futures::StreamExt;
use h2ai_agent::dispatch::DispatchLoop;
use h2ai_config::H2AIConfig;
use h2ai_types::agent::{AgentDescriptor, ContextPayload, CostTier, TaskPayload, WaveMode};
use h2ai_types::identity::{AgentId, TaskId};
use h2ai_types::sizing::TauValue;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::time::Duration;

async fn connect_nats() -> Option<Arc<h2ai_state::NatsClient>> {
    let nats_url = H2AIConfig::default().nats_url;
    match h2ai_state::NatsClient::connect(&nats_url).await {
        Ok(c) => {
            let c = Arc::new(c);
            c.ensure_infrastructure().await.ok()?;
            Some(c)
        }
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            None
        }
    }
}

fn make_payload(agent_id: &AgentId, task_id: &TaskId, context: ContextPayload) -> TaskPayload {
    TaskPayload {
        task_id: task_id.clone(),
        agent_id: agent_id.clone(),
        agent: AgentDescriptor {
            model: "mock".into(),
            tools: vec![],
            cost_tier: CostTier::Low,
        },
        instructions: "test task".into(),
        context,
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 64,
        wave_mode: WaveMode::Normal,
    }
}

async fn wait_for_result(
    nats: &h2ai_state::NatsClient,
    task_id: &TaskId,
    timeout_secs: u64,
) -> h2ai_types::agent::TaskResult {
    let result_subject = h2ai_nats::subjects::task_result_subject(task_id);
    let js = async_nats::jetstream::new(nats.client.clone());
    let stream = js.get_stream("H2AI_RESULTS").await.unwrap();
    let consumer = stream
        .create_consumer(async_nats::jetstream::consumer::pull::Config {
            filter_subject: result_subject,
            ack_policy: async_nats::jetstream::consumer::AckPolicy::Explicit,
            ..Default::default()
        })
        .await
        .unwrap();
    let mut messages = consumer.messages().await.unwrap();
    let msg = tokio::time::timeout(Duration::from_secs(timeout_secs), messages.next())
        .await
        .expect("timeout waiting for result")
        .expect("stream closed")
        .expect("message error");
    serde_json::from_slice(&msg.payload).unwrap()
}

fn spawn_dispatch(
    nats: &h2ai_state::NatsClient,
    agent_id: AgentId,
    adapter: Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    cfg: H2AIConfig,
) -> (
    tokio::task::JoinHandle<()>,
    tokio::sync::oneshot::Receiver<()>,
) {
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let client = nats.client.clone();
    let active = Arc::new(AtomicU32::new(0));
    let handle = tokio::spawn(async move {
        DispatchLoop::new(client, agent_id, adapter, active, Arc::new(cfg))
            .run_with_ready(ready_tx)
            .await
            .unwrap();
    });
    (handle, ready_rx)
}

#[tokio::test]
async fn dispatch_executes_addressed_task_and_publishes_result() {
    let nats_url = h2ai_config::H2AIConfig::default().nats_url;
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
        Arc::new(h2ai_test_utils::MockAdapter::new("mock output".into()));
    let active_tasks = Arc::new(AtomicU32::new(0));

    // Subscribe to the result subject BEFORE starting dispatch
    let result_subject = h2ai_nats::subjects::task_result_subject(&task_id);
    let js = async_nats::jetstream::new(nats.client.clone());
    let stream = js.get_stream("H2AI_RESULTS").await.unwrap();
    let consumer = stream
        .create_consumer(async_nats::jetstream::consumer::pull::Config {
            filter_subject: result_subject.clone(),
            ack_policy: async_nats::jetstream::consumer::AckPolicy::Explicit,
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
    let dispatch_cfg = Arc::new(h2ai_config::H2AIConfig::default());
    let _dispatch_handle = tokio::spawn(async move {
        DispatchLoop::new(
            dispatch_client,
            dispatch_agent_id,
            dispatch_adapter,
            dispatch_active,
            dispatch_cfg,
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
        wave_mode: h2ai_types::agent::WaveMode::Normal,
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

// ── Lines 82-86: malformed TaskPayload bytes are silently skipped ─────────────

#[tokio::test]
async fn dispatch_skips_malformed_payload_and_continues() {
    let Some(nats) = connect_nats().await else {
        return;
    };

    let agent_id = AgentId::from(uuid::Uuid::new_v4().to_string());
    let task_id = TaskId::new();
    let adapter: Arc<dyn h2ai_types::adapter::IComputeAdapter> =
        Arc::new(h2ai_test_utils::MockAdapter::new("ok".into()));

    let (_, ready_rx) = spawn_dispatch(&nats, agent_id.clone(), adapter, H2AIConfig::default());
    ready_rx.await.unwrap();

    // Publish garbage bytes — loop must survive and process the next valid message
    nats.client
        .publish(
            h2ai_nats::subjects::ephemeral_task_subject(&task_id),
            b"not valid json!!!".as_ref().into(),
        )
        .await
        .unwrap();

    // Now send a valid task — it must still be processed
    let valid_id = TaskId::new();
    let payload = make_payload(&agent_id, &valid_id, ContextPayload::Inline("ctx".into()));
    nats.client
        .publish(
            h2ai_nats::subjects::ephemeral_task_subject(&valid_id),
            serde_json::to_vec(&payload).unwrap().into(),
        )
        .await
        .unwrap();

    let result = wait_for_result(&nats, &valid_id, 5).await;
    assert_eq!(
        result.task_id, valid_id,
        "valid task after bad payload must succeed"
    );
}

// ── Lines 87-89: task for a different agent_id is silently ignored ────────────

#[tokio::test]
async fn dispatch_ignores_task_for_different_agent() {
    let Some(nats) = connect_nats().await else {
        return;
    };

    let my_agent = AgentId::from(uuid::Uuid::new_v4().to_string());
    let other_agent = AgentId::from(uuid::Uuid::new_v4().to_string());
    let adapter: Arc<dyn h2ai_types::adapter::IComputeAdapter> =
        Arc::new(h2ai_test_utils::MockAdapter::new("ok".into()));

    let (_, ready_rx) = spawn_dispatch(&nats, my_agent.clone(), adapter, H2AIConfig::default());
    ready_rx.await.unwrap();

    // Publish a task addressed to a different agent
    let foreign_task_id = TaskId::new();
    let foreign_payload = make_payload(
        &other_agent,
        &foreign_task_id,
        ContextPayload::Inline("ctx".into()),
    );
    nats.client
        .publish(
            h2ai_nats::subjects::ephemeral_task_subject(&foreign_task_id),
            serde_json::to_vec(&foreign_payload).unwrap().into(),
        )
        .await
        .unwrap();

    // Then send a valid task for our agent — must be processed
    let my_task_id = TaskId::new();
    let my_payload = make_payload(&my_agent, &my_task_id, ContextPayload::Inline("ctx".into()));
    nats.client
        .publish(
            h2ai_nats::subjects::ephemeral_task_subject(&my_task_id),
            serde_json::to_vec(&my_payload).unwrap().into(),
        )
        .await
        .unwrap();

    let result = wait_for_result(&nats, &my_task_id, 5).await;
    assert_eq!(result.agent_id, my_agent, "must only process own tasks");
}

// ── Lines 112-119: ContextPayload::Ref falls back to empty context ────────────

#[tokio::test]
async fn dispatch_ref_context_falls_back_to_empty_string() {
    let Some(nats) = connect_nats().await else {
        return;
    };

    let agent_id = AgentId::from(uuid::Uuid::new_v4().to_string());
    let task_id = TaskId::new();
    let adapter: Arc<dyn h2ai_types::adapter::IComputeAdapter> = Arc::new(
        h2ai_test_utils::MockAdapter::new("ref-context-output".into()),
    );

    let (_, ready_rx) = spawn_dispatch(&nats, agent_id.clone(), adapter, H2AIConfig::default());
    ready_rx.await.unwrap();

    let payload = make_payload(
        &agent_id,
        &task_id,
        ContextPayload::Ref {
            hash: "a".repeat(64),
            byte_len: 100,
        },
    );
    nats.client
        .publish(
            h2ai_nats::subjects::ephemeral_task_subject(&task_id),
            serde_json::to_vec(&payload).unwrap().into(),
        )
        .await
        .unwrap();

    let result = wait_for_result(&nats, &task_id, 5).await;
    assert_eq!(result.task_id, task_id, "Ref context task must complete");
    assert!(result.error.is_none(), "Ref context must not produce error");
}

// ── Lines 143-145: adapter_failed path sets error field ──────────────────────

#[tokio::test]
async fn dispatch_sets_error_field_when_adapter_fails() {
    let Some(nats) = connect_nats().await else {
        return;
    };

    let agent_id = AgentId::from(uuid::Uuid::new_v4().to_string());
    let task_id = TaskId::new();
    let adapter: Arc<dyn h2ai_types::adapter::IComputeAdapter> =
        Arc::new(h2ai_test_utils::FailingMockAdapter::new());

    let (_, ready_rx) = spawn_dispatch(&nats, agent_id.clone(), adapter, H2AIConfig::default());
    ready_rx.await.unwrap();

    let payload = make_payload(&agent_id, &task_id, ContextPayload::Inline("ctx".into()));
    nats.client
        .publish(
            h2ai_nats::subjects::ephemeral_task_subject(&task_id),
            serde_json::to_vec(&payload).unwrap().into(),
        )
        .await
        .unwrap();

    let result = wait_for_result(&nats, &task_id, 5).await;
    assert!(
        result.error.is_some(),
        "failed adapter must set error field on TaskResult"
    );
}

// ── Lines 133-138: truncated path logs warn when iteration cap hit ────────────

#[tokio::test]
async fn dispatch_truncated_result_when_adapter_always_returns_tool_call() {
    let Some(nats) = connect_nats().await else {
        return;
    };

    let agent_id = AgentId::from(uuid::Uuid::new_v4().to_string());
    let task_id = TaskId::new();
    // Adapter always returns a shell tool-call JSON → TaoAgent hits iteration cap
    let tool_call_json = r#"{"tool":"shell","input":{"command":"echo","args":["hi"]}}"#;
    let adapter: Arc<dyn h2ai_types::adapter::IComputeAdapter> =
        Arc::new(h2ai_test_utils::MockAdapter::new(tool_call_json.into()));

    let cfg = H2AIConfig {
        agent_max_tool_iterations: 1,
        ..H2AIConfig::default()
    };
    let (_, ready_rx) = spawn_dispatch(&nats, agent_id.clone(), adapter, cfg);
    ready_rx.await.unwrap();

    let payload = make_payload(&agent_id, &task_id, ContextPayload::Inline("ctx".into()));
    nats.client
        .publish(
            h2ai_nats::subjects::ephemeral_task_subject(&task_id),
            serde_json::to_vec(&payload).unwrap().into(),
        )
        .await
        .unwrap();

    let result = wait_for_result(&nats, &task_id, 5).await;
    // Task completes (truncated internally, but result is still published)
    assert_eq!(
        result.task_id, task_id,
        "truncated task must publish a result"
    );
}
