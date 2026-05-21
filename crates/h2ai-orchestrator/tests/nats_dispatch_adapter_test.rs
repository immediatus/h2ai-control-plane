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
use async_trait::async_trait;
use h2ai_orchestrator::nats_dispatch_adapter::{NatsDispatchAdapter, NatsDispatchConfig};
use h2ai_provisioner::error::ProvisionError;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_state::NatsClient;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::agent::{AgentDescriptor, CostTier, TaskRequirements};
use h2ai_types::identity::AgentId;
use h2ai_types::sizing::TauValue;
use std::sync::Arc;
use std::time::Duration;

struct FakeProvider {
    agent_id: AgentId,
}

#[async_trait]
impl AgentProvider for FakeProvider {
    async fn ensure_agent_capacity(
        &self,
        _descriptor: &AgentDescriptor,
        _task_load: usize,
    ) -> Result<(), ProvisionError> {
        Ok(())
    }

    async fn terminate_agent(&self, _agent_id: &AgentId) -> Result<(), ProvisionError> {
        Ok(())
    }

    async fn select_agent(
        &self,
        _requirements: &TaskRequirements,
    ) -> Result<AgentId, ProvisionError> {
        Ok(self.agent_id.clone())
    }
}

/// Integration test: requires a running NATS server at NATS_URL (default nats://localhost:4222).
/// Run with: NATS_URL=nats://localhost:4222 cargo test -p h2ai-orchestrator -- --ignored
#[tokio::test]
async fn nats_dispatch_adapter_round_trip() {
    use futures::StreamExt;

    let nats_url = h2ai_config::H2AIConfig::default().nats_url;

    // Connect two clients: one for the adapter, one for the mock edge agent
    let nats_adapter = Arc::new(match NatsClient::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return;
        }
    });
    let nats_agent = NatsClient::connect(&nats_url)
        .await
        .expect("connect agent client");

    nats_adapter
        .ensure_infrastructure()
        .await
        .expect("ensure infrastructure");

    let agent_id = AgentId::from("mock-edge-agent");
    let provider = Arc::new(FakeProvider {
        agent_id: agent_id.clone(),
    });

    let descriptor = AgentDescriptor {
        model: "mock-model".into(),
        tools: vec![],
        cost_tier: CostTier::Low,
    };
    let requirements = TaskRequirements {
        max_cost_tier: CostTier::High,
        required_tools: vec![],
    };

    let adapter = NatsDispatchAdapter::new(NatsDispatchConfig {
        nats: nats_adapter,
        provider,
        agent_descriptor: descriptor,
        task_requirements: requirements,
        task_timeout: Duration::from_secs(5),
        payload_store: std::sync::Arc::new(
            h2ai_orchestrator::payload_store::MemoryPayloadStore::new(),
        ),
        offload_threshold_bytes: 524_288,
    });

    // Spawn a mock edge agent that subscribes to ephemeral task subjects and publishes results.
    let agent_handle = tokio::spawn(async move {
        use h2ai_types::agent::TaskResult;

        // Subscribe to all ephemeral task subjects
        let mut sub = nats_agent
            .client
            .subscribe("h2ai.tasks.ephemeral.>".to_owned())
            .await
            .expect("subscribe");

        // Wait for the task payload
        let msg = tokio::time::timeout(Duration::from_secs(5), sub.next())
            .await
            .expect("timeout waiting for task payload")
            .expect("message");

        let payload: h2ai_types::agent::TaskPayload =
            serde_json::from_slice(&msg.payload).expect("deserialize TaskPayload");

        // Publish a result echoing the instructions back
        let result = TaskResult {
            task_id: payload.task_id.clone(),
            agent_id: payload.agent_id.clone(),
            output: format!("echo: {}", payload.instructions),
            token_cost: 42,
            error: None,
            tool_calls: vec![],
        };

        let js = async_nats::jetstream::new(nats_agent.client.clone());
        let result_subject = h2ai_nats::subjects::task_result_subject(&payload.task_id);
        js.publish(result_subject, serde_json::to_vec(&result).unwrap().into())
            .await
            .expect("publish result")
            .await
            .expect("ack publish");
    });

    let request = ComputeRequest {
        system_context: "test context".into(),
        task: "hello from test".into(),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 128,
    };

    let response = adapter
        .execute(request)
        .await
        .expect("adapter execute succeeded");

    assert_eq!(response.output, "echo: hello from test");
    assert_eq!(response.token_cost, 42);

    agent_handle.await.expect("agent task completed");
}
