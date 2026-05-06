use async_nats::Client;
use futures::StreamExt;
use h2ai_config::H2AIConfig;
use h2ai_nats::subjects::task_result_subject;
use h2ai_tools::registry::ToolRegistry;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::agent::{ContextPayload, TaskPayload, TaskResult};
use h2ai_types::identity::AgentId;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub struct DispatchLoop {
    client: Client,
    agent_id: AgentId,
    adapter: Arc<dyn IComputeAdapter>,
    active_tasks: Arc<AtomicU32>,
    cfg: Arc<H2AIConfig>,
}

impl DispatchLoop {
    pub fn new(
        client: Client,
        agent_id: AgentId,
        adapter: Arc<dyn IComputeAdapter>,
        active_tasks: Arc<AtomicU32>,
        cfg: Arc<H2AIConfig>,
    ) -> Self {
        Self {
            client,
            agent_id,
            adapter,
            active_tasks,
            cfg,
        }
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let mut sub = self
            .client
            .subscribe("h2ai.tasks.ephemeral.>".to_owned())
            .await?;

        let terminate_subject = h2ai_nats::subjects::agent_terminate_subject(&self.agent_id);
        let mut terminate_sub = self.client.subscribe(terminate_subject).await?;

        loop {
            tokio::select! {
                Some(msg) = sub.next() => {
                    let payload: TaskPayload = match serde_json::from_slice(&msg.payload) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::warn!(error = %e, "failed to deserialize TaskPayload");
                            continue;
                        }
                    };
                    if payload.agent_id != self.agent_id {
                        continue;
                    }
                    self.active_tasks.fetch_add(1, Ordering::Relaxed);
                    let result = self.execute_task(payload).await;
                    self.active_tasks.fetch_sub(1, Ordering::Relaxed);
                    if let Err(e) = self.publish_result(result).await {
                        tracing::error!(error = %e, "failed to publish task result");
                    }
                }
                Some(_) = terminate_sub.next() => {
                    tracing::info!(agent_id = %self.agent_id, "received terminate signal");
                    break;
                }
                else => break,
            }
        }
        Ok(())
    }

    async fn execute_task(&self, payload: TaskPayload) -> TaskResult {
        let registry = ToolRegistry::for_wave(&self.cfg, payload.wave_mode);

        let system_context = match payload.context {
            ContextPayload::Inline(s) => s,
            ContextPayload::Ref { hash, byte_len } => {
                tracing::warn!(
                    hash = %hash,
                    byte_len = byte_len,
                    "Ref context payload received; NatsObjectStoreBackend not yet implemented — using empty context"
                );
                String::new()
            }
        };

        let tao_input = crate::tao_agent::TaoAgentInput {
            instructions: payload.instructions.clone(),
            system_context,
            tau: payload.tau,
            max_tokens: payload.max_tokens,
        };

        let result = crate::tao_agent::TaoAgent::new(self.adapter.as_ref(), registry, &self.cfg)
            .run(tao_input)
            .await;

        if result.truncated {
            tracing::warn!(
                task_id = %payload.task_id,
                tool_calls = result.tool_calls.len(),
                "task result is truncated — iteration cap reached before LLM produced a final answer"
            );
        }

        // Extract error before moving result.output so the AdapterError cause string
        // is visible to the control plane, not just in tracing logs.
        let error = if result.adapter_failed {
            Some(result.output.clone())
        } else {
            None
        };
        TaskResult {
            task_id: payload.task_id,
            agent_id: self.agent_id.clone(),
            output: result.output,
            token_cost: result.total_token_cost,
            error,
            tool_calls: result.tool_calls,
        }
    }

    async fn publish_result(&self, result: TaskResult) -> anyhow::Result<()> {
        let subject = task_result_subject(&result.task_id);
        let bytes = serde_json::to_vec(&result)?;
        let js = async_nats::jetstream::new(self.client.clone());
        js.publish(subject, bytes.into()).await?.await?;
        Ok(())
    }
}

/// Convenience entry point called from main.rs
pub async fn run(
    client: Client,
    agent_id: AgentId,
    adapter: Arc<dyn IComputeAdapter>,
    active_tasks: Arc<AtomicU32>,
    cfg: Arc<H2AIConfig>,
) -> anyhow::Result<()> {
    DispatchLoop::new(client, agent_id, adapter, active_tasks, cfg)
        .run()
        .await
}
