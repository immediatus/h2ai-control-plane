use async_nats::Client;
use futures::StreamExt;
use h2ai_nats::subjects::task_result_subject;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::agent::{ContextPayload, TaskPayload, TaskResult};
use h2ai_types::identity::AgentId;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub struct DispatchLoop {
    client: Client,
    agent_id: AgentId,
    adapter: Arc<dyn IComputeAdapter>,
    active_tasks: Arc<AtomicU32>,
}

impl DispatchLoop {
    pub fn new(
        client: Client,
        agent_id: AgentId,
        adapter: Arc<dyn IComputeAdapter>,
        active_tasks: Arc<AtomicU32>,
    ) -> Self {
        Self {
            client,
            agent_id,
            adapter,
            active_tasks,
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
        let system_context = match payload.context {
            ContextPayload::Inline(s) => s,
            ContextPayload::Ref { hash, byte_len } => {
                // Object store backend not yet available on the agent side.
                // Large contexts (Ref payloads) are unsupported until NatsObjectStoreBackend ships.
                tracing::warn!(
                    hash = %hash,
                    byte_len = byte_len,
                    "Ref context payload received; NatsObjectStoreBackend not yet implemented — using empty context"
                );
                String::new()
            }
        };
        let request = ComputeRequest {
            system_context,
            task: payload.instructions.clone(),
            tau: payload.tau,
            max_tokens: payload.max_tokens,
        };
        match self.adapter.execute(request).await {
            Ok(resp) => TaskResult {
                task_id: payload.task_id,
                agent_id: self.agent_id.clone(),
                output: resp.output,
                token_cost: resp.token_cost,
                error: None,
            },
            Err(e) => TaskResult {
                task_id: payload.task_id,
                agent_id: self.agent_id.clone(),
                output: String::new(),
                token_cost: 0,
                error: Some(e.to_string()),
            },
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
) -> anyhow::Result<()> {
    DispatchLoop::new(client, agent_id, adapter, active_tasks)
        .run()
        .await
}
