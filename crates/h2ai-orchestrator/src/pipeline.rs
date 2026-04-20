use crate::error::OrchestratorError;
use async_nats::Client as NatsClient;
use futures::StreamExt;
use h2ai_memory::provider::MemoryProvider;
use h2ai_nats::subjects::{agent_telemetry_subject, ephemeral_task_subject, task_result_subject};
use h2ai_provisioner::provider::AgentProvider;
use h2ai_telemetry::provider::AuditProvider;
use h2ai_telemetry::redaction::redact_event;
use h2ai_types::agent::{AgentDescriptor, AgentTelemetryEvent, TaskPayload, TaskResult};
use h2ai_types::identity::{AgentId, TaskId};
use h2ai_types::physics::TauValue;
use std::time::Duration;

pub struct OrchestratorPipeline<M, P, A> {
    memory: M,
    provisioner: P,
    auditor: A,
    nats: NatsClient,
}

impl<M, P, A> OrchestratorPipeline<M, P, A>
where
    M: MemoryProvider,
    P: AgentProvider,
    A: AuditProvider,
{
    pub fn new(memory: M, provisioner: P, auditor: A, nats: NatsClient) -> Self {
        Self {
            memory,
            provisioner,
            auditor,
            nats,
        }
    }

    async fn assemble_context(&self, session_id: &str) -> Result<String, OrchestratorError> {
        let history = self
            .memory
            .get_recent_history(session_id, 10)
            .await
            .map_err(|e| OrchestratorError::Memory(e.to_string()))?;
        let context = history
            .iter()
            .filter_map(|v| v.get("content").and_then(|c| c.as_str()).map(str::to_owned))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(context)
    }

    pub async fn execute(
        &self,
        session_id: &str,
        instructions: &str,
        agent: AgentDescriptor,
        tau: TauValue,
        max_tokens: u64,
    ) -> Result<TaskId, OrchestratorError> {
        let context = self.assemble_context(session_id).await?;

        let task_id = TaskId::new();
        let agent_id = AgentId::from(task_id.to_string());
        let payload = TaskPayload {
            task_id: task_id.clone(),
            agent_id: agent_id.clone(),
            agent: agent.clone(),
            instructions: instructions.to_string(),
            context,
            tau,
            max_tokens,
        };

        self.provisioner
            .ensure_agent_capacity(&agent, 1)
            .await
            .map_err(|e| OrchestratorError::Provision(e.to_string()))?;

        let subject = ephemeral_task_subject(&task_id);
        let payload_json = serde_json::to_string(&payload)
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;
        self.nats
            .publish(subject, payload_json.into())
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        Ok(task_id)
    }

    /// Commit a completed TaskResult to memory and flush audit log.
    pub async fn finalize(
        &self,
        session_id: &str,
        result: &TaskResult,
    ) -> Result<(), OrchestratorError> {
        let memory_entry = serde_json::json!({
            "role": "assistant",
            "content": result.output,
            "task_id": result.task_id.to_string(),
            "token_cost": result.token_cost,
        });
        self.memory
            .commit_new_memories(session_id, vec![memory_entry])
            .await
            .map_err(|e| OrchestratorError::Memory(e.to_string()))?;

        self.auditor
            .flush()
            .await
            .map_err(|e| OrchestratorError::Telemetry(e.to_string()))?;

        Ok(())
    }

    /// Record a telemetry event, applying redaction first.
    pub async fn record_telemetry(
        &self,
        event: AgentTelemetryEvent,
    ) -> Result<(), OrchestratorError> {
        let redacted = redact_event(event);
        self.auditor
            .record_event(redacted)
            .await
            .map_err(|e| OrchestratorError::Telemetry(e.to_string()))?;
        Ok(())
    }

    /// Full dispatch-and-await pipeline.
    ///
    /// Publishes TaskPayload, subscribes to telemetry and result subjects, drives a
    /// `tokio::select!` loop routing telemetry to the AuditProvider and returning the
    /// TaskResult once received. Finalizes (commits memory + flushes audit) on success.
    /// Returns `Err(Timeout)` if no result arrives within `timeout`.
    pub async fn execute_and_await(
        &self,
        session_id: &str,
        instructions: &str,
        agent: AgentDescriptor,
        tau: TauValue,
        max_tokens: u64,
        timeout: Duration,
    ) -> Result<TaskResult, OrchestratorError> {
        let context = self.assemble_context(session_id).await?;

        let task_id = TaskId::new();
        let agent_id = AgentId::from(task_id.to_string());
        let payload = TaskPayload {
            task_id: task_id.clone(),
            agent_id: agent_id.clone(),
            agent: agent.clone(),
            instructions: instructions.to_string(),
            context,
            tau,
            max_tokens,
        };

        self.provisioner
            .ensure_agent_capacity(&agent, 1)
            .await
            .map_err(|e| OrchestratorError::Provision(e.to_string()))?;

        let payload_json = serde_json::to_string(&payload)
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;
        self.nats
            .publish(ephemeral_task_subject(&task_id), payload_json.into())
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        let mut telemetry_sub = self
            .nats
            .subscribe(agent_telemetry_subject(&agent_id))
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        let mut result_sub = self
            .nats
            .subscribe(task_result_subject(&task_id))
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            tokio::select! {
                msg = result_sub.next() => {
                    match msg {
                        Some(msg) => {
                            let result = serde_json::from_slice::<TaskResult>(&msg.payload)
                                .map_err(|e| OrchestratorError::Deserialize(e.to_string()))?;
                            self.finalize(session_id, &result).await?;
                            return Ok(result);
                        }
                        None => return Err(OrchestratorError::Transport(
                            "result subject closed unexpectedly".into(),
                        )),
                    }
                }
                msg = telemetry_sub.next() => {
                    if let Some(msg) = msg {
                        if let Ok(event) = serde_json::from_slice::<AgentTelemetryEvent>(&msg.payload) {
                            if let Err(e) = self.record_telemetry(event).await {
                                tracing::warn!("telemetry record failed: {e}");
                            }
                        }
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(OrchestratorError::Timeout {
                        task_id: task_id.to_string(),
                    });
                }
            }
        }
    }
}
