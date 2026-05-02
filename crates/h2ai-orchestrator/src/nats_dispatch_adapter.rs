use async_trait::async_trait;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_state::NatsClient;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::agent::{AgentDescriptor, TaskPayload, TaskRequirements};
use h2ai_types::config::AdapterKind;
use h2ai_types::identity::TaskId;
use std::sync::Arc;
use std::time::Duration;

use crate::payload_store::{offload_if_large, PayloadStore};

pub struct NatsDispatchConfig {
    pub nats: Arc<NatsClient>,
    pub provider: Arc<dyn AgentProvider>,
    /// Used in TaskPayload.agent field (tells edge agent what capabilities to use).
    pub agent_descriptor: AgentDescriptor,
    /// Used for select_agent — what we require from the selected agent.
    pub task_requirements: TaskRequirements,
    pub task_timeout: Duration,
    /// Content-addressed store for large context offloading.
    pub payload_store: Arc<dyn PayloadStore>,
    /// Byte threshold above which system_context is offloaded to the store.
    pub offload_threshold_bytes: usize,
}

pub struct NatsDispatchAdapter {
    nats: Arc<NatsClient>,
    provider: Arc<dyn AgentProvider>,
    descriptor: AgentDescriptor,
    requirements: TaskRequirements,
    timeout: Duration,
    kind: AdapterKind,
    payload_store: Arc<dyn PayloadStore>,
    offload_threshold_bytes: usize,
}

impl std::fmt::Debug for NatsDispatchAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NatsDispatchAdapter")
            .field("descriptor", &self.descriptor)
            .field("timeout", &self.timeout)
            .field("kind", &self.kind)
            .field("offload_threshold_bytes", &self.offload_threshold_bytes)
            .finish_non_exhaustive()
    }
}

impl NatsDispatchAdapter {
    pub fn new(cfg: NatsDispatchConfig) -> Self {
        Self {
            nats: cfg.nats,
            provider: cfg.provider,
            descriptor: cfg.agent_descriptor,
            requirements: cfg.task_requirements,
            timeout: cfg.task_timeout,
            payload_store: cfg.payload_store,
            offload_threshold_bytes: cfg.offload_threshold_bytes,
            kind: AdapterKind::CloudGeneric {
                endpoint: "nats://dispatch".into(),
                api_key_env: String::new(),
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for NatsDispatchAdapter {
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }

    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let agent_id = self
            .provider
            .select_agent(&self.requirements)
            .await
            .map_err(|e| AdapterError::NetworkError(format!("agent selection failed: {e}")))?;

        let task_id = TaskId::new();

        let context = offload_if_large(
            request.system_context,
            self.offload_threshold_bytes,
            self.payload_store.as_ref(),
        )
        .await
        .map_err(|e| AdapterError::NetworkError(format!("payload offload failed: {e}")))?;

        let payload = TaskPayload {
            task_id: task_id.clone(),
            agent_id,
            agent: self.descriptor.clone(),
            instructions: request.task,
            context,
            tau: request.tau,
            max_tokens: request.max_tokens,
        };

        // Set up result consumer BEFORE publishing — critical ordering invariant.
        let nats = self.nats.clone();
        let timeout = self.timeout;
        let tid = task_id.clone();
        let waiter = tokio::spawn(async move { nats.await_task_result_once(&tid, timeout).await });

        // Yield once to give the consumer setup time before publish.
        tokio::task::yield_now().await;

        self.nats
            .publish_task_payload(&payload)
            .await
            .map_err(|e| AdapterError::NetworkError(format!("publish failed: {e}")))?;

        let result = waiter
            .await
            .map_err(|e| AdapterError::NetworkError(format!("waiter join error: {e}")))?
            .map_err(|e| AdapterError::NetworkError(format!("result await failed: {e}")))?;

        if let Some(err) = result.error {
            return Err(AdapterError::NetworkError(format!("agent error: {err}")));
        }

        Ok(ComputeResponse {
            output: result.output,
            token_cost: result.token_cost,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
        })
    }
}
