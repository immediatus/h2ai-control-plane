use async_nats::Client;
use chrono::Utc;
use h2ai_nats::subjects::HEARTBEAT_PREFIX;
use h2ai_types::agent::{AgentDescriptor, AgentHeartbeat};
use h2ai_types::identity::AgentId;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;

pub struct HeartbeatTask {
    client: Client,
    agent_id: AgentId,
    descriptor: AgentDescriptor,
    interval: Duration,
    active_tasks: Arc<AtomicU32>,
}

impl HeartbeatTask {
    pub fn new(
        client: Client,
        agent_id: AgentId,
        descriptor: AgentDescriptor,
        interval: Duration,
        active_tasks: Arc<AtomicU32>,
    ) -> Self {
        Self {
            client,
            agent_id,
            descriptor,
            interval,
            active_tasks,
        }
    }

    pub fn start(self) -> JoinHandle<()> {
        tokio::spawn(async move {
            let subject = format!("{HEARTBEAT_PREFIX}.{}", self.agent_id);
            let mut ticker = tokio::time::interval(self.interval);
            loop {
                ticker.tick().await;
                let hb = AgentHeartbeat {
                    agent_id: self.agent_id.clone(),
                    descriptor: self.descriptor.clone(),
                    timestamp: Utc::now(),
                    active_tasks: self.active_tasks.load(Ordering::Relaxed),
                };
                match serde_json::to_vec(&hb) {
                    Ok(bytes) => {
                        if let Err(e) = self.client.publish(subject.clone(), bytes.into()).await {
                            tracing::warn!(error = %e, "heartbeat publish failed");
                        }
                    }
                    Err(e) => tracing::error!(error = %e, "heartbeat serialize failed"),
                }
            }
        })
    }
}
