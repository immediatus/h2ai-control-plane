use crate::error::ProvisionError;
use crate::provider::AgentProvider;
use crate::scheduling::{AgentCandidate, LeastLoadedPolicy, SchedulingPolicy};
use async_trait::async_trait;
use dashmap::DashMap;
use futures::StreamExt;
use h2ai_nats::subjects::agent_terminate_subject;
use h2ai_types::agent::{AgentDescriptor, AgentHeartbeat, TaskRequirements};
use h2ai_types::identity::AgentId;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct AgentRegistration {
    pub descriptor: AgentDescriptor,
    pub last_seen: Instant,
    /// Self-reported by the edge agent in each heartbeat. May be up to TTL-seconds
    /// stale. Used as an approximation for least-loaded scheduling — not authoritative.
    pub active_tasks: u32,
}

pub struct NatsAgentProvider {
    registry: Arc<DashMap<String, AgentRegistration>>,
    ttl: Duration,
    nats: Option<async_nats::Client>,
    handles: Vec<tokio::task::JoinHandle<()>>,
    policy: Arc<dyn SchedulingPolicy>,
}

impl NatsAgentProvider {
    pub async fn new(nats: async_nats::Client, ttl: Duration) -> Result<Self, ProvisionError> {
        if ttl.is_zero() {
            return Err(ProvisionError::Transport(
                "agent registry TTL must be positive".into(),
            ));
        }
        let registry: Arc<DashMap<String, AgentRegistration>> = Arc::new(DashMap::new());

        let mut sub = nats
            .subscribe("h2ai.heartbeat.>")
            .await
            .map_err(|e| ProvisionError::Transport(e.to_string()))?;

        let registry_hb = registry.clone();
        let heartbeat_handle = tokio::spawn(async move {
            while let Some(msg) = sub.next().await {
                if let Ok(hb) = serde_json::from_slice::<AgentHeartbeat>(&msg.payload) {
                    registry_hb.insert(
                        hb.agent_id.to_string(),
                        AgentRegistration {
                            descriptor: hb.descriptor,
                            last_seen: Instant::now(),
                            active_tasks: hb.active_tasks,
                        },
                    );
                }
            }
            tracing::warn!("heartbeat subscriber closed — agent registry will drain over TTL");
        });

        let registry_clean = registry.clone();
        let ttl_clean = ttl;
        let cleanup_handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(ttl_clean / 2);
            loop {
                ticker.tick().await;
                let cutoff = Instant::now() - ttl_clean;
                registry_clean.retain(|_, r| r.last_seen > cutoff);
                tracing::debug!(live_agents = registry_clean.len(), "agent registry pruned");
            }
        });

        Ok(Self {
            registry,
            ttl,
            nats: Some(nats),
            handles: vec![heartbeat_handle, cleanup_handle],
            policy: Arc::new(LeastLoadedPolicy),
        })
    }

    pub fn with_policy(mut self, policy: Arc<dyn SchedulingPolicy>) -> Self {
        self.policy = policy;
        self
    }

    fn live_matching(&self, descriptor: &AgentDescriptor) -> usize {
        let cutoff = Instant::now() - self.ttl;
        self.registry
            .iter()
            .filter(|r| r.descriptor == *descriptor && r.last_seen > cutoff)
            .count()
    }

    #[cfg(feature = "testing")]
    pub fn new_test_only() -> Self {
        Self {
            registry: Arc::new(DashMap::new()),
            ttl: std::time::Duration::from_secs(30),
            nats: None,
            handles: vec![],
            policy: Arc::new(LeastLoadedPolicy),
        }
    }

    #[cfg(feature = "testing")]
    pub fn inject_registration(
        &self,
        agent_id: AgentId,
        descriptor: AgentDescriptor,
        active_tasks: u32,
    ) {
        self.registry.insert(
            agent_id.to_string(),
            AgentRegistration {
                descriptor,
                last_seen: Instant::now(),
                active_tasks,
            },
        );
    }
}

impl Drop for NatsAgentProvider {
    fn drop(&mut self) {
        for h in &self.handles {
            h.abort();
        }
    }
}

#[async_trait]
impl AgentProvider for NatsAgentProvider {
    async fn ensure_agent_capacity(
        &self,
        descriptor: &AgentDescriptor,
        task_load: usize,
    ) -> Result<(), ProvisionError> {
        let live = self.live_matching(descriptor);
        if live >= task_load {
            Ok(())
        } else {
            Err(ProvisionError::CapacityLimitReached {
                agent_type: format!("{} (live={live}, need={task_load})", descriptor.model),
            })
        }
    }

    async fn terminate_agent(&self, agent_id: &AgentId) -> Result<(), ProvisionError> {
        if let Some(ref nats) = self.nats {
            nats.publish(agent_terminate_subject(agent_id), bytes::Bytes::new())
                .await
                .map_err(|e| ProvisionError::Transport(e.to_string()))?;
            self.registry.remove(&agent_id.to_string());
            Ok(())
        } else {
            Err(ProvisionError::Transport(
                "no NATS connection in test-only provider".into(),
            ))
        }
    }

    async fn select_agent(
        &self,
        requirements: &TaskRequirements,
    ) -> Result<AgentId, ProvisionError> {
        let cutoff = Instant::now() - self.ttl;

        let candidates: Vec<AgentCandidate> = self
            .registry
            .iter()
            .filter(|r| {
                r.last_seen > cutoff
                    && r.descriptor.cost_tier <= requirements.max_cost_tier
                    && requirements
                        .required_tools
                        .iter()
                        .all(|t| r.descriptor.tools.contains(t))
            })
            .map(|r| AgentCandidate {
                agent_id: AgentId::from(r.key().as_str()),
                descriptor: r.descriptor.clone(),
                active_tasks: r.active_tasks,
            })
            .collect();

        self.policy
            .select(&candidates)
            .ok_or_else(|| ProvisionError::NoAgentsAvailable {
                max_tier: requirements.max_cost_tier.clone(),
                tools: requirements.required_tools.clone(),
            })
    }
}
