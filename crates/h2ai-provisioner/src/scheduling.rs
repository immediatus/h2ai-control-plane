use h2ai_types::agent::AgentDescriptor;
use h2ai_types::identity::AgentId;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct AgentCandidate {
    pub agent_id: AgentId,
    pub descriptor: AgentDescriptor,
    pub active_tasks: u32,
}

pub trait SchedulingPolicy: Send + Sync {
    fn select(&self, candidates: &[AgentCandidate]) -> Option<AgentId>;
}

/// Default: cheapest tier → fewest active_tasks → AgentId tiebreaker.
pub struct LeastLoadedPolicy;

impl SchedulingPolicy for LeastLoadedPolicy {
    fn select(&self, candidates: &[AgentCandidate]) -> Option<AgentId> {
        if candidates.is_empty() {
            return None;
        }
        let mut sorted: Vec<&AgentCandidate> = candidates.iter().collect();
        sorted.sort_by(|a, b| {
            a.descriptor
                .cost_tier
                .cmp(&b.descriptor.cost_tier)
                .then(a.active_tasks.cmp(&b.active_tasks))
                .then(a.agent_id.to_string().cmp(&b.agent_id.to_string()))
        });
        Some(sorted[0].agent_id.clone())
    }
}

/// Cycles through eligible candidates by sorted AgentId. Ignores cost and load.
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl Default for RoundRobinPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulingPolicy for RoundRobinPolicy {
    fn select(&self, candidates: &[AgentCandidate]) -> Option<AgentId> {
        if candidates.is_empty() {
            return None;
        }
        let mut sorted: Vec<&AgentCandidate> = candidates.iter().collect();
        sorted.sort_by(|a, b| a.agent_id.to_string().cmp(&b.agent_id.to_string()));
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % sorted.len();
        Some(sorted[idx].agent_id.clone())
    }
}
