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

/// Routes to the lowest cost tier that has at least one agent below `spillover_threshold`.
/// When all tiers are at or above the threshold, falls back to globally least-loaded.
pub struct CostAwareSpilloverPolicy {
    pub spillover_threshold: usize,
}

impl SchedulingPolicy for CostAwareSpilloverPolicy {
    fn select(&self, candidates: &[AgentCandidate]) -> Option<AgentId> {
        if candidates.is_empty() {
            return None;
        }

        let mut tiers: Vec<_> = candidates
            .iter()
            .map(|c| c.descriptor.cost_tier.clone())
            .collect();
        tiers.sort();
        tiers.dedup();

        for tier in &tiers {
            let tier_agents: Vec<_> = candidates
                .iter()
                .filter(|c| &c.descriptor.cost_tier == tier)
                .collect();
            let min_load = tier_agents
                .iter()
                .map(|c| c.active_tasks)
                .min()
                .unwrap_or(u32::MAX);
            if (min_load as usize) < self.spillover_threshold {
                return tier_agents
                    .iter()
                    .filter(|c| c.active_tasks == min_load)
                    .min_by(|a, b| a.agent_id.to_string().cmp(&b.agent_id.to_string()))
                    .map(|c| c.agent_id.clone());
            }
        }

        // All tiers saturated: fall back to globally least loaded
        let min_load = candidates.iter().map(|c| c.active_tasks).min().unwrap_or(0);
        candidates
            .iter()
            .filter(|c| c.active_tasks == min_load)
            .min_by(|a, b| a.agent_id.to_string().cmp(&b.agent_id.to_string()))
            .map(|c| c.agent_id.clone())
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
