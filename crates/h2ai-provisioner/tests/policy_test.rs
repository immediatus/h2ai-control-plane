//! Direct unit tests for SchedulingPolicy implementations.
//! Tests the policy logic in isolation from NatsAgentProvider filtering.

use h2ai_provisioner::scheduling::{
    AgentCandidate, CostAwareSpilloverPolicy, LeastLoadedPolicy, RoundRobinPolicy, SchedulingPolicy,
};
use h2ai_types::agent::{AgentDescriptor, CostTier};
use h2ai_types::identity::AgentId;

fn candidate(id: &str, tier: CostTier, active: u32) -> AgentCandidate {
    AgentCandidate {
        agent_id: AgentId::from(id),
        descriptor: AgentDescriptor {
            model: id.into(),
            tools: vec![],
            cost_tier: tier,
        },
        active_tasks: active,
    }
}

// LeastLoadedPolicy tests

#[test]
fn least_loaded_prefers_cheaper_tier_over_lower_load() {
    // Critical invariant: cost tier takes absolute priority over load.
    // A Low-tier agent with 99 tasks beats a High-tier agent with 0 tasks.
    let candidates = vec![
        candidate("low-busy", CostTier::Low, 99),
        candidate("high-idle", CostTier::High, 0),
    ];
    let selected = LeastLoadedPolicy.select(&candidates).unwrap();
    assert_eq!(
        selected,
        AgentId::from("low-busy"),
        "cheapest tier must win regardless of load"
    );
}

#[test]
fn least_loaded_picks_least_loaded_within_same_tier() {
    let candidates = vec![
        candidate("a-busy", CostTier::Mid, 10),
        candidate("b-idle", CostTier::Mid, 0),
        candidate("c-mid", CostTier::Mid, 5),
    ];
    let selected = LeastLoadedPolicy.select(&candidates).unwrap();
    assert_eq!(
        selected,
        AgentId::from("b-idle"),
        "should pick least loaded when tiers are equal"
    );
}

#[test]
fn least_loaded_uses_agent_id_as_tiebreaker() {
    // When tier and load are identical, pick lexicographically smallest AgentId.
    let candidates = vec![
        candidate("beta", CostTier::Low, 0),
        candidate("alpha", CostTier::Low, 0),
        candidate("gamma", CostTier::Low, 0),
    ];
    let selected = LeastLoadedPolicy.select(&candidates).unwrap();
    assert_eq!(
        selected,
        AgentId::from("alpha"),
        "should use AgentId as stable tiebreaker"
    );
}

#[test]
fn least_loaded_returns_none_for_empty_candidates() {
    assert!(LeastLoadedPolicy.select(&[]).is_none());
}

#[test]
fn least_loaded_single_candidate_always_selected() {
    let candidates = vec![candidate("only", CostTier::High, 42)];
    assert_eq!(
        LeastLoadedPolicy.select(&candidates).unwrap(),
        AgentId::from("only")
    );
}

// RoundRobinPolicy tests

#[test]
fn round_robin_alternates_between_two() {
    let policy = RoundRobinPolicy::new();
    let candidates = vec![
        candidate("agent-a", CostTier::Low, 0),
        candidate("agent-b", CostTier::Low, 0),
    ];
    let first = policy.select(&candidates).unwrap();
    let second = policy.select(&candidates).unwrap();
    let third = policy.select(&candidates).unwrap();

    assert_ne!(first, second, "should alternate");
    assert_eq!(first, third, "should cycle back to first after two calls");
}

#[test]
fn round_robin_is_deterministic_within_same_order() {
    // Two fresh policies should make the same first selection given identical input.
    let p1 = RoundRobinPolicy::new();
    let p2 = RoundRobinPolicy::new();
    let candidates = vec![
        candidate("agent-b", CostTier::Low, 0),
        candidate("agent-a", CostTier::Low, 0),
    ];
    assert_eq!(
        p1.select(&candidates),
        p2.select(&candidates),
        "same input must produce same selection from a fresh policy"
    );
}

#[test]
fn round_robin_returns_none_for_empty_candidates() {
    assert!(RoundRobinPolicy::new().select(&[]).is_none());
}

// CostAwareSpilloverPolicy tests

fn spillover(threshold: usize) -> CostAwareSpilloverPolicy {
    CostAwareSpilloverPolicy {
        spillover_threshold: threshold,
    }
}

#[test]
fn spillover_prefers_tier1_when_under_threshold() {
    let candidates = vec![
        candidate("low", CostTier::Low, 5),
        candidate("high", CostTier::High, 0),
    ];
    let selected = spillover(10).select(&candidates).unwrap();
    assert_eq!(
        selected,
        AgentId::from("low"),
        "tier-1 under threshold must win"
    );
}

#[test]
fn spillover_spills_to_tier2_when_tier1_saturated() {
    let candidates = vec![
        candidate("low-busy", CostTier::Low, 10),
        candidate("high-idle", CostTier::High, 0),
    ];
    let selected = spillover(10).select(&candidates).unwrap();
    assert_eq!(
        selected,
        AgentId::from("high-idle"),
        "saturated tier-1 must spill to tier-2"
    );
}

#[test]
fn spillover_all_tiers_saturated_falls_back_to_least_loaded() {
    let candidates = vec![
        candidate("low-busy", CostTier::Low, 15),
        candidate("high-less-busy", CostTier::High, 12),
    ];
    let selected = spillover(10).select(&candidates).unwrap();
    assert_eq!(
        selected,
        AgentId::from("high-less-busy"),
        "all saturated → globally least loaded"
    );
}

#[test]
fn spillover_empty_returns_none() {
    assert!(spillover(10).select(&[]).is_none());
}

#[test]
fn spillover_single_candidate_always_selected() {
    let candidates = vec![candidate("only", CostTier::High, 99)];
    assert_eq!(
        spillover(5).select(&candidates).unwrap(),
        AgentId::from("only")
    );
}

#[test]
fn spillover_picks_least_loaded_within_winning_tier() {
    let candidates = vec![
        candidate("low-busy", CostTier::Low, 5),
        candidate("low-idle", CostTier::Low, 1),
        candidate("high-idle", CostTier::High, 0),
    ];
    let selected = spillover(10).select(&candidates).unwrap();
    assert_eq!(
        selected,
        AgentId::from("low-idle"),
        "pick least loaded within winning tier"
    );
}
