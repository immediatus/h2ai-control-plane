use h2ai_provisioner::nats_provider::NatsAgentProvider;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_provisioner::scheduling::{AgentCandidate, LeastLoadedPolicy, RoundRobinPolicy, SchedulingPolicy};
use h2ai_types::agent::{AgentDescriptor, AgentTool, CostTier, TaskRequirements};
use h2ai_types::identity::AgentId;

fn candidate(id: &str, tier: CostTier, active: u32) -> AgentCandidate {
    AgentCandidate {
        agent_id: AgentId::from(id),
        descriptor: AgentDescriptor {
            model: "test".into(),
            tools: vec![],
            cost_tier: tier,
        },
        active_tasks: active,
    }
}

fn make_provider() -> NatsAgentProvider {
    NatsAgentProvider::new_test_only()
}

fn register(
    provider: &NatsAgentProvider,
    id: &AgentId,
    model: &str,
    tier: CostTier,
    tools: Vec<AgentTool>,
    active: u32,
) {
    provider.inject_registration(
        id.clone(),
        AgentDescriptor {
            model: model.into(),
            tools,
            cost_tier: tier,
        },
        active,
    );
}

#[tokio::test]
async fn selects_cheapest_capable_agent() {
    let p = make_provider();
    let cheap = AgentId::from("cheap");
    let expensive = AgentId::from("expensive");
    register(&p, &cheap, "haiku", CostTier::Low, vec![], 0);
    register(&p, &expensive, "opus", CostTier::High, vec![], 0);

    let req = TaskRequirements {
        max_cost_tier: CostTier::High,
        required_tools: vec![],
    };
    assert_eq!(p.select_agent(&req).await.unwrap(), cheap);
}

#[tokio::test]
async fn respects_cost_ceiling() {
    let p = make_provider();
    register(
        &p,
        &AgentId::from("cheap"),
        "haiku",
        CostTier::Low,
        vec![],
        0,
    );
    register(
        &p,
        &AgentId::from("mid"),
        "sonnet",
        CostTier::Mid,
        vec![],
        0,
    );

    let req = TaskRequirements {
        max_cost_tier: CostTier::Low,
        required_tools: vec![],
    };
    assert_eq!(p.select_agent(&req).await.unwrap(), AgentId::from("cheap"));
}

#[tokio::test]
async fn filters_by_required_tools() {
    let p = make_provider();
    register(
        &p,
        &AgentId::from("notool"),
        "haiku",
        CostTier::Low,
        vec![],
        0,
    );
    register(
        &p,
        &AgentId::from("hasTool"),
        "haiku",
        CostTier::Low,
        vec![AgentTool::Shell],
        0,
    );

    let req = TaskRequirements {
        max_cost_tier: CostTier::High,
        required_tools: vec![AgentTool::Shell],
    };
    assert_eq!(
        p.select_agent(&req).await.unwrap(),
        AgentId::from("hasTool")
    );
}

#[tokio::test]
async fn prefers_least_loaded_within_same_tier() {
    let p = make_provider();
    register(
        &p,
        &AgentId::from("busy"),
        "haiku",
        CostTier::Low,
        vec![],
        5,
    );
    register(
        &p,
        &AgentId::from("idle"),
        "haiku",
        CostTier::Low,
        vec![],
        0,
    );

    let req = TaskRequirements {
        max_cost_tier: CostTier::High,
        required_tools: vec![],
    };
    assert_eq!(p.select_agent(&req).await.unwrap(), AgentId::from("idle"));
}

#[tokio::test]
async fn returns_error_when_no_capable_agent() {
    let p = make_provider();
    register(&p, &AgentId::from("a"), "haiku", CostTier::Low, vec![], 0);

    let req = TaskRequirements {
        max_cost_tier: CostTier::High,
        required_tools: vec![AgentTool::CodeExecution],
    };
    assert!(p.select_agent(&req).await.is_err());
}

#[tokio::test]
async fn returns_error_when_all_exceed_cost_ceiling() {
    let p = make_provider();
    register(&p, &AgentId::from("a"), "opus", CostTier::High, vec![], 0);

    let req = TaskRequirements {
        max_cost_tier: CostTier::Low,
        required_tools: vec![],
    };
    assert!(p.select_agent(&req).await.is_err());
}

#[tokio::test]
async fn cost_tier_priority_beats_load() {
    // The most critical invariant: even a fully loaded cheap agent beats an idle expensive one.
    let p = make_provider();
    register(
        &p,
        &AgentId::from("low-busy"),
        "haiku",
        CostTier::Low,
        vec![],
        99,
    );
    register(
        &p,
        &AgentId::from("high-idle"),
        "opus",
        CostTier::High,
        vec![],
        0,
    );

    let req = TaskRequirements {
        max_cost_tier: CostTier::High,
        required_tools: vec![],
    };
    let selected = p.select_agent(&req).await.unwrap();
    assert_eq!(
        selected,
        AgentId::from("low-busy"),
        "cost tier must take absolute priority over active task count"
    );
}

#[test]
fn least_loaded_policy_empty_returns_none() {
    let policy = LeastLoadedPolicy;
    assert!(policy.select(&[]).is_none());
}

#[test]
fn least_loaded_policy_single_candidate() {
    let policy = LeastLoadedPolicy;
    let candidates = vec![candidate("only", CostTier::Low, 0)];
    assert_eq!(policy.select(&candidates).unwrap(), AgentId::from("only"));
}

#[test]
fn least_loaded_policy_identical_load_uses_id_tiebreaker() {
    let policy = LeastLoadedPolicy;
    let candidates = vec![
        candidate("zzz", CostTier::Low, 5),
        candidate("aaa", CostTier::Low, 5),
    ];
    // Same tier and load — tiebreaker is AgentId string order → "aaa" wins.
    assert_eq!(policy.select(&candidates).unwrap(), AgentId::from("aaa"));
}

#[test]
fn round_robin_policy_empty_returns_none() {
    let policy = RoundRobinPolicy::new();
    assert!(policy.select(&[]).is_none());
}

#[test]
fn round_robin_policy_single_candidate_always_same() {
    let policy = RoundRobinPolicy::new();
    let candidates = vec![candidate("solo", CostTier::Low, 0)];
    assert_eq!(policy.select(&candidates).unwrap(), AgentId::from("solo"));
    assert_eq!(policy.select(&candidates).unwrap(), AgentId::from("solo"));
}

#[tokio::test]
async fn round_robin_policy_cycles() {
    use h2ai_provisioner::scheduling::RoundRobinPolicy;
    use std::sync::Arc;

    let p = NatsAgentProvider::new_test_only().with_policy(Arc::new(RoundRobinPolicy::new()));
    register(
        &p,
        &AgentId::from("agent-a"),
        "haiku",
        CostTier::Low,
        vec![],
        0,
    );
    register(
        &p,
        &AgentId::from("agent-b"),
        "haiku",
        CostTier::Low,
        vec![],
        0,
    );

    let req = TaskRequirements {
        max_cost_tier: CostTier::High,
        required_tools: vec![],
    };
    let first = p.select_agent(&req).await.unwrap();
    let second = p.select_agent(&req).await.unwrap();
    assert_ne!(first, second, "round-robin should alternate");
}
