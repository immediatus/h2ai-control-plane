use h2ai_types::config::AgentRole;
use h2ai_types::knowledge::{profile_for_role, RetrievalMode};

#[test]
fn coordinator_profile() {
    let p = profile_for_role(&AgentRole::Coordinator);
    assert_eq!(p.mode, RetrievalMode::CollapsedTree);
    assert_eq!(p.expand_hops, 0);
    assert_eq!(p.top_k, 3);
    assert!(!p.domain_tag_boost);
    assert!(p.explicit_ids.is_empty());
}

#[test]
fn executor_profile() {
    let p = profile_for_role(&AgentRole::Executor);
    assert_eq!(p.mode, RetrievalMode::TreeTraversal);
    assert_eq!(p.expand_hops, 2);
    assert_eq!(p.top_k, 5);
    assert!(p.domain_tag_boost);
    assert!(p.explicit_ids.is_empty());
}

#[test]
fn evaluator_profile() {
    let p = profile_for_role(&AgentRole::Evaluator);
    assert_eq!(p.mode, RetrievalMode::TreeTraversal);
    assert_eq!(p.expand_hops, 0);
    assert_eq!(p.top_k, 4);
    assert!(p.domain_tag_boost);
}

#[test]
fn synthesizer_profile() {
    let p = profile_for_role(&AgentRole::Synthesizer);
    assert_eq!(p.mode, RetrievalMode::CollapsedTree);
    assert_eq!(p.expand_hops, 1);
    assert_eq!(p.top_k, 5);
    assert!(!p.domain_tag_boost);
}

#[test]
fn custom_role_defaults_to_executor_profile() {
    use h2ai_types::sizing::TauValue;
    let p = profile_for_role(&AgentRole::Custom {
        name: "specialist".into(),
        tau: TauValue::new(0.5).unwrap(),
        role_error_cost: 0.5,
    });
    assert_eq!(p.mode, RetrievalMode::TreeTraversal);
    assert_eq!(p.expand_hops, 2);
}
