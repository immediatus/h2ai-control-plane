use chrono::Utc;
use h2ai_state::bft::BftConsensus;
use h2ai_types::config::AdapterKind;
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::TauValue;

fn cloud() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://x.com".into(),
        api_key_env: "K".into(),
    }
}

fn proposal(task_id: TaskId, output: &str, token_cost: u64) -> ProposalEvent {
    ProposalEvent {
        task_id,
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.1).unwrap(),
        raw_output: output.into(),
        token_cost,
        adapter_kind: cloud(),
        timestamp: Utc::now(),
    }
}

#[test]
fn bft_selects_proposal_with_lowest_token_cost() {
    let tid = TaskId::new();
    let proposals = vec![
        proposal(tid.clone(), "expensive", 500),
        proposal(tid.clone(), "cheap", 100),
        proposal(tid.clone(), "mid", 300),
    ];
    let result = BftConsensus::resolve(&proposals).unwrap();
    assert_eq!(result.raw_output, "cheap");
}

#[test]
fn bft_returns_none_when_no_proposals() {
    let result = BftConsensus::resolve(&[]);
    assert!(result.is_none());
}

#[test]
fn bft_single_proposal_returns_it() {
    let tid = TaskId::new();
    let p = proposal(tid, "only", 200);
    let result = BftConsensus::resolve(std::slice::from_ref(&p)).unwrap();
    assert_eq!(result.raw_output, "only");
}
