use chrono::Utc;
use h2ai_state::bft::ConsensusMedian;
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
        generation: 0,
        raw_output: output.into(),
        token_cost,
        adapter_kind: cloud(),
        timestamp: Utc::now(),
    }
}

#[tokio::test]
async fn condorcet_selects_proposal_most_similar_to_group() {
    let tid = TaskId::new();
    // Two proposals share vocabulary; one outlier does not.
    // Condorcet should pick one of the consensus pair, not the outlier.
    let a = proposal(
        tid.clone(),
        "stateless JWT auth token ADR-001 compliant",
        50,
    );
    let b = proposal(
        tid.clone(),
        "stateless authentication JWT ADR-001 token rotation",
        60,
    );
    let outlier = proposal(
        tid.clone(),
        "blockchain proof-of-work hash rainbow table",
        10,
    );
    let proposals = vec![a.clone(), b.clone(), outlier.clone()];
    let result = ConsensusMedian::resolve(&proposals, None).await.unwrap();
    assert!(
        result.raw_output == a.raw_output || result.raw_output == b.raw_output,
        "Condorcet should pick consensus proposal, not outlier; got: {}",
        result.raw_output
    );
}

#[tokio::test]
async fn condorcet_returns_none_when_no_proposals() {
    let result = ConsensusMedian::resolve(&[], None).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn condorcet_single_proposal_returns_it() {
    let tid = TaskId::new();
    let p = proposal(tid, "only", 200);
    let proposals = [p.clone()];
    let result = ConsensusMedian::resolve(&proposals, None).await.unwrap();
    assert_eq!(result.raw_output, "only");
}
