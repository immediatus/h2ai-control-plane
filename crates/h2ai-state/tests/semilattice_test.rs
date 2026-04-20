use chrono::Utc;
use h2ai_state::semilattice::{ProposalSet, SemilatticeResult};
use h2ai_types::config::AdapterKind;
use h2ai_types::events::{BranchPrunedEvent, ProposalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::{RoleErrorCost, TauValue};

fn cloud() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://x.com".into(),
        api_key_env: "K".into(),
    }
}

fn proposal(explorer_id: ExplorerId, task_id: TaskId, output: &str) -> ProposalEvent {
    ProposalEvent {
        task_id,
        explorer_id,
        tau: TauValue::new(0.5).unwrap(),
        raw_output: output.into(),
        token_cost: 10,
        adapter_kind: cloud(),
        timestamp: Utc::now(),
    }
}

fn pruned(explorer_id: ExplorerId, task_id: TaskId) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id,
        explorer_id,
        reason: "ADR-004".into(),
        constraint_error_cost: RoleErrorCost::new(0.85).unwrap(),
        timestamp: Utc::now(),
    }
}

#[test]
fn join_is_idempotent() {
    let tid = TaskId::new();
    let eid = ExplorerId::new();
    let p = proposal(eid.clone(), tid.clone(), "out");
    let mut set = ProposalSet::new();
    set.insert(p.clone());
    set.insert(p.clone());
    assert_eq!(set.len(), 1);
}

#[test]
fn join_is_commutative() {
    let tid = TaskId::new();
    let e1 = ExplorerId::new();
    let e2 = ExplorerId::new();
    let p1 = proposal(e1.clone(), tid.clone(), "out1");
    let p2 = proposal(e2.clone(), tid.clone(), "out2");

    let mut set_ab = ProposalSet::new();
    set_ab.insert(p1.clone());
    set_ab.insert(p2.clone());

    let mut set_ba = ProposalSet::new();
    set_ba.insert(p2.clone());
    set_ba.insert(p1.clone());

    assert_eq!(set_ab.len(), set_ba.len());
}

#[test]
fn join_includes_all_distinct_explorers() {
    let tid = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(ExplorerId::new(), tid.clone(), "a"));
    set.insert(proposal(ExplorerId::new(), tid.clone(), "b"));
    set.insert(proposal(ExplorerId::new(), tid.clone(), "c"));
    assert_eq!(set.len(), 3);
}

#[test]
fn semilattice_result_valid_proposals_excludes_pruned() {
    let tid = TaskId::new();
    let e1 = ExplorerId::new();
    let e2 = ExplorerId::new();
    let mut proposals = ProposalSet::new();
    proposals.insert(proposal(e1.clone(), tid.clone(), "out1"));
    proposals.insert(proposal(e2.clone(), tid.clone(), "out2"));

    let pruned_list = vec![pruned(e2.clone(), tid.clone())];
    let result = SemilatticeResult::compile(tid.clone(), proposals, pruned_list);

    assert_eq!(result.valid_proposals.len(), 1);
    assert_eq!(result.pruned_proposals.len(), 1);
}

#[test]
fn semilattice_result_empty_when_all_pruned() {
    let tid = TaskId::new();
    let e1 = ExplorerId::new();
    let mut proposals = ProposalSet::new();
    proposals.insert(proposal(e1.clone(), tid.clone(), "out1"));

    let pruned_list = vec![pruned(e1.clone(), tid.clone())];
    let result = SemilatticeResult::compile(tid.clone(), proposals, pruned_list);

    assert!(result.valid_proposals.is_empty());
    assert_eq!(result.pruned_proposals.len(), 1);
}
