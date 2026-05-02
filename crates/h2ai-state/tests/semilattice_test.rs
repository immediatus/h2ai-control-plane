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
        generation: 0,
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
        violated_constraints: vec![],
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

#[test]
fn insert_scored_same_explorer_first_value_wins() {
    // ProposalSet is keyed by explorer_id: inserting twice keeps the first entry.
    let tid = TaskId::new();
    let e = ExplorerId::new();
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(e.clone(), tid.clone(), "first output"), 0.9);
    set.insert_scored(proposal(e.clone(), tid.clone(), "second output"), 0.1);
    // Only one entry should exist and it must be the first.
    assert_eq!(
        set.len(),
        1,
        "duplicate explorer_id must not add a second entry"
    );
    let result = SemilatticeResult::compile(tid, set, vec![]);
    assert_eq!(result.valid_proposals.len(), 1);
    assert_eq!(
        result.valid_proposals[0].raw_output, "first output",
        "first insertion wins — or_insert semantics"
    );
}

#[test]
fn proposal_set_len_and_is_empty() {
    let tid = TaskId::new();
    let mut set = ProposalSet::new();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
    set.insert(proposal(ExplorerId::new(), tid.clone(), "a"));
    assert!(!set.is_empty());
    assert_eq!(set.len(), 1);
    set.insert(proposal(ExplorerId::new(), tid.clone(), "b"));
    assert_eq!(set.len(), 2);
}

#[test]
fn semilattice_result_empty_proposals_with_no_pruned() {
    let result = SemilatticeResult::compile(TaskId::new(), ProposalSet::new(), vec![]);
    assert!(result.valid_proposals.is_empty());
    assert!(result.pruned_proposals.is_empty());
}

#[test]
fn semilattice_valid_proposals_sorted_by_score_descending() {
    // ScoreOrdered merge depends on first element having the highest score.
    let tid = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(ExplorerId::new(), tid.clone(), "low"), 0.2);
    set.insert_scored(proposal(ExplorerId::new(), tid.clone(), "high"), 0.9);
    set.insert_scored(proposal(ExplorerId::new(), tid.clone(), "mid"), 0.5);
    let result = SemilatticeResult::compile(tid, set, vec![]);
    assert_eq!(result.valid_proposals.len(), 3);
    // First must be highest score.
    assert_eq!(
        result.valid_proposals[0].raw_output, "high",
        "valid_proposals must be sorted by score descending"
    );
    assert_eq!(result.valid_proposals[2].raw_output, "low");
}

#[test]
fn crdt_higher_generation_supersedes_higher_score() {
    // P6 fix: TAO retry loop produces gen=1 with score 0.5 — must supersede gen=0 score 0.9.
    let tid = TaskId::new();
    let eid = ExplorerId::new();

    let p0 = ProposalEvent {
        task_id: tid.clone(),
        explorer_id: eid.clone(),
        tau: TauValue::new(0.5).unwrap(),
        generation: 0,
        raw_output: "gen0_high_score".into(),
        token_cost: 1,
        adapter_kind: cloud(),
        timestamp: Utc::now(),
    };
    let p1 = ProposalEvent {
        task_id: tid.clone(),
        explorer_id: eid.clone(),
        tau: TauValue::new(0.5).unwrap(),
        generation: 1,
        raw_output: "gen1_low_score".into(),
        token_cost: 1,
        adapter_kind: cloud(),
        timestamp: Utc::now(),
    };

    let mut set = ProposalSet::new();
    set.insert_scored(p0, 0.9);
    set.insert_scored(p1, 0.5);

    let entry = set.get(&eid).expect("explorer must be present");
    assert_eq!(
        entry.raw_output, "gen1_low_score",
        "higher generation must win regardless of score"
    );
    assert_eq!(entry.generation, 1);
}

#[test]
fn crdt_same_generation_higher_score_wins() {
    let tid = TaskId::new();
    let eid = ExplorerId::new();

    let pa = ProposalEvent {
        task_id: tid.clone(),
        explorer_id: eid.clone(),
        tau: TauValue::new(0.5).unwrap(),
        generation: 2,
        raw_output: "gen2_low".into(),
        token_cost: 1,
        adapter_kind: cloud(),
        timestamp: Utc::now(),
    };
    let pb = ProposalEvent {
        task_id: tid.clone(),
        explorer_id: eid.clone(),
        tau: TauValue::new(0.5).unwrap(),
        generation: 2,
        raw_output: "gen2_high".into(),
        token_cost: 1,
        adapter_kind: cloud(),
        timestamp: Utc::now(),
    };

    let mut set = ProposalSet::new();
    set.insert_scored(pa, 0.3);
    set.insert_scored(pb, 0.8);

    let entry = set.get(&eid).expect("explorer must be present");
    assert_eq!(
        entry.raw_output, "gen2_high",
        "within same generation, higher score must win"
    );
}
