use chrono::Utc;
use h2ai_autonomic::merger::{MergeEngine, MergeOutcome};
use h2ai_state::semilattice::ProposalSet;
use h2ai_types::config::AdapterKind;
use h2ai_types::events::{BranchPrunedEvent, ProposalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::{MergeStrategy, RoleErrorCost, TauValue};

fn adapter() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://api.test".into(),
        api_key_env: "K".into(),
    }
}

fn proposal(task_id: &TaskId, explorer_id: ExplorerId, output: &str, cost: u64) -> ProposalEvent {
    ProposalEvent {
        task_id: task_id.clone(),
        explorer_id,
        tau: TauValue::new(0.4).unwrap(),
        raw_output: output.into(),
        token_cost: cost,
        adapter_kind: adapter(),
        timestamp: Utc::now(),
    }
}

fn pruned(task_id: &TaskId, explorer_id: &ExplorerId) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: task_id.clone(),
        explorer_id: explorer_id.clone(),
        reason: "ADR violation".into(),
        constraint_error_cost: RoleErrorCost::new(0.9).unwrap(),
        timestamp: Utc::now(),
    }
}

#[test]
fn merge_engine_resolves_crdt_when_valid_proposals_exist() {
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(&task_id, ExplorerId::new(), "answer A", 10));
    set.insert(proposal(&task_id, ExplorerId::new(), "answer B", 20));

    let outcome = MergeEngine::resolve(task_id, set, vec![], MergeStrategy::CrdtSemilattice, 0);
    assert!(matches!(outcome, MergeOutcome::Resolved { .. }));
}

#[test]
fn merge_engine_emits_zero_survival_when_all_pruned() {
    let task_id = TaskId::new();
    let explorer_id = ExplorerId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(&task_id, explorer_id.clone(), "output", 5));
    let pruned_events = vec![pruned(&task_id, &explorer_id)];

    let outcome = MergeEngine::resolve(
        task_id,
        set,
        pruned_events,
        MergeStrategy::CrdtSemilattice,
        0,
    );
    assert!(matches!(outcome, MergeOutcome::ZeroSurvival(_)));
}

#[test]
fn merge_engine_zero_survival_when_proposal_set_empty() {
    let task_id = TaskId::new();
    let outcome = MergeEngine::resolve(
        task_id,
        ProposalSet::new(),
        vec![],
        MergeStrategy::CrdtSemilattice,
        0,
    );
    assert!(matches!(outcome, MergeOutcome::ZeroSurvival(_)));
}

#[test]
fn merge_engine_bft_picks_min_token_cost_proposal() {
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(&task_id, ExplorerId::new(), "expensive", 100));
    set.insert(proposal(&task_id, ExplorerId::new(), "cheap", 10));

    let outcome = MergeEngine::resolve(task_id, set, vec![], MergeStrategy::BftConsensus, 0);
    if let MergeOutcome::Resolved { resolved, .. } = outcome {
        assert_eq!(resolved.resolved_output, "cheap");
    } else {
        panic!("expected Resolved");
    }
}

#[test]
fn merge_engine_resolved_outcome_carries_semilattice_compiled_event() {
    let task_id = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(&task_id, ExplorerId::new(), "output", 5));

    let outcome = MergeEngine::resolve(task_id, set, vec![], MergeStrategy::CrdtSemilattice, 0);
    if let MergeOutcome::Resolved { compiled, .. } = outcome {
        assert!(!compiled.valid_proposals.is_empty());
    } else {
        panic!("expected Resolved");
    }
}
