#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use chrono::Utc;
use h2ai_state::semilattice::{ProposalSet, SemilatticeResult};
use h2ai_types::config::AdapterKind;
use h2ai_types::events::{BranchPrunedEvent, ProposalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::{RoleErrorCost, TauValue};

fn cloud() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "https://x.com".into(),
        api_key_env: "K".into(),
        model: None,
        provider: Default::default(),
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
        raw_output: String::new(),
        constraint_error_cost: RoleErrorCost::new(0.85).unwrap(),
        violated_constraints: vec![],
        timestamp: Utc::now(),
    }
}

#[test]
fn join_is_idempotent() {
    let tid = TaskId::new();
    let eid = ExplorerId::new();
    let p = proposal(eid, tid, "out");
    let mut set = ProposalSet::new();
    set.insert(p.clone());
    set.insert(p);
    assert_eq!(set.len(), 1);
}

#[test]
fn join_is_commutative() {
    let tid = TaskId::new();
    let e1 = ExplorerId::new();
    let e2 = ExplorerId::new();
    let p1 = proposal(e1, tid.clone(), "out1");
    let p2 = proposal(e2, tid, "out2");

    let mut set_ab = ProposalSet::new();
    set_ab.insert(p1.clone());
    set_ab.insert(p2.clone());

    let mut set_ba = ProposalSet::new();
    set_ba.insert(p2);
    set_ba.insert(p1);

    assert_eq!(set_ab.len(), set_ba.len());
}

#[test]
fn join_includes_all_distinct_explorers() {
    let tid = TaskId::new();
    let mut set = ProposalSet::new();
    set.insert(proposal(ExplorerId::new(), tid.clone(), "a"));
    set.insert(proposal(ExplorerId::new(), tid.clone(), "b"));
    set.insert(proposal(ExplorerId::new(), tid, "c"));
    assert_eq!(set.len(), 3);
}

#[test]
fn semilattice_result_valid_proposals_excludes_pruned() {
    let tid = TaskId::new();
    let e1 = ExplorerId::new();
    let e2 = ExplorerId::new();
    let mut proposals = ProposalSet::new();
    proposals.insert_scored(proposal(e1, tid.clone(), "out1"), 0.8);
    proposals.insert_scored(proposal(e2.clone(), tid.clone(), "out2"), 0.6);

    let pruned_list = vec![pruned(e2, tid.clone())];
    let result = SemilatticeResult::compile(tid, proposals, pruned_list);

    assert_eq!(result.valid_proposals.len(), 1);
    assert_eq!(result.pruned_proposals.len(), 1);
    assert_eq!(
        result.valid_proposal_scores.len(),
        result.valid_proposals.len(),
        "scores must be parallel to valid_proposals"
    );
    assert!(
        result.valid_proposal_scores.iter().all(|&s| s > 0.0),
        "all valid scores must be > 0.0"
    );
}

#[test]
fn semilattice_result_empty_when_all_pruned() {
    let tid = TaskId::new();
    let e1 = ExplorerId::new();
    let mut proposals = ProposalSet::new();
    proposals.insert_scored(proposal(e1.clone(), tid.clone(), "out1"), 0.9);

    let pruned_list = vec![pruned(e1, tid.clone())];
    let result = SemilatticeResult::compile(tid, proposals, pruned_list);

    assert!(result.valid_proposals.is_empty());
    assert!(
        result.valid_proposal_scores.is_empty(),
        "no valid proposals → no scores"
    );
    assert_eq!(result.pruned_proposals.len(), 1);
}

#[test]
fn semilattice_result_zero_score_goes_to_failed_not_valid() {
    let tid = TaskId::new();
    let e1 = ExplorerId::new();
    let e2 = ExplorerId::new();
    let mut proposals = ProposalSet::new();
    proposals.insert_scored(proposal(e1, tid.clone(), "passing"), 0.8);
    proposals.insert_scored(proposal(e2, tid.clone(), "failing"), 0.0);

    let result = SemilatticeResult::compile(tid, proposals, vec![]);

    assert_eq!(
        result.valid_proposals.len(),
        1,
        "only score>0 proposals feed selection"
    );
    assert_eq!(
        result.failed_proposals.len(),
        1,
        "score=0.0 goes to failed_proposals"
    );
    assert_eq!(result.valid_proposals[0].raw_output, "passing");
    assert_eq!(result.failed_proposals[0].raw_output, "failing");
    assert_eq!(
        result.valid_proposal_scores.len(),
        result.valid_proposals.len(),
        "scores must be parallel to valid_proposals"
    );
    assert!(
        (result.valid_proposal_scores[0] - 0.8).abs() < 1e-9,
        "valid score for 'passing' must be 0.8"
    );
}

#[test]
fn semilattice_result_all_zero_score_yields_empty_valid() {
    let tid = TaskId::new();
    let mut proposals = ProposalSet::new();
    proposals.insert_scored(proposal(ExplorerId::new(), tid.clone(), "fail1"), 0.0);
    proposals.insert_scored(proposal(ExplorerId::new(), tid.clone(), "fail2"), 0.0);

    let result = SemilatticeResult::compile(tid, proposals, vec![]);

    assert!(
        result.valid_proposals.is_empty(),
        "no valid proposals → ZeroSurvival path"
    );
    assert!(
        result.valid_proposal_scores.is_empty(),
        "no valid proposals → no scores"
    );
    assert_eq!(result.failed_proposals.len(), 2);
}

#[test]
fn insert_scored_same_explorer_first_value_wins() {
    // ProposalSet is keyed by explorer_id: inserting twice keeps the first entry.
    let tid = TaskId::new();
    let e = ExplorerId::new();
    let mut set = ProposalSet::new();
    set.insert_scored(proposal(e.clone(), tid.clone(), "first output"), 0.9);
    set.insert_scored(proposal(e, tid.clone(), "second output"), 0.1);
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
    set.insert(proposal(ExplorerId::new(), tid, "b"));
    assert_eq!(set.len(), 2);
}

#[test]
fn semilattice_result_empty_proposals_with_no_pruned() {
    let result = SemilatticeResult::compile(TaskId::new(), ProposalSet::new(), vec![]);
    assert!(result.valid_proposals.is_empty());
    assert!(
        result.valid_proposal_scores.is_empty(),
        "no valid proposals → no scores"
    );
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
    assert_eq!(
        result.valid_proposal_scores.len(),
        result.valid_proposals.len(),
        "scores must be parallel to valid_proposals"
    );
    // First must be highest score.
    assert_eq!(
        result.valid_proposals[0].raw_output, "high",
        "valid_proposals must be sorted by score descending"
    );
    assert!(
        (result.valid_proposal_scores[0] - 0.9).abs() < 1e-9,
        "first score must be 0.9"
    );
    assert!(
        (result.valid_proposal_scores[2] - 0.2).abs() < 1e-9,
        "last score must be 0.2"
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
        task_id: tid,
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
fn join_semilattice_idempotent_via_join_method() {
    // Tests ProposalSet::join(S, S) = S (idempotency using the join() method,
    // distinct from insert-based idempotency tested in join_is_idempotent).
    let tid = TaskId::new();
    let eid = ExplorerId::new();
    let p = proposal(eid, tid.clone(), "proposal text");

    let mut s1 = ProposalSet::new();
    s1.insert_scored(p.clone(), 0.7);
    let mut s2 = ProposalSet::new();
    s2.insert_scored(p, 0.7);

    let joined = ProposalSet::join(s1, s2);
    let result = SemilatticeResult::compile(tid, joined, vec![]);
    assert_eq!(
        result.valid_proposals.len(),
        1,
        "join(S, S) = S (idempotent)"
    );
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
        task_id: tid,
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
