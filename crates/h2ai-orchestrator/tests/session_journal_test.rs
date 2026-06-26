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
use async_trait::async_trait;
use chrono::Utc;
use futures::stream::BoxStream;
use h2ai_orchestrator::session_journal::SessionJournal;
use h2ai_orchestrator::task_store::{TaskPhase, TaskState};
use h2ai_state::backend::{SnapshotStore, TailEvents};
use h2ai_state::in_memory::InMemoryStateBackend;
use h2ai_state::nats::NatsError;
use h2ai_types::config::{AdapterKind, AuditorConfig, ParetoWeights, TopologyKind};
use h2ai_types::events::{
    BranchPrunedEvent, H2AIEvent, LeaderElectedEvent, MergeResolvedEvent,
    MultiplicationConditionFailedEvent, ProposalEvent, SelectionResolvedEvent,
    TaskBootstrappedEvent, TaskComplexityAssessedEvent, TaskFailedEvent, TaskSnapshot,
    TerminalCause, TopologyProvisionedEvent, VerificationScoredEvent, ZeroSurvivalEvent,
};
use h2ai_types::identity::{ExplorerId, TaskId, TenantId};
use h2ai_types::sizing::{
    MergeStrategy, MultiplicationConditionFailure, ProbeSkipReason, RoleErrorCost, TaskQuadrant,
    TauValue,
};
use std::sync::Arc;

// ── Error-injecting backends (generics, no dyn dispatch) ─────────────────────

/// Backend whose `get_snapshot` always returns `Err` (tests snapshot-error fallback path).
struct SnapshotErrBackend;

#[async_trait]
impl SnapshotStore for SnapshotErrBackend {
    async fn put_snapshot(&self, _: &TaskSnapshot) -> Result<(), NatsError> {
        Err(NatsError::KvError("mock put error".into()))
    }
    async fn get_snapshot(&self, _: &TaskId) -> Result<Option<TaskSnapshot>, NatsError> {
        Err(NatsError::KvError("mock get error".into()))
    }
}

#[async_trait]
impl TailEvents for SnapshotErrBackend {
    async fn tail_task_events_boxed(
        &self,
        _: &TaskId,
        _: u64,
    ) -> Result<BoxStream<'static, Result<(u64, H2AIEvent), NatsError>>, NatsError> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// Backend whose `tail_task_events_boxed` always returns `Err` (tests stream-error path).
struct StreamErrBackend;

#[async_trait]
impl SnapshotStore for StreamErrBackend {
    async fn put_snapshot(&self, _: &TaskSnapshot) -> Result<(), NatsError> {
        Ok(())
    }
    async fn get_snapshot(&self, _: &TaskId) -> Result<Option<TaskSnapshot>, NatsError> {
        Ok(None)
    }
}

#[async_trait]
impl TailEvents for StreamErrBackend {
    async fn tail_task_events_boxed(
        &self,
        _: &TaskId,
        _: u64,
    ) -> Result<BoxStream<'static, Result<(u64, H2AIEvent), NatsError>>, NatsError> {
        Err(NatsError::StreamError("mock stream error".into()))
    }
}

/// Backend that returns a snapshot with invalid JSON (tests bad-snapshot fallback).
struct BadSnapshotBackend {
    task_id: TaskId,
}

#[async_trait]
impl SnapshotStore for BadSnapshotBackend {
    async fn put_snapshot(&self, _: &TaskSnapshot) -> Result<(), NatsError> {
        Ok(())
    }
    async fn get_snapshot(&self, _: &TaskId) -> Result<Option<TaskSnapshot>, NatsError> {
        Ok(Some(TaskSnapshot {
            task_id: self.task_id.clone(),
            last_sequence: 5,
            task_state_json: "{{{not valid json".into(),
            taken_at: Utc::now(),
        }))
    }
}

#[async_trait]
impl TailEvents for BadSnapshotBackend {
    async fn tail_task_events_boxed(
        &self,
        _: &TaskId,
        _: u64,
    ) -> Result<BoxStream<'static, Result<(u64, H2AIEvent), NatsError>>, NatsError> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn task_id() -> TaskId {
    TaskId::new()
}

fn explorer_id() -> ExplorerId {
    ExplorerId::new()
}

fn pareto_weights() -> ParetoWeights {
    ParetoWeights::new(0.33, 0.34, 0.33).unwrap()
}

// ── 1: TaskBootstrapped → pending / Bootstrap ────────────────────────────────

#[test]
fn apply_bootstrapped_sets_pending() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
            task_id: tid.clone(),
            system_context: "ctx".into(),
            pareto_weights: pareto_weights(),
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.status, "pending");
    assert_eq!(
        TaskPhase::try_from(state.phase).unwrap(),
        TaskPhase::Bootstrap
    );
    assert_eq!(state.phase_name, "Bootstrap");
}

// ── 2: Proposal → generating / ParallelGeneration ────────────────────────────

#[test]
fn apply_proposal_increments_completed_and_sets_generating() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    assert_eq!(state.explorers_completed, 0);
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::Proposal(ProposalEvent {
            task_id: tid.clone(),
            explorer_id: explorer_id(),
            tau: TauValue::new(0.5).unwrap(),
            generation: 0,
            raw_output: "output".into(),
            token_cost: 100,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: "http://example.com".into(),
                api_key_env: "KEY".into(),
                model: None,
                provider: Default::default(),
            },
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.explorers_completed, 1);
    assert_eq!(state.status, "generating");
    assert_eq!(
        TaskPhase::try_from(state.phase).unwrap(),
        TaskPhase::ParallelGeneration
    );
}

// ── 3: VerificationScored passed → proposals_valid += 1 ──────────────────────

#[test]
fn apply_verification_scored_passed_increments_valid() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    assert_eq!(state.proposals_valid, 0);
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::VerificationScored(VerificationScoredEvent {
            task_id: tid.clone(),
            explorer_id: explorer_id(),
            score: 0.9,
            reason: "good".into(),
            passed: true,
            cache_hit: false,
            passed_checks: None,
            total_checks: None,
            score_lower: None,
            score_upper: None,
            per_check_verdicts: vec![],
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.proposals_valid, 1);
    assert_eq!(state.proposals_pruned, 0);
}

// ── 4: VerificationScored failed → proposals_pruned += 1 ─────────────────────

#[test]
fn apply_verification_scored_failed_increments_pruned() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    assert_eq!(state.proposals_pruned, 0);
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::VerificationScored(VerificationScoredEvent {
            task_id: tid.clone(),
            explorer_id: explorer_id(),
            score: 0.1,
            reason: "bad".into(),
            passed: false,
            cache_hit: false,
            passed_checks: None,
            total_checks: None,
            score_lower: None,
            score_upper: None,
            per_check_verdicts: vec![],
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.proposals_pruned, 1);
    assert_eq!(state.proposals_valid, 0);
}

// ── 5: ZeroSurvival → autonomic_retries += 1 ─────────────────────────────────

#[test]
fn apply_zero_survival_increments_retries() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    assert_eq!(state.autonomic_retries, 0);
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::ZeroSurvival(ZeroSurvivalEvent {
            task_id: tid.clone(),
            retry_count: 1,
            timestamp: Utc::now(),
            n_eff_cosine_actual: None,
            failure_mode: None,
        }),
    );
    assert_eq!(state.autonomic_retries, 1);
}

// ── 6: MergeResolved → resolved / Resolved ───────────────────────────────────

#[test]
fn apply_merge_resolved_sets_resolved() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::MergeResolved(MergeResolvedEvent {
            task_id: tid.clone(),
            resolved_output: "final answer".into(),
            j_eff: None,
            oracle_gate_passed: None,
            timestamp: Utc::now(),
            zone3_hints: None,
            contradiction_analysis: None,
        }),
    );
    assert_eq!(state.status, "resolved");
    assert_eq!(
        TaskPhase::try_from(state.phase).unwrap(),
        TaskPhase::Resolved
    );
    assert_eq!(state.phase_name, "Resolved");
}

// ── 7: TaskFailed → failed / Failed ──────────────────────────────────────────

#[test]
fn apply_task_failed_sets_failed() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::TaskFailed(TaskFailedEvent {
            task_id: tid.clone(),
            primary_cause: TerminalCause::Unknown,
            contributing_causes: vec![],
            top_violated_constraints: vec![],
            last_selection_valid_count: None,
            pruned_events: vec![],
            topologies_tried: vec![TopologyKind::Ensemble],
            tau_values_tried: vec![],
            multiplication_condition_failure: None,
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.status, "failed");
    assert_eq!(TaskPhase::try_from(state.phase).unwrap(), TaskPhase::Failed);
    assert_eq!(state.phase_name, "Failed");
    assert_eq!(state.autonomic_retries, 1);
}

#[test]
fn apply_branch_pruned_increments_proposals_pruned() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::BranchPruned(BranchPrunedEvent {
            task_id: tid.clone(),
            explorer_id: explorer_id(),
            reason: "constraint violation".into(),
            raw_output: String::new(),
            constraint_error_cost: RoleErrorCost::new(0.8).unwrap(),
            violated_constraints: vec![],
            timestamp: Utc::now(),
            retry_count: 0,
            bypass_reason: None,
        }),
    );
    assert_eq!(state.proposals_pruned, 1);
    assert_eq!(state.proposals_valid, 0);
}

// ── TaskState serde ───────────────────────────────────────────────────────────

#[test]
fn task_state_serde_roundtrip() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    state.status = "generating".into();
    state.phase = TaskPhase::ParallelGeneration as u8;
    state.phase_name = "ParallelGeneration".into();
    state.explorers_completed = 3;
    state.explorers_total = 5;
    state.proposals_valid = 2;
    state.proposals_pruned = 1;
    state.autonomic_retries = 1;

    let json = serde_json::to_string(&state).unwrap();
    let back: TaskState = serde_json::from_str(&json).unwrap();
    assert_eq!(back.task_id, tid);
    assert_eq!(back.status, "generating");
    assert_eq!(back.explorers_completed, 3);
    assert_eq!(back.proposals_valid, 2);
    assert_eq!(back.autonomic_retries, 1);
}

// ── SessionJournal constructors ───────────────────────────────────────────────

#[test]
fn new_noop_creates_journal_without_nats() {
    let journal = SessionJournal::new_noop();
    let tid = task_id();
    let state = TaskState::new(tid.clone(), TenantId::default_tenant());
    journal.note_event(&tid, 1, &state);
}

#[test]
fn with_snapshot_interval_zero_note_event_noop() {
    let journal = SessionJournal::new_noop().with_snapshot_interval(0);
    let tid = task_id();
    let state = TaskState::new(tid.clone(), TenantId::default_tenant());
    journal.note_event(&tid, 1, &state);
}

#[test]
fn with_snapshot_interval_nonzero_note_event_no_panic_when_no_nats() {
    let journal = SessionJournal::new_noop().with_snapshot_interval(5);
    let tid = task_id();
    let state = TaskState::new(tid.clone(), TenantId::default_tenant());
    for i in 0u64..10 {
        journal.note_event(&tid, i, &state);
    }
}

#[tokio::test]
async fn replay_noop_returns_none() {
    let journal = SessionJournal::new_noop();
    let tid = task_id();
    let result = journal
        .replay(&tid)
        .await
        .expect("replay on noop must be Ok");
    assert!(result.is_none(), "noop journal has no NATS → always None");
}

// ── apply_event — previously uncovered variants ───────────────────────────────

#[test]
fn apply_topology_provisioned_sets_provisioning() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());

    let event: TopologyProvisionedEvent = serde_json::from_value(serde_json::json!({
        "task_id": tid.to_string(),
        "topology_kind": "Ensemble",
        "explorer_configs": [
            {
                "explorer_id": ExplorerId::new().to_string(),
                "tau": 0.5,
                "adapter": {"CloudGeneric": {"endpoint": "http://x", "api_key_env": "X", "model": null}},
                "role": null,
                "is_reasoning_model": false
            }
        ],
        "auditor_config": serde_json::to_value(AuditorConfig::default()).unwrap(),
        "n_max": 2.0,
        "interface_n_max": null,
        "beta_eff": 0.1,
        "role_error_costs": [],
        "merge_strategy": "ScoreOrdered",
        "coordination_threshold": 0.5,
        "review_gates": [],
        "retry_count": 1u32,
        "timestamp": Utc::now().to_rfc3339(),
        "constraint_tombstone": null
    }))
    .expect("deserialization must succeed");

    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::TopologyProvisioned(event),
    );

    assert_eq!(state.status, "provisioning");
    assert_eq!(
        TaskPhase::try_from(state.phase).unwrap(),
        TaskPhase::Provisioning
    );
    assert_eq!(state.explorers_total, 1);
    assert_eq!(state.autonomic_retries, 1);
}

#[test]
fn apply_multiplication_condition_failed_sets_phase() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::MultiplicationConditionFailed(MultiplicationConditionFailedEvent {
            task_id: tid.clone(),
            failure: MultiplicationConditionFailure::InsufficientCompetence {
                actual: 0.3,
                required: 0.7,
            },
            retry_count: 0,
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.status, "validating");
    assert_eq!(
        TaskPhase::try_from(state.phase).unwrap(),
        TaskPhase::MultiplicationCheck
    );
}

#[test]
fn apply_selection_resolved_sets_merging() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::SelectionResolved(SelectionResolvedEvent {
            task_id: tid.clone(),
            valid_proposals: vec![],
            pruned_proposals: vec![],
            merge_strategy: MergeStrategy::ScoreOrdered,
            timestamp: Utc::now(),
            merge_elapsed_secs: None,
            n_input_proposals: 0,
            n_failed_proposals: 0,
            merge_selection_mode: None,
        }),
    );
    assert_eq!(state.status, "merging");
    assert_eq!(
        TaskPhase::try_from(state.phase).unwrap(),
        TaskPhase::Merging
    );
}

#[test]
fn apply_task_complexity_assessed_sets_assessing() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::TaskComplexityAssessed(TaskComplexityAssessedEvent {
            task_id: tid.clone(),
            tcc_structural: 0.5,
            tcc_empirical: None,
            tcc_effective: 0.5,
            n_eff_pool: None,
            task_quadrant: TaskQuadrant::Precision,
            probe_skipped: true,
            probe_skip_reason: ProbeSkipReason::None,
            heavy_fraction: 0.0,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: 0,
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.status, "assessing");
    assert_eq!(
        TaskPhase::try_from(state.phase).unwrap(),
        TaskPhase::ComplexityAssessed
    );
}

#[test]
fn apply_unhandled_event_leaves_state_unchanged() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone(), TenantId::default_tenant());
    let before_status = state.status.clone();
    let before_phase = state.phase;
    SessionJournal::<InMemoryStateBackend>::apply_event(
        &mut state,
        H2AIEvent::LeaderElected(LeaderElectedEvent {
            task_id: tid.clone(),
            term: 1,
            leader_explorer_id: ExplorerId::new(),
            q_confidence: 0.8,
            credibility_score: 1.0,
            rotation_reason: None,
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.status, before_status);
    assert_eq!(state.phase, before_phase);
}

// ── replay with InMemoryStateBackend (mocked NATS) ────────────────────────────

#[tokio::test]
async fn replay_reconstructs_resolved_task_state() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tid = task_id();

    use h2ai_state::backend::EventPublisher;
    backend
        .publish_event(
            &tid,
            &H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
                task_id: tid.clone(),
                system_context: "ctx".into(),
                pareto_weights: pareto_weights(),
                timestamp: Utc::now(),
            }),
        )
        .await
        .unwrap();
    backend
        .publish_event(
            &tid,
            &H2AIEvent::MergeResolved(MergeResolvedEvent {
                task_id: tid.clone(),
                resolved_output: "final".into(),
                j_eff: None,
                oracle_gate_passed: None,
                timestamp: Utc::now(),
                zone3_hints: None,
                contradiction_analysis: None,
            }),
        )
        .await
        .unwrap();

    let journal = SessionJournal::new(backend);
    let state = journal
        .replay(&tid)
        .await
        .expect("replay must succeed")
        .expect("must have state");

    assert_eq!(state.status, "resolved");
    assert_eq!(
        TaskPhase::try_from(state.phase).unwrap(),
        TaskPhase::Resolved
    );
}

#[tokio::test]
async fn replay_returns_none_when_no_events() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tid = task_id();
    let journal = SessionJournal::new(backend);
    let result = journal.replay(&tid).await.expect("replay must not fail");
    assert!(result.is_none(), "no events → None");
}

#[tokio::test]
async fn replay_stops_at_terminal_task_failed_event() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tid = task_id();

    use h2ai_state::backend::EventPublisher;
    backend
        .publish_event(
            &tid,
            &H2AIEvent::TaskFailed(TaskFailedEvent {
                task_id: tid.clone(),
                primary_cause: TerminalCause::Unknown,
                contributing_causes: vec![],
                top_violated_constraints: vec![],
                last_selection_valid_count: None,
                pruned_events: vec![],
                topologies_tried: vec![TopologyKind::Ensemble],
                tau_values_tried: vec![],
                multiplication_condition_failure: None,
                timestamp: Utc::now(),
            }),
        )
        .await
        .unwrap();
    // This event is after the terminal — must NOT be applied.
    backend
        .publish_event(
            &tid,
            &H2AIEvent::MergeResolved(MergeResolvedEvent {
                task_id: tid.clone(),
                resolved_output: "should not appear".into(),
                j_eff: None,
                oracle_gate_passed: None,
                timestamp: Utc::now(),
                zone3_hints: None,
                contradiction_analysis: None,
            }),
        )
        .await
        .unwrap();

    let journal = SessionJournal::new(backend);
    let state = journal.replay(&tid).await.unwrap().unwrap();
    assert_eq!(
        state.status, "failed",
        "must stop at TaskFailed, not apply MergeResolved"
    );
}

#[tokio::test]
async fn snapshot_restores_state_then_replays_tail() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tid = task_id();

    use h2ai_state::backend::EventPublisher;
    // Seq 1: bootstrap event
    let seq = backend
        .publish_event_seq(
            &tid,
            &H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
                task_id: tid.clone(),
                system_context: "ctx".into(),
                pareto_weights: pareto_weights(),
                timestamp: Utc::now(),
            }),
        )
        .await
        .unwrap();

    // Store a snapshot after seq 1.
    let state_snap = TaskState::new(tid.clone(), TenantId::default_tenant());
    let snap = TaskSnapshot {
        task_id: tid.clone(),
        last_sequence: seq,
        task_state_json: serde_json::to_string(&state_snap).unwrap(),
        taken_at: Utc::now(),
    };
    use h2ai_state::backend::SnapshotStore as SS;
    backend.put_snapshot(&snap).await.unwrap();

    // Seq 2: resolved event after snapshot.
    backend
        .publish_event(
            &tid,
            &H2AIEvent::MergeResolved(MergeResolvedEvent {
                task_id: tid.clone(),
                resolved_output: "final".into(),
                j_eff: None,
                oracle_gate_passed: None,
                timestamp: Utc::now(),
                zone3_hints: None,
                contradiction_analysis: None,
            }),
        )
        .await
        .unwrap();

    let journal = SessionJournal::new(backend);
    let recovered = journal.replay(&tid).await.unwrap().unwrap();
    assert_eq!(recovered.status, "resolved");
}

#[tokio::test]
async fn note_event_triggers_snapshot_at_interval() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tid = task_id();
    let journal = SessionJournal::new(backend.clone()).with_snapshot_interval(2);
    let state = TaskState::new(tid.clone(), TenantId::default_tenant());

    journal.note_event(&tid, 1, &state);
    journal.note_event(&tid, 2, &state); // interval hit → spawns snapshot write

    // Poll until the spawned snapshot task has written to the backend
    use h2ai_state::backend::SnapshotStore as SS;
    let stored = loop {
        let s = backend.get_snapshot(&tid).await.unwrap();
        if s.is_some() {
            break s;
        }
        tokio::task::yield_now().await;
    };
    assert_eq!(stored.unwrap().last_sequence, 2);
}

// ── error path coverage ───────────────────────────────────────────────────────

#[tokio::test]
async fn replay_snapshot_error_falls_back_to_full_replay_and_returns_none() {
    let backend = Arc::new(SnapshotErrBackend);
    let journal = SessionJournal::new(backend);
    let tid = task_id();
    // No events in stream, snapshot errors → fallback → no events → None
    let result = journal
        .replay(&tid)
        .await
        .expect("must not propagate snapshot error");
    assert!(result.is_none());
}

#[tokio::test]
async fn replay_stream_error_propagates_as_transport_error() {
    let backend = Arc::new(StreamErrBackend);
    let journal = SessionJournal::new(backend);
    let tid = task_id();
    let err = journal
        .replay(&tid)
        .await
        .expect_err("stream error must propagate");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Transport") || msg.contains("stream"),
        "got: {msg}"
    );
}

#[tokio::test]
async fn replay_bad_snapshot_json_falls_back_and_returns_none() {
    let tid = task_id();
    let backend = Arc::new(BadSnapshotBackend {
        task_id: tid.clone(),
    });
    let journal = SessionJournal::new(backend);
    // Bad JSON snapshot → deserialization fails → full replay from seq 0 → no events → None
    let result = journal
        .replay(&tid)
        .await
        .expect("must not error on bad snapshot JSON");
    assert!(result.is_none());
}

// ── snapshot-only recovery (no new events after snapshot) ────────────────────

#[tokio::test]
async fn replay_snapshot_only_with_no_tail_events_returns_state() {
    let backend = Arc::new(InMemoryStateBackend::new());
    let tid = task_id();

    use h2ai_state::backend::EventPublisher;
    // Publish seq 1.
    let seq = backend
        .publish_event_seq(
            &tid,
            &H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
                task_id: tid.clone(),
                system_context: "ctx".into(),
                pareto_weights: pareto_weights(),
                timestamp: Utc::now(),
            }),
        )
        .await
        .unwrap();

    // Snapshot that covers seq 1 — no events after it.
    let mut snap_state = TaskState::new(tid.clone(), TenantId::default_tenant());
    snap_state.status = "pending".into();
    snap_state.phase = TaskPhase::Bootstrap as u8;
    let snap = TaskSnapshot {
        task_id: tid.clone(),
        last_sequence: seq,
        task_state_json: serde_json::to_string(&snap_state).unwrap(),
        taken_at: Utc::now(),
    };
    use h2ai_state::backend::SnapshotStore as SS;
    backend.put_snapshot(&snap).await.unwrap();

    let journal = SessionJournal::new(backend);
    // from_seq = seq → no events after snapshot → events_seen = 0 but from_seq != 0 → returns Some
    let recovered = journal.replay(&tid).await.unwrap();
    assert!(
        recovered.is_some(),
        "snapshot alone (from_seq != 0) must return Some"
    );
    assert_eq!(recovered.unwrap().status, "pending");
}

// ── note_event early returns with live backend ────────────────────────────────

#[test]
fn note_event_nats_some_snapshot_interval_zero_is_noop() {
    // Covers line 63: `if self.snapshot_interval == 0 { return; }` when nats IS Some.
    // new(backend) sets nats=Some, then .with_snapshot_interval(0) sets interval=0.
    // note_event must pass the first guard (nats is Some) then hit the interval==0 return.
    let backend = Arc::new(InMemoryStateBackend::new());
    let journal = SessionJournal::new(backend).with_snapshot_interval(0);
    let tid = task_id();
    let state = TaskState::new(tid.clone(), TenantId::default_tenant());
    // Must not panic; the snapshot_interval==0 guard fires before any tokio::spawn.
    journal.note_event(&tid, 1, &state);
    journal.note_event(&tid, 2, &state);
}

#[tokio::test]
async fn note_event_snapshot_write_failure_does_not_panic() {
    // Covers line 79: `tracing::warn!` inside the spawned task when put_snapshot errors.
    // SnapshotErrBackend.put_snapshot always returns Err → the warn! on line 79 fires.
    let backend = Arc::new(SnapshotErrBackend);
    // interval=1 so every event triggers a snapshot write attempt.
    let journal = SessionJournal::new(backend).with_snapshot_interval(1);
    let tid = task_id();
    let state = TaskState::new(tid.clone(), TenantId::default_tenant());
    // Trigger the snapshot write (spawned background task).
    journal.note_event(&tid, 1, &state);
    // Yield to let the spawned task run and hit the warn! path.
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;
    // No assertion needed — the goal is that neither the spawn nor the warn! panics.
}

// ── replay stream item error path (lines 164-165) ────────────────────────────

/// Backend that returns a stream which yields a single `Err` item.
/// This exercises the `Ok(Some(Err(e)))` arm in the replay loop (lines 164-165).
struct StreamItemErrBackend;

#[async_trait]
impl SnapshotStore for StreamItemErrBackend {
    async fn put_snapshot(&self, _: &TaskSnapshot) -> Result<(), NatsError> {
        Ok(())
    }
    async fn get_snapshot(&self, _: &TaskId) -> Result<Option<TaskSnapshot>, NatsError> {
        Ok(None)
    }
}

#[async_trait]
impl TailEvents for StreamItemErrBackend {
    async fn tail_task_events_boxed(
        &self,
        _: &TaskId,
        _: u64,
    ) -> Result<BoxStream<'static, Result<(u64, H2AIEvent), NatsError>>, NatsError> {
        // Return a stream that yields one Err item (not an Err from the outer call).
        let item: Result<(u64, H2AIEvent), NatsError> =
            Err(NatsError::StreamError("item-level stream error".into()));
        Ok(Box::pin(futures::stream::once(async move { item })))
    }
}

#[tokio::test]
async fn replay_stream_item_error_propagates_as_transport_error() {
    // Covers lines 164-165: `Ok(Some(Err(e))) => return Err(...)`.
    // The stream successfully starts but yields a single Err item.
    let backend = Arc::new(StreamItemErrBackend);
    let journal = SessionJournal::new(backend);
    let tid = task_id();
    let err = journal
        .replay(&tid)
        .await
        .expect_err("stream item error must propagate as Err");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Transport") || msg.contains("stream"),
        "error must be a Transport variant, got: {msg}"
    );
}

// ── live NATS integration tests (skipped when NATS unavailable) ───────────────

#[tokio::test]
async fn live_nats_replay_reconstructs_resolved_task_state() {
    use h2ai_state::nats::NatsClient;

    let nats_url = h2ai_config::H2AIConfig::default().nats_url;
    let nats = Arc::new(match NatsClient::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return;
        }
    });
    nats.ensure_infrastructure()
        .await
        .expect("infra setup failed");

    let tid = task_id();

    // Use publish_event_seq (awaits JetStream ack) so messages are persisted before replay
    nats.publish_event_seq(
        &tid,
        &H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
            task_id: tid.clone(),
            system_context: "ctx".into(),
            pareto_weights: pareto_weights(),
            timestamp: Utc::now(),
        }),
    )
    .await
    .unwrap();

    nats.publish_event_seq(
        &tid,
        &H2AIEvent::MergeResolved(MergeResolvedEvent {
            task_id: tid.clone(),
            resolved_output: "final".into(),
            j_eff: None,
            oracle_gate_passed: None,
            timestamp: Utc::now(),
            zone3_hints: None,
            contradiction_analysis: None,
        }),
    )
    .await
    .unwrap();

    let journal = SessionJournal::new(nats);
    let state = journal
        .replay(&tid)
        .await
        .expect("replay failed")
        .expect("should have events");

    assert_eq!(state.status, "resolved");
    assert_eq!(
        TaskPhase::try_from(state.phase).unwrap(),
        TaskPhase::Resolved
    );
}
