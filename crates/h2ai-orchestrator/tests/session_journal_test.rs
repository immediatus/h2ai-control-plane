use chrono::Utc;
use h2ai_orchestrator::session_journal::SessionJournal;
use h2ai_orchestrator::task_store::{TaskPhase, TaskState};
use h2ai_types::config::{AdapterKind, ParetoWeights, TopologyKind};
use h2ai_types::events::{
    BranchPrunedEvent, H2AIEvent, MergeResolvedEvent, ProposalEvent, TaskBootstrappedEvent,
    TaskFailedEvent, VerificationScoredEvent, ZeroSurvivalEvent,
};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::{RoleErrorCost, TauValue};

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
    let mut state = TaskState::new(tid.clone());
    SessionJournal::apply_event(
        &mut state,
        H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
            task_id: tid.clone(),
            system_context: "ctx".into(),
            pareto_weights: pareto_weights(),
            j_eff: 1.0,
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
    let mut state = TaskState::new(tid.clone());
    assert_eq!(state.explorers_completed, 0);
    SessionJournal::apply_event(
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
    let mut state = TaskState::new(tid.clone());
    assert_eq!(state.proposals_valid, 0);
    SessionJournal::apply_event(
        &mut state,
        H2AIEvent::VerificationScored(VerificationScoredEvent {
            task_id: tid.clone(),
            explorer_id: explorer_id(),
            score: 0.9,
            reason: "good".into(),
            passed: true,
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
    let mut state = TaskState::new(tid.clone());
    assert_eq!(state.proposals_pruned, 0);
    SessionJournal::apply_event(
        &mut state,
        H2AIEvent::VerificationScored(VerificationScoredEvent {
            task_id: tid.clone(),
            explorer_id: explorer_id(),
            score: 0.1,
            reason: "bad".into(),
            passed: false,
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
    let mut state = TaskState::new(tid.clone());
    assert_eq!(state.autonomic_retries, 0);
    SessionJournal::apply_event(
        &mut state,
        H2AIEvent::ZeroSurvival(ZeroSurvivalEvent {
            task_id: tid.clone(),
            retry_count: 1,
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.autonomic_retries, 1);
}

// ── 6: MergeResolved → resolved / Resolved ───────────────────────────────────

#[test]
fn apply_merge_resolved_sets_resolved() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone());
    SessionJournal::apply_event(
        &mut state,
        H2AIEvent::MergeResolved(MergeResolvedEvent {
            task_id: tid.clone(),
            resolved_output: "final answer".into(),
            timestamp: Utc::now(),
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
    let mut state = TaskState::new(tid.clone());
    SessionJournal::apply_event(
        &mut state,
        H2AIEvent::TaskFailed(TaskFailedEvent {
            task_id: tid.clone(),
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
    // topologies_tried has 1 entry → autonomic_retries = 1
    assert_eq!(state.autonomic_retries, 1);
}

#[test]
fn apply_branch_pruned_increments_proposals_pruned() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone());
    SessionJournal::apply_event(
        &mut state,
        H2AIEvent::BranchPruned(BranchPrunedEvent {
            task_id: tid.clone(),
            explorer_id: explorer_id(),
            reason: "constraint violation".into(),
            constraint_error_cost: RoleErrorCost::new(0.8).unwrap(),
            violated_constraints: vec![],
            timestamp: Utc::now(),
        }),
    );
    assert_eq!(state.proposals_pruned, 1);
    assert_eq!(state.proposals_valid, 0);
}

// ── TaskState serde ───────────────────────────────────────────────────────────

#[test]
fn task_state_serde_roundtrip() {
    let tid = task_id();
    let mut state = TaskState::new(tid.clone());
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

// ── integration (ignored — requires live NATS) ────────────────────────────────

#[tokio::test]
#[ignore]
async fn replay_reconstructs_resolved_task_state() {
    use h2ai_orchestrator::session_journal::SessionJournal;
    use h2ai_state::nats::NatsClient;
    use std::sync::Arc;

    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
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

    nats.publish_event(
        &tid,
        &H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
            task_id: tid.clone(),
            system_context: "ctx".into(),
            pareto_weights: pareto_weights(),
            j_eff: 1.0,
            timestamp: Utc::now(),
        }),
    )
    .await
    .unwrap();

    nats.publish_event(
        &tid,
        &H2AIEvent::MergeResolved(MergeResolvedEvent {
            task_id: tid.clone(),
            resolved_output: "final".into(),
            timestamp: Utc::now(),
        }),
    )
    .await
    .unwrap();

    // Small delay to let JetStream persist messages.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

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

#[tokio::test]
#[ignore]
async fn snapshot_written_and_recovered_via_replay() {
    use h2ai_orchestrator::session_journal::SessionJournal;
    use h2ai_state::nats::NatsClient;
    use std::sync::Arc;

    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
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

    // Publish events and capture the last sequence.
    let seq = nats
        .publish_event_seq(
            &tid,
            &H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
                task_id: tid.clone(),
                system_context: "ctx".into(),
                pareto_weights: pareto_weights(),
                j_eff: 1.0,
                timestamp: Utc::now(),
            }),
        )
        .await
        .expect("publish_event_seq");

    // Manually write a snapshot as if note_event had fired.
    let state_before = TaskState::new(tid.clone());
    let snap = h2ai_types::events::TaskSnapshot {
        task_id: tid.clone(),
        last_sequence: seq,
        task_state_json: serde_json::to_string(&state_before).unwrap(),
        taken_at: Utc::now(),
    };
    nats.put_snapshot(&snap).await.expect("put_snapshot");

    // Now publish a MergeResolved event AFTER the snapshot.
    nats.publish_event(
        &tid,
        &H2AIEvent::MergeResolved(MergeResolvedEvent {
            task_id: tid.clone(),
            resolved_output: "final".into(),
            timestamp: Utc::now(),
        }),
    )
    .await
    .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let journal = SessionJournal::new(nats.clone()).with_snapshot_interval(50);
    let recovered = journal
        .replay(&tid)
        .await
        .expect("replay")
        .expect("state present");

    // The journal should have loaded the snapshot and then replayed only MergeResolved.
    assert_eq!(recovered.status, "resolved");

    // Verify get_snapshot round-trips correctly.
    let loaded = nats.get_snapshot(&tid).await.expect("get_snapshot");
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap().last_sequence, seq);
}
