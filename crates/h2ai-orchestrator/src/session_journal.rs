use crate::error::OrchestratorError;
use crate::task_store::{TaskPhase, TaskState};
use dashmap::DashMap;
use futures::StreamExt;
use h2ai_state::nats::NatsClient;
use h2ai_types::events::{H2AIEvent, TaskSnapshot};
use h2ai_types::identity::TaskId;
use std::sync::Arc;
use std::time::Duration;

struct EventCounter {
    interval: usize,
    count: usize,
}

impl EventCounter {
    fn new(interval: usize) -> Self {
        Self { interval, count: 0 }
    }

    /// Increments the counter and returns `true` when a snapshot should be taken.
    fn tick(&mut self) -> bool {
        self.count += 1;
        self.interval > 0 && self.count % self.interval == 0
    }
}

#[cfg(test)]
mod counter_tests {
    use super::EventCounter;

    #[test]
    fn zero_interval_never_triggers() {
        let mut c = EventCounter::new(0);
        for _ in 0..100 {
            assert!(!c.tick());
        }
    }

    #[test]
    fn triggers_at_every_multiple_of_interval() {
        let mut c = EventCounter::new(5);
        for i in 1usize..=20 {
            let triggered = c.tick();
            assert_eq!(triggered, i % 5 == 0, "at event {i}");
        }
    }

    #[test]
    fn interval_of_one_triggers_every_event() {
        let mut c = EventCounter::new(1);
        for _ in 0..10 {
            assert!(c.tick());
        }
    }
}

pub struct SessionJournal {
    nats: Arc<NatsClient>,
    snapshot_interval: usize,
    counters: Arc<DashMap<TaskId, EventCounter>>,
}

impl SessionJournal {
    pub fn new(nats: Arc<NatsClient>) -> Self {
        Self {
            nats,
            snapshot_interval: 0,
            counters: Arc::new(DashMap::new()),
        }
    }

    /// Enable periodic snapshotting: write a snapshot every `interval` events per task.
    /// `interval = 0` disables snapshotting (default).
    pub fn with_snapshot_interval(mut self, interval: usize) -> Self {
        self.snapshot_interval = interval;
        self
    }

    /// Record that an event at `seq` was published for `task_id`.
    /// When the per-task event count hits the snapshot interval, fires a fire-and-forget
    /// background task to write the current state to NATS KV.
    pub fn note_event(&self, task_id: &TaskId, seq: u64, state: &TaskState) {
        if self.snapshot_interval == 0 {
            return;
        }
        let mut entry = self
            .counters
            .entry(task_id.clone())
            .or_insert_with(|| EventCounter::new(self.snapshot_interval));
        if entry.tick() {
            let snapshot = TaskSnapshot {
                task_id: task_id.clone(),
                last_sequence: seq,
                task_state_json: serde_json::to_string(state).unwrap_or_default(),
                taken_at: chrono::Utc::now(),
            };
            let nats = self.nats.clone();
            tokio::spawn(async move {
                if let Err(e) = nats.put_snapshot(&snapshot).await {
                    tracing::warn!(
                        target: "h2ai.journal",
                        task_id = %snapshot.task_id,
                        error = %e,
                        "snapshot write failed"
                    );
                }
            });
        }
    }

    /// Replay all stored H2AIEvents for `task_id` from JetStream, reconstructing `TaskState`.
    /// If a snapshot exists, restores state from it and replays only events after the snapshot
    /// sequence number. Falls back to full replay (from sequence 0) when no snapshot is found.
    /// Stops on the first terminal event (MergeResolved / TaskFailed) or after 200 ms of inactivity.
    pub async fn replay(&self, task_id: &TaskId) -> Result<Option<TaskState>, OrchestratorError> {
        // Try to load the latest snapshot to short-circuit full history replay.
        let (mut state, from_seq) = match self.nats.get_snapshot(task_id).await {
            Ok(Some(snapshot)) => {
                match serde_json::from_str::<TaskState>(&snapshot.task_state_json) {
                    Ok(restored) => (restored, snapshot.last_sequence),
                    Err(e) => {
                        tracing::warn!(
                            target: "h2ai.journal",
                            task_id = %task_id,
                            error = %e,
                            "snapshot deserialization failed, falling back to full replay"
                        );
                        (TaskState::new(task_id.clone()), 0)
                    }
                }
            }
            Ok(None) => (TaskState::new(task_id.clone()), 0),
            Err(e) => {
                tracing::warn!(
                    target: "h2ai.journal",
                    task_id = %task_id,
                    error = %e,
                    "snapshot load failed, falling back to full replay"
                );
                (TaskState::new(task_id.clone()), 0)
            }
        };

        let stream = self
            .nats
            .tail_task_events(task_id, from_seq)
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        futures::pin_mut!(stream);
        let mut events_seen: u32 = 0;

        loop {
            match tokio::time::timeout(Duration::from_millis(200), stream.next()).await {
                Ok(Some(Ok((_, event)))) => {
                    events_seen += 1;
                    let terminal = matches!(
                        event,
                        H2AIEvent::MergeResolved(_) | H2AIEvent::TaskFailed(_)
                    );
                    Self::apply_event(&mut state, event);
                    if terminal {
                        break;
                    }
                }
                Ok(Some(Err(e))) => {
                    return Err(OrchestratorError::Transport(e.to_string()));
                }
                // Stream exhausted or 200 ms passed with no new events → caught up.
                Ok(None) | Err(_) => break,
            }
        }

        // A snapshot alone (no new events) is a valid recovered state.
        if events_seen == 0 && from_seq == 0 {
            Ok(None)
        } else {
            Ok(Some(state))
        }
    }

    /// Apply a single `H2AIEvent` to `state` in place. Pure function — no I/O.
    /// Exposed as `pub` so unit tests can exercise event mapping without a NATS connection.
    pub fn apply_event(state: &mut TaskState, event: H2AIEvent) {
        match event {
            H2AIEvent::TaskBootstrapped(_) => {
                state.status = TaskPhase::Bootstrap.status_str().into();
                state.phase = TaskPhase::Bootstrap as u8;
                state.phase_name = TaskPhase::Bootstrap.name_str().into();
            }
            H2AIEvent::TopologyProvisioned(e) => {
                state.status = TaskPhase::Provisioning.status_str().into();
                state.phase = TaskPhase::Provisioning as u8;
                state.phase_name = TaskPhase::Provisioning.name_str().into();
                state.explorers_total = e.explorer_configs.len() as u32;
                state.autonomic_retries = e.retry_count;
            }
            H2AIEvent::MultiplicationConditionFailed(_) => {
                state.status = TaskPhase::MultiplicationCheck.status_str().into();
                state.phase = TaskPhase::MultiplicationCheck as u8;
                state.phase_name = TaskPhase::MultiplicationCheck.name_str().into();
            }
            H2AIEvent::Proposal(_) | H2AIEvent::ProposalFailed(_) => {
                state.explorers_completed += 1;
                state.status = TaskPhase::ParallelGeneration.status_str().into();
                state.phase = TaskPhase::ParallelGeneration as u8;
                state.phase_name = TaskPhase::ParallelGeneration.name_str().into();
            }
            H2AIEvent::VerificationScored(e) => {
                if e.passed {
                    state.proposals_valid += 1;
                } else {
                    state.proposals_pruned += 1;
                }
                state.status = TaskPhase::AuditorGate.status_str().into();
                state.phase = TaskPhase::AuditorGate as u8;
                state.phase_name = TaskPhase::AuditorGate.name_str().into();
            }
            H2AIEvent::BranchPruned(_) => {
                state.proposals_pruned += 1;
            }
            H2AIEvent::ZeroSurvival(_) => {
                state.autonomic_retries += 1;
            }
            H2AIEvent::SemilatticeCompiled(_) => {
                state.status = TaskPhase::Merging.status_str().into();
                state.phase = TaskPhase::Merging as u8;
                state.phase_name = TaskPhase::Merging.name_str().into();
            }
            H2AIEvent::MergeResolved(_) => {
                state.status = TaskPhase::Resolved.status_str().into();
                state.phase = TaskPhase::Resolved as u8;
                state.phase_name = TaskPhase::Resolved.name_str().into();
            }
            H2AIEvent::TaskFailed(e) => {
                state.status = TaskPhase::Failed.status_str().into();
                state.phase = TaskPhase::Failed as u8;
                state.phase_name = TaskPhase::Failed.name_str().into();
                state.autonomic_retries = e.topologies_tried.len() as u32;
            }
            _ => {}
        }
    }
}
