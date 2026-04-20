use crate::error::OrchestratorError;
use crate::task_store::{TaskPhase, TaskState};
use futures::StreamExt;
use h2ai_state::nats::NatsClient;
use h2ai_types::events::H2AIEvent;
use h2ai_types::identity::TaskId;
use std::sync::Arc;
use std::time::Duration;

pub struct SessionJournal {
    nats: Arc<NatsClient>,
}

impl SessionJournal {
    pub fn new(nats: Arc<NatsClient>) -> Self {
        Self { nats }
    }

    /// Replay all stored H2AIEvents for `task_id` from JetStream offset 0 and
    /// reconstruct the current `TaskState`. Stops on the first terminal event
    /// (MergeResolved / TaskFailed) or after 200 ms of inactivity (in-flight tasks).
    pub async fn replay(&self, task_id: &TaskId) -> Result<Option<TaskState>, OrchestratorError> {
        let stream = self
            .nats
            .tail_task_events(task_id, 0)
            .await
            .map_err(|e| OrchestratorError::Transport(e.to_string()))?;

        futures::pin_mut!(stream);
        let mut state = TaskState::new(task_id.clone());
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

        if events_seen == 0 {
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
