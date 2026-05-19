use crate::config::TopologyKind;
use crate::identity::{TaskId, TenantId};
use crate::sizing::TaskQuadrant;
use serde::{Deserialize, Serialize};

/// Phase reached by the reasoning checkpoint progressive write sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningCheckpointPhase {
    /// Checkpoint written at task start — manifest and tags captured.
    Created,
    /// Thinking loop completed — shared_understanding, tensions, archetype_selection written.
    ThinkingDone,
    /// Adapter wave `k` (0-based) completed — wave outputs appended.
    WaveCompleted(u32),
    /// Merge phase completed — synthesis output available.
    MergeDone,
    /// Task fully resolved — TaskMetaState has been projected and written.
    Resolved,
}

impl ReasoningCheckpointPhase {
    /// Returns `true` when the phase is at least as advanced as `other`.
    pub fn is_at_least(&self, other: &Self) -> bool {
        self.rank() >= other.rank()
    }

    fn rank(&self) -> u32 {
        match self {
            Self::Created => 0,
            Self::ThinkingDone => 1,
            Self::WaveCompleted(k) => 2 + k,
            Self::MergeDone => u32::MAX - 1,
            Self::Resolved => u32::MAX,
        }
    }
}

/// Progressive reasoning checkpoint written by the engine at each phase gate.
///
/// Stored in `H2AI_CHECKPOINT_{tenant_id}` KV bucket (per-tenant). TTL: 7 days.
/// All writes are fire-and-forget — a write failure never blocks the task result.
///
/// Distinct from `TaskCheckpoint` (execution-phase checkpoints storing
/// proposals/auditor-survivors). This checkpoint captures reasoning artifacts
/// used for warm-start recovery and induction cycle input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskReasoningCheckpoint {
    pub task_id: TaskId,
    pub tenant_id: TenantId,
    /// Unix timestamp, seconds since epoch.
    pub created_at: u64,
    /// Unix timestamp, seconds since epoch.
    pub last_updated: u64,
    pub phase: ReasoningCheckpointPhase,

    // Set at task start
    pub constraint_tags: Vec<String>,
    pub domain: Option<String>,
    pub task_quadrant: Option<TaskQuadrant>,
    pub system_context_with_rubric_hash: u64,
    pub constraint_corpus_fingerprint: u64,

    // Populated after thinking loop completes (phase >= ThinkingDone)
    pub shared_understanding: Option<String>,
    pub tensions: Option<Vec<String>>,
    pub archetype_selection: Option<Vec<ArchetypeSelection>>,
    pub thinking_iterations: Option<u32>,

    // Appended after each adapter wave
    pub completed_waves: Vec<CompletedWave>,

    // Populated at resolution
    pub retry_count: u32,
    pub retry_context_that_resolved: Option<String>,
    pub tried_topologies: Vec<TopologyKind>,
    pub tau_values_that_converged: Option<Vec<f64>>,
    /// `HarnessAttribution` serialized as JSON. Stored at resolution so that
    /// `run_from_checkpoint` can hydrate a full `EngineOutput` without re-running
    /// inference, preventing zeroed attribution from corrupting downstream analytics.
    pub resolved_attribution_json: Option<String>,
    /// `EngineOutput::waste_ratio` captured at resolution for the same reason.
    pub resolved_waste_ratio: Option<f64>,

    /// Number of consecutive HITL gates where the timeout fired without a human response.
    /// Drives the adaptive timeout decay formula. Reset to 0 when any signal is received.
    #[serde(default)]
    pub hitl_timeouts_fired: u32,
}

impl TaskReasoningCheckpoint {
    /// Construct the initial checkpoint written at task start.
    pub fn new_created(
        task_id: TaskId,
        tenant_id: TenantId,
        constraint_tags: Vec<String>,
        domain: Option<String>,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            task_id,
            tenant_id,
            created_at: now,
            last_updated: now,
            phase: ReasoningCheckpointPhase::Created,
            constraint_tags,
            domain,
            task_quadrant: None,
            system_context_with_rubric_hash: 0,
            constraint_corpus_fingerprint: 0,
            shared_understanding: None,
            tensions: None,
            archetype_selection: None,
            thinking_iterations: None,
            completed_waves: Vec::new(),
            retry_count: 0,
            retry_context_that_resolved: None,
            tried_topologies: Vec::new(),
            tau_values_that_converged: None,
            resolved_attribution_json: None,
            resolved_waste_ratio: None,
            hitl_timeouts_fired: 0,
        }
    }

    /// Project into an immutable `TaskMetaState` after resolution.
    /// Returns `None` when thinking artifacts are missing (pre-thinking-loop tasks).
    pub fn into_meta_state(self) -> Option<TaskMetaState> {
        let shared_understanding = self.shared_understanding?;
        let tensions = self.tensions.unwrap_or_default();
        let archetype_results = self
            .archetype_selection
            .unwrap_or_default()
            .into_iter()
            .map(|a| ArchetypeResult {
                name: a.name,
                confidence: a.confidence,
            })
            .collect();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Some(TaskMetaState {
            task_id: self.task_id,
            tenant_id: self.tenant_id,
            resolved_at: now,
            constraint_tags: self.constraint_tags,
            domain: self.domain,
            task_quadrant: self.task_quadrant,
            shared_understanding,
            tensions,
            archetype_results,
            thinking_iterations: self.thinking_iterations.unwrap_or(0),
            retry_count: self.retry_count,
            retry_context_that_resolved: self.retry_context_that_resolved,
            tried_topologies: self.tried_topologies,
            tau_values_that_converged: self.tau_values_that_converged,
            system_context_with_rubric_hash: self.system_context_with_rubric_hash,
            constraint_corpus_fingerprint: self.constraint_corpus_fingerprint,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeSelection {
    pub name: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedWave {
    pub wave_index: u32,
    pub adapter_outputs: Vec<AdapterWaveOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterWaveOutput {
    pub adapter_id: String,
    /// xxHash64 of the output text — full text is not re-stored to keep checkpoints small.
    pub output_hash: u64,
    pub survived: bool,
}

/// Immutable projection of `TaskReasoningCheckpoint` written at resolution.
///
/// Wave-level detail is dropped; only reasoning artifacts and retrieval index
/// are kept. Stored in `H2AI_META_{tenant_id}` KV bucket. TTL: 90 days.
/// Read by `InductionScheduler` for distillation cycles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetaState {
    pub task_id: TaskId,
    pub tenant_id: TenantId,
    /// Unix timestamp, seconds since epoch.
    pub resolved_at: u64,

    // Retrieval index
    pub constraint_tags: Vec<String>,
    pub domain: Option<String>,
    pub task_quadrant: Option<TaskQuadrant>,

    // Thinking loop artifacts
    pub shared_understanding: String,
    pub tensions: Vec<String>,
    pub archetype_results: Vec<ArchetypeResult>,
    pub thinking_iterations: u32,

    // Retry artifacts
    pub retry_count: u32,
    pub retry_context_that_resolved: Option<String>,
    pub tried_topologies: Vec<TopologyKind>,
    pub tau_values_that_converged: Option<Vec<f64>>,

    // Rubric fingerprints
    pub system_context_with_rubric_hash: u64,
    pub constraint_corpus_fingerprint: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeResult {
    pub name: String,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::TaskId;

    fn make_checkpoint() -> TaskReasoningCheckpoint {
        TaskReasoningCheckpoint::new_created(
            TaskId::new(),
            TenantId::default_tenant(),
            vec!["api-security".into()],
            Some("security".into()),
        )
    }

    #[test]
    fn new_checkpoint_phase_is_created() {
        let cp = make_checkpoint();
        assert_eq!(cp.phase, ReasoningCheckpointPhase::Created);
    }

    #[test]
    fn phase_ordering_created_lt_thinking_done() {
        assert!(
            !ReasoningCheckpointPhase::Created.is_at_least(&ReasoningCheckpointPhase::ThinkingDone)
        );
        assert!(
            ReasoningCheckpointPhase::ThinkingDone.is_at_least(&ReasoningCheckpointPhase::Created)
        );
    }

    #[test]
    fn phase_ordering_wave0_lt_wave1() {
        let w0 = ReasoningCheckpointPhase::WaveCompleted(0);
        let w1 = ReasoningCheckpointPhase::WaveCompleted(1);
        assert!(w1.is_at_least(&w0));
        assert!(!w0.is_at_least(&w1));
    }

    #[test]
    fn phase_ordering_merge_done_gt_wave5() {
        let w5 = ReasoningCheckpointPhase::WaveCompleted(5);
        assert!(ReasoningCheckpointPhase::MergeDone.is_at_least(&w5));
    }

    #[test]
    fn into_meta_state_returns_none_without_thinking() {
        let cp = make_checkpoint();
        assert!(cp.into_meta_state().is_none());
    }

    #[test]
    fn phase_ordering_resolved_gt_merge_done() {
        assert!(
            ReasoningCheckpointPhase::Resolved.is_at_least(&ReasoningCheckpointPhase::MergeDone)
        );
        assert!(
            !ReasoningCheckpointPhase::MergeDone.is_at_least(&ReasoningCheckpointPhase::Resolved)
        );
    }

    #[test]
    fn into_meta_state_maps_archetype_selection() {
        let mut cp = make_checkpoint();
        cp.shared_understanding = Some("test understanding".into());
        cp.archetype_selection = Some(vec![ArchetypeSelection {
            name: "socratic".into(),
            confidence: 0.9,
        }]);
        cp.phase = ReasoningCheckpointPhase::Resolved;
        let meta = cp.into_meta_state().unwrap();
        assert_eq!(meta.archetype_results.len(), 1);
        assert_eq!(meta.archetype_results[0].name, "socratic");
        assert!((meta.archetype_results[0].confidence - 0.9).abs() < 1e-10);
    }

    #[test]
    fn into_meta_state_task_quadrant_none_when_not_set() {
        let mut cp = make_checkpoint();
        cp.shared_understanding = Some("test".into());
        cp.phase = ReasoningCheckpointPhase::Resolved;
        // task_quadrant was not set — should be None in projection
        let meta = cp.into_meta_state().unwrap();
        assert!(meta.task_quadrant.is_none());
    }

    #[test]
    fn into_meta_state_with_thinking_artifacts() {
        let mut cp = make_checkpoint();
        cp.shared_understanding = Some("The task requires…".into());
        cp.tensions = Some(vec!["security vs usability".into()]);
        cp.phase = ReasoningCheckpointPhase::Resolved;
        let meta = cp.into_meta_state();
        assert!(meta.is_some());
        let m = meta.unwrap();
        assert_eq!(m.tensions.len(), 1);
        assert_eq!(m.constraint_tags, vec!["api-security"]);
    }
}
