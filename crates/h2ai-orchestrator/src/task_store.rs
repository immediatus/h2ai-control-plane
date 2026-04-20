use dashmap::DashMap;
use h2ai_types::identity::TaskId;
use std::sync::Arc;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPhase {
    Bootstrap = 1,
    Provisioning = 2,
    MultiplicationCheck = 3,
    ParallelGeneration = 4,
    AuditorGate = 5,
    Merging = 6,
    Resolved = 7,
    Failed = 8,
}

impl TaskPhase {
    pub fn status_str(&self) -> &'static str {
        match self {
            Self::Bootstrap => "pending",
            Self::Provisioning => "provisioning",
            Self::MultiplicationCheck => "provisioning",
            Self::ParallelGeneration => "generating",
            Self::AuditorGate => "auditing",
            Self::Merging => "merging",
            Self::Resolved => "resolved",
            Self::Failed => "failed",
        }
    }

    pub fn name_str(&self) -> &'static str {
        match self {
            Self::Bootstrap => "Bootstrap",
            Self::Provisioning => "TopologyProvisioning",
            Self::MultiplicationCheck => "MultiplicationCheck",
            Self::ParallelGeneration => "ParallelGeneration",
            Self::AuditorGate => "AuditorGate",
            Self::Merging => "Merging",
            Self::Resolved => "Resolved",
            Self::Failed => "Failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskState {
    pub task_id: TaskId,
    pub status: String,
    pub phase: u8,
    pub phase_name: String,
    pub explorers_completed: u32,
    pub explorers_total: u32,
    pub proposals_valid: u32,
    pub proposals_pruned: u32,
    pub autonomic_retries: u32,
}

impl TaskState {
    pub fn new(task_id: TaskId) -> Self {
        Self {
            task_id,
            status: "pending".into(),
            phase: TaskPhase::Bootstrap as u8,
            phase_name: TaskPhase::Bootstrap.name_str().into(),
            explorers_completed: 0,
            explorers_total: 0,
            proposals_valid: 0,
            proposals_pruned: 0,
            autonomic_retries: 0,
        }
    }
}

#[derive(Clone, Default)]
pub struct TaskStore(Arc<DashMap<String, TaskState>>);

impl TaskStore {
    pub fn new() -> Self {
        Self(Arc::new(DashMap::new()))
    }

    pub fn insert(&self, id: TaskId, state: TaskState) {
        self.0.insert(id.to_string(), state);
    }

    pub fn get(&self, id: &TaskId) -> Option<TaskState> {
        self.0.get(&id.to_string()).map(|r| r.value().clone())
    }

    pub fn set_phase(&self, id: &TaskId, phase: TaskPhase, explorer_count: u32, retries: u32) {
        if let Some(mut entry) = self.0.get_mut(&id.to_string()) {
            entry.phase = phase as u8;
            entry.phase_name = phase.name_str().into();
            entry.status = phase.status_str().into();
            entry.explorers_total = explorer_count;
            entry.autonomic_retries = retries;
        }
    }

    pub fn increment_completed(&self, id: &TaskId) {
        if let Some(mut entry) = self.0.get_mut(&id.to_string()) {
            entry.explorers_completed += 1;
        }
    }

    pub fn record_validation(&self, id: &TaskId, valid: bool) {
        if let Some(mut entry) = self.0.get_mut(&id.to_string()) {
            if valid {
                entry.proposals_valid += 1;
            } else {
                entry.proposals_pruned += 1;
            }
        }
    }

    pub fn mark_resolved(&self, id: &TaskId) {
        if let Some(mut entry) = self.0.get_mut(&id.to_string()) {
            entry.status = "resolved".into();
            entry.phase = TaskPhase::Resolved as u8;
            entry.phase_name = TaskPhase::Resolved.name_str().into();
        }
    }

    pub fn mark_failed(&self, id: &TaskId) {
        if let Some(mut entry) = self.0.get_mut(&id.to_string()) {
            entry.status = "failed".into();
            entry.phase = TaskPhase::Failed as u8;
            entry.phase_name = TaskPhase::Failed.name_str().into();
        }
    }
}
