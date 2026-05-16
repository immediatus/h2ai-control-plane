use dashmap::DashMap;
use h2ai_types::identity::{TaskId, TenantId};
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
    /// Phase 1.5: task complexity assessed, routing quadrant assigned.
    /// Uses value 9 to avoid reordering existing phase discriminants.
    ComplexityAssessed = 9,
    /// Phase for crash-recovery: task is paused and awaiting human approval before resuming.
    AwaitingApproval = 10,
}

impl TryFrom<u8> for TaskPhase {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            x if x == Self::Bootstrap as u8 => Ok(Self::Bootstrap),
            x if x == Self::Provisioning as u8 => Ok(Self::Provisioning),
            x if x == Self::MultiplicationCheck as u8 => Ok(Self::MultiplicationCheck),
            x if x == Self::ParallelGeneration as u8 => Ok(Self::ParallelGeneration),
            x if x == Self::AuditorGate as u8 => Ok(Self::AuditorGate),
            x if x == Self::Merging as u8 => Ok(Self::Merging),
            x if x == Self::Resolved as u8 => Ok(Self::Resolved),
            x if x == Self::Failed as u8 => Ok(Self::Failed),
            x if x == Self::ComplexityAssessed as u8 => Ok(Self::ComplexityAssessed),
            x if x == Self::AwaitingApproval as u8 => Ok(Self::AwaitingApproval),
            other => Err(other),
        }
    }
}

impl TaskPhase {
    pub fn status_str(&self) -> &'static str {
        match self {
            Self::Bootstrap => "pending",
            Self::ComplexityAssessed => "assessing",
            Self::Provisioning => "provisioning",
            Self::MultiplicationCheck => "validating",
            Self::ParallelGeneration => "generating",
            Self::AuditorGate => "auditing",
            Self::Merging => "merging",
            Self::Resolved => "resolved",
            Self::Failed => "failed",
            Self::AwaitingApproval => "awaiting_approval",
        }
    }

    pub fn name_str(&self) -> &'static str {
        match self {
            Self::Bootstrap => "Bootstrap",
            Self::ComplexityAssessed => "ComplexityAssessment",
            Self::Provisioning => "TopologyProvisioning",
            Self::MultiplicationCheck => "MultiplicationCheck",
            Self::ParallelGeneration => "ParallelGeneration",
            Self::AuditorGate => "AuditorGate",
            Self::Merging => "Merging",
            Self::Resolved => "Resolved",
            Self::Failed => "Failed",
            Self::AwaitingApproval => "AwaitingApproval",
        }
    }

    pub fn try_from_name_str(s: &str) -> Option<Self> {
        match s {
            "Bootstrap" => Some(Self::Bootstrap),
            "TopologyProvisioning" => Some(Self::Provisioning),
            "MultiplicationCheck" => Some(Self::MultiplicationCheck),
            "ParallelGeneration" => Some(Self::ParallelGeneration),
            "AuditorGate" => Some(Self::AuditorGate),
            "Merging" => Some(Self::Merging),
            "Resolved" => Some(Self::Resolved),
            "Failed" => Some(Self::Failed),
            "ComplexityAssessment" => Some(Self::ComplexityAssessed),
            "AwaitingApproval" => Some(Self::AwaitingApproval),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskState {
    pub task_id: TaskId,
    pub tenant_id: TenantId,
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
    pub fn new(task_id: TaskId, tenant_id: TenantId) -> Self {
        Self {
            task_id,
            tenant_id,
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

    /// Return the task state only when it belongs to `tenant_id`.
    ///
    /// Returns `None` when the task doesn't exist or belongs to a different tenant.
    /// Use this on external-facing routes to prevent cross-tenant task enumeration.
    pub fn get_for_tenant(&self, id: &TaskId, tenant_id: &TenantId) -> Option<TaskState> {
        self.0
            .get(&id.to_string())
            .filter(|r| &r.tenant_id == tenant_id)
            .map(|r| r.value().clone())
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

    pub fn set_awaiting_approval(&self, id: &TaskId) {
        if let Some(mut entry) = self.0.get_mut(&id.to_string()) {
            entry.phase = TaskPhase::AwaitingApproval as u8;
            entry.phase_name = TaskPhase::AwaitingApproval.name_str().into();
            entry.status = TaskPhase::AwaitingApproval.status_str().into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_types::identity::{TaskId, TenantId};

    #[test]
    fn get_for_tenant_returns_none_for_wrong_tenant() {
        let store = TaskStore::new();
        let task_id = TaskId::new();
        store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), TenantId::from("acme")),
        );
        assert!(store
            .get_for_tenant(&task_id, &TenantId::from("beta"))
            .is_none());
    }

    #[test]
    fn get_for_tenant_returns_state_for_owner() {
        let store = TaskStore::new();
        let task_id = TaskId::new();
        let tenant = TenantId::from("acme");
        store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), tenant.clone()),
        );
        assert!(store.get_for_tenant(&task_id, &tenant).is_some());
    }

    #[test]
    fn get_without_tenant_still_works_for_backward_compat() {
        let store = TaskStore::new();
        let task_id = TaskId::new();
        store.insert(
            task_id.clone(),
            TaskState::new(task_id.clone(), TenantId::default_tenant()),
        );
        assert!(store.get(&task_id).is_some());
    }
}
