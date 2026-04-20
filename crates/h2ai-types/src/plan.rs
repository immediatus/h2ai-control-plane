use crate::config::AgentRole;
use crate::identity::{SubtaskId, TaskId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    pub id: SubtaskId,
    /// The specific instruction for this subtask's explorers.
    pub description: String,
    /// Subtask IDs that must complete before this one can start.
    pub depends_on: Vec<SubtaskId>,
    /// Optional role override for this subtask's explorers.
    pub role_hint: Option<AgentRole>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "data", rename_all = "snake_case")]
pub enum PlanStatus {
    Draft,
    PendingReview,
    Approved,
    Rejected { reason: String },
    Executing { completed: usize, total: usize },
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskPlan {
    pub plan_id: TaskId,
    pub parent_task_id: TaskId,
    pub subtasks: Vec<Subtask>,
    pub status: PlanStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskResult {
    pub subtask_id: SubtaskId,
    pub output: String,
    pub token_cost: u64,
    pub timestamp: DateTime<Utc>,
}
