use crate::checkpoint::TaskCheckpoint;
use crate::identity::TaskId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCheckpointEntry {
    pub task_id: TaskId,
    pub seq: u32,
    pub base_seq: u32,
    pub kind: CheckpointKind,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum CheckpointKind {
    Base(Box<TaskCheckpoint>),
    Delta(Vec<json_patch::PatchOperation>),
}
