use crate::identity::{AgentId, TaskId};
use crate::physics::TauValue;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentTool {
    Shell,
    WebSearch,
    CodeExecution,
    FileSystem,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentDescriptor {
    pub model: String,
    pub tools: Vec<AgentTool>,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", content = "message")]
pub enum AgentState {
    Idle,
    Executing,
    AwaitingApproval,
    Failed(String),
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskPayload {
    pub task_id: TaskId,
    pub agent: AgentDescriptor,
    pub instructions: String,
    pub context: String,
    pub tau: TauValue,
    pub max_tokens: u64,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub agent_id: AgentId,
    pub output: String,
    pub token_cost: u64,
    pub error: Option<String>,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum AgentTelemetryEvent {
    LlmPromptSent {
        task_id: TaskId,
        agent_id: AgentId,
        prompt: String,
        timestamp: DateTime<Utc>,
    },
    LlmResponseReceived {
        task_id: TaskId,
        agent_id: AgentId,
        response: String,
        token_cost: u64,
        timestamp: DateTime<Utc>,
    },
    ShellCommandExecuted {
        task_id: TaskId,
        agent_id: AgentId,
        command: String,
        stdout: String,
        stderr: String,
        exit_code: i32,
        timestamp: DateTime<Utc>,
    },
    SystemError {
        task_id: TaskId,
        agent_id: AgentId,
        error: String,
        timestamp: DateTime<Utc>,
    },
}
