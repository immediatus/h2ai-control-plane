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

/// Ordered Low < Mid < High. Agents declare their tier; tasks declare a maximum.
#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum CostTier {
    Low,
    #[default]
    Mid,
    High,
}

/// Scheduling requirements a task passes to AgentProvider::select_agent.
#[typeshare]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
    pub max_cost_tier: CostTier,
    pub required_tools: Vec<AgentTool>,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentDescriptor {
    pub model: String,
    pub tools: Vec<AgentTool>,
    #[serde(default)]
    pub cost_tier: CostTier,
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
    /// Identity the edge agent must use when publishing telemetry and results.
    pub agent_id: AgentId,
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
pub struct AgentHeartbeat {
    pub agent_id: AgentId,
    pub descriptor: AgentDescriptor,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub active_tasks: u32,
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
