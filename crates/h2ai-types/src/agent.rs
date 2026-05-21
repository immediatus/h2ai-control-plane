use crate::identity::{AgentId, TaskId};
use crate::sizing::TauValue;
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

/// Scheduling requirements a task passes to `AgentProvider::select_agent`.
#[typeshare]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
    pub max_cost_tier: CostTier,
    pub required_tools: Vec<AgentTool>,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentDescriptor {
    pub model: String,
    pub tools: Vec<AgentTool>,
    #[serde(default)]
    pub cost_tier: CostTier,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "message")]
pub enum AgentState {
    Idle,
    Executing,
    AwaitingApproval,
    Failed(String),
}

/// System context carried in a NATS task message — either inlined (small payloads) or
/// referenced by SHA-256 hex hash in a content-addressed object store (large payloads).
///
/// Inline is used when `len(context_bytes) ≤ payload_offload_threshold_bytes`.
/// Ref is used when the context exceeds the threshold, preventing NATS size-limit failures.
#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum ContextPayload {
    /// Full context string inline.
    Inline(String),
    /// Content-addressed reference: SHA-256 hash (hex) + original byte length.
    Ref { hash: String, byte_len: usize },
}

impl ContextPayload {
    #[must_use]
    pub const fn as_inline(&self) -> Option<&str> {
        match self {
            Self::Inline(s) => Some(s.as_str()),
            Self::Ref { .. } => None,
        }
    }
}

#[typeshare]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WaveMode {
    #[default]
    Normal,
    Hardened,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskPayload {
    pub task_id: TaskId,
    /// Identity the edge agent must use when publishing telemetry and results.
    pub agent_id: AgentId,
    pub agent: AgentDescriptor,
    pub instructions: String,
    pub context: ContextPayload,
    pub tau: TauValue,
    pub max_tokens: u64,
    pub wave_mode: WaveMode,
}

/// Single tool invocation record, carried in `TaskResult` for the NATS audit trail.
#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub tool: AgentTool,
    /// Serialized JSON string of the input passed to the tool executor.
    pub input_json: String,
    /// Raw string output returned by the tool executor.
    pub output: String,
    /// 1-based TAO iteration index at which this call was made.
    pub iteration: u8,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub agent_id: AgentId,
    pub output: String,
    pub token_cost: u64,
    pub error: Option<String>,
    /// Ordered list of tool calls made during the local TAO loop. Empty for direct (no-tool) executions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallRecord>,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentHeartbeat {
    pub agent_id: AgentId,
    pub descriptor: AgentDescriptor,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub active_tasks: u32,
}

#[typeshare]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
        args: Vec<String>,
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
