use h2ai_types::identity::{AgentId, TaskId};

pub const TASKS_EPHEMERAL_PREFIX: &str = "h2ai.tasks.ephemeral";
pub const RESULTS_PREFIX: &str = "h2ai.results";
pub const TELEMETRY_WILDCARD: &str = "h2ai.telemetry.*";
pub const AUDIT_EVENTS_WILDCARD: &str = "audit.events.*";
pub const HEARTBEAT_PREFIX: &str = "h2ai.heartbeat";
pub const CONTROL_PREFIX: &str = "h2ai.control";

/// Subject on which the control plane publishes a TaskPayload for an ephemeral edge agent.
pub fn ephemeral_task_subject(task_id: &TaskId) -> String {
    format!("{TASKS_EPHEMERAL_PREFIX}.{task_id}")
}

/// Subject on which the edge agent publishes its completed TaskResult.
/// Control plane subscribes here after dispatching the TaskPayload.
pub fn task_result_subject(task_id: &TaskId) -> String {
    format!("{RESULTS_PREFIX}.{task_id}")
}

pub fn agent_telemetry_subject(agent_id: &AgentId) -> String {
    format!("h2ai.telemetry.{agent_id}")
}

pub fn agent_terminate_subject(agent_id: &AgentId) -> String {
    format!("{CONTROL_PREFIX}.terminate.{agent_id}")
}

pub fn audit_event_subject(agent_id: &AgentId) -> String {
    format!("audit.events.{agent_id}")
}
