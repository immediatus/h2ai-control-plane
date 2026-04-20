use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProvisionError {
    #[error("agent not available: {agent_id}")]
    AgentUnavailable { agent_id: String },
    #[error("capacity limit reached for {agent_type}")]
    CapacityLimitReached { agent_type: String },
    #[error("transport error: {0}")]
    Transport(String),
    #[error("internal error: {0}")]
    Internal(String),
}
