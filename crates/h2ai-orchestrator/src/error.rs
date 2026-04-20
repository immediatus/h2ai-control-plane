use thiserror::Error;

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("memory error: {0}")]
    Memory(String),
    #[error("provisioning error: {0}")]
    Provision(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("telemetry error: {0}")]
    Telemetry(String),
    #[error("timeout waiting for task result: task_id={task_id}")]
    Timeout { task_id: String },
    #[error("deserialize error: {0}")]
    Deserialize(String),
}
