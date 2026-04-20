use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("session not found: {session_id}")]
    SessionNotFound { session_id: String },
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("storage error: {0}")]
    Storage(String),
}
