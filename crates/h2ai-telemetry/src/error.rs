use thiserror::Error;

#[derive(Debug, Error)]
pub enum AuditError {
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("flush error: {0}")]
    Flush(String),
}
