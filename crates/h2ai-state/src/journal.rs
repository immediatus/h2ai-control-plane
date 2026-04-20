use async_trait::async_trait;
use h2ai_types::events::H2AIEvent;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JournalError {
    #[error("append failed: {0}")]
    AppendFailed(String),
    #[error("replay failed: {0}")]
    ReplayFailed(String),
}

#[async_trait]
pub trait JournalBackend: Send + Sync {
    async fn append(&self, event: H2AIEvent) -> Result<(), JournalError>;
    async fn read_from(&self, offset: usize) -> Result<Vec<H2AIEvent>, JournalError>;
}

pub struct InMemoryBackend {
    log: Arc<Mutex<Vec<H2AIEvent>>>,
}

impl InMemoryBackend {
    pub fn new() -> Self {
        Self {
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Default for InMemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl JournalBackend for InMemoryBackend {
    async fn append(&self, event: H2AIEvent) -> Result<(), JournalError> {
        self.log.lock().unwrap().push(event);
        Ok(())
    }

    async fn read_from(&self, offset: usize) -> Result<Vec<H2AIEvent>, JournalError> {
        let log = self.log.lock().unwrap();
        Ok(log[offset.min(log.len())..].to_vec())
    }
}

pub struct EventJournal<B: JournalBackend> {
    backend: B,
}

impl<B: JournalBackend> EventJournal<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub async fn append(&self, event: H2AIEvent) -> Result<(), JournalError> {
        self.backend.append(event).await
    }

    pub async fn replay(&self, from_offset: usize) -> Result<Vec<H2AIEvent>, JournalError> {
        self.backend.read_from(from_offset).await
    }
}
