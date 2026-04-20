use crate::error::MemoryError;
use async_trait::async_trait;

#[async_trait]
pub trait MemoryProvider: Send + Sync {
    async fn get_recent_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>, MemoryError>;

    async fn commit_new_memories(
        &self,
        session_id: &str,
        memories: Vec<serde_json::Value>,
    ) -> Result<(), MemoryError>;

    async fn retrieve_relevant_context(
        &self,
        session_id: &str,
        query: &str,
    ) -> Result<Vec<String>, MemoryError>;
}
