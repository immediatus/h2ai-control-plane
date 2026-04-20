use crate::error::MemoryError;
use crate::provider::MemoryProvider;
use async_trait::async_trait;
use dashmap::DashMap;

pub struct InMemoryCache {
    store: DashMap<String, Vec<serde_json::Value>>,
}

impl InMemoryCache {
    pub fn new() -> Self {
        Self {
            store: DashMap::new(),
        }
    }
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryProvider for InMemoryCache {
    async fn get_recent_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>, MemoryError> {
        Ok(self
            .store
            .get(session_id)
            .map(|v| v.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default())
    }

    async fn commit_new_memories(
        &self,
        session_id: &str,
        memories: Vec<serde_json::Value>,
    ) -> Result<(), MemoryError> {
        self.store
            .entry(session_id.to_string())
            .or_default()
            .extend(memories);
        Ok(())
    }

    async fn retrieve_relevant_context(
        &self,
        _session_id: &str,
        _query: &str,
    ) -> Result<Vec<String>, MemoryError> {
        // Semantic search not implemented for in-memory cache
        Ok(vec![])
    }
}
