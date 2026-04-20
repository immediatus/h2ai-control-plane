use crate::error::MemoryError;
use crate::provider::MemoryProvider;
use async_nats::jetstream::kv::Store;
use async_trait::async_trait;

pub struct NatsKvStore {
    kv: Store,
}

impl NatsKvStore {
    pub fn new(kv: Store) -> Self {
        Self { kv }
    }
}

#[async_trait]
impl MemoryProvider for NatsKvStore {
    async fn get_recent_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>, MemoryError> {
        let key = format!("history.{session_id}");
        match self.kv.get(&key).await {
            Ok(Some(bytes)) => {
                let all: Vec<serde_json::Value> = serde_json::from_slice(&bytes)
                    .map_err(|e| MemoryError::Serialization(e.to_string()))?;
                Ok(all.into_iter().rev().take(limit).collect())
            }
            Ok(None) => Ok(vec![]),
            Err(e) => Err(MemoryError::Storage(e.to_string())),
        }
    }

    async fn commit_new_memories(
        &self,
        session_id: &str,
        memories: Vec<serde_json::Value>,
    ) -> Result<(), MemoryError> {
        let key = format!("history.{session_id}");
        // Load existing, append, store back
        let mut existing: Vec<serde_json::Value> = match self.kv.get(&key).await {
            Ok(Some(bytes)) => serde_json::from_slice(&bytes)
                .map_err(|e| MemoryError::Serialization(e.to_string()))?,
            Ok(None) => vec![],
            Err(e) => return Err(MemoryError::Storage(e.to_string())),
        };
        existing.extend(memories);
        let bytes =
            serde_json::to_vec(&existing).map_err(|e| MemoryError::Serialization(e.to_string()))?;
        self.kv
            .put(&key, bytes.into())
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn retrieve_relevant_context(
        &self,
        _session_id: &str,
        _query: &str,
    ) -> Result<Vec<String>, MemoryError> {
        // Semantic search not implemented for NATS KV
        Ok(vec![])
    }
}
