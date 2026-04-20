use crate::error::MemoryError;
use crate::provider::MemoryProvider;
use async_nats::jetstream::kv::UpdateErrorKind;
use async_nats::jetstream::{self, kv};
use async_trait::async_trait;

pub struct NatsKvStore {
    kv: kv::Store,
}

impl NatsKvStore {
    /// Creates the named KV bucket. If the bucket already exists, the server
    /// returns it unchanged (async-nats behavior).
    pub async fn create(nats: &async_nats::Client, bucket: &str) -> Result<Self, MemoryError> {
        let js = jetstream::new(nats.clone());
        let store = js
            .create_key_value(kv::Config {
                bucket: bucket.to_string(),
                description: "H2AI durable session memory".to_string(),
                history: 1,
                storage: jetstream::stream::StorageType::File,
                ..Default::default()
            })
            .await
            .map_err(|e| MemoryError::Storage(e.to_string()))?;
        Ok(Self { kv: store })
    }

    /// Wrap a pre-created KV store (used when the store is created by `ensure_infrastructure`).
    pub fn new(kv: kv::Store) -> Self {
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
                // Return the last `limit` entries in chronological (oldest-first) order.
                let start = all.len().saturating_sub(limit);
                Ok(all[start..].to_vec())
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
        // CAS retry loop: read current revision, append, update with revision check.
        // Retries up to 8 times before giving up (guards against thundering-herd bursts).
        for _ in 0..8u8 {
            let (mut existing, revision) = match self
                .kv
                .entry(&key)
                .await
                .map_err(|e| MemoryError::Storage(e.to_string()))?
            {
                Some(e) => {
                    let v: Vec<serde_json::Value> = serde_json::from_slice(&e.value)
                        .map_err(|e| MemoryError::Serialization(e.to_string()))?;
                    (v, Some(e.revision))
                }
                None => (Vec::new(), None),
            };
            existing.extend_from_slice(&memories);
            let bytes: bytes::Bytes = serde_json::to_vec(&existing)
                .map_err(|e| MemoryError::Serialization(e.to_string()))?
                .into();

            match revision {
                Some(rev) => {
                    match self.kv.update(&key, bytes, rev).await {
                        Ok(_) => return Ok(()),
                        Err(e) => {
                            if e.kind() == UpdateErrorKind::WrongLastRevision {
                                // CAS miss: another writer updated concurrently — re-read and retry.
                                continue;
                            }
                            return Err(MemoryError::Storage(e.to_string()));
                        }
                    }
                }
                // First write: no revision yet; `put` is the only option.
                // Concurrent first-writers: second put overwrites first — acceptable
                // because sessions are owned by a single pipeline instance at creation.
                None => {
                    self.kv
                        .put(&key, bytes)
                        .await
                        .map_err(|e| MemoryError::Storage(e.to_string()))?;
                    return Ok(());
                }
            }
        }
        Err(MemoryError::Storage(
            "commit_new_memories: too many concurrent writers for session".into(),
        ))
    }

    async fn retrieve_relevant_context(
        &self,
        _session_id: &str,
        _query: &str,
    ) -> Result<Vec<String>, MemoryError> {
        Ok(vec![])
    }
}
