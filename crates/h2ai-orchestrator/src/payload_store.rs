use async_trait::async_trait;
use dashmap::DashMap;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use thiserror::Error;

use h2ai_types::agent::ContextPayload;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("payload not found for hash")]
    NotFound,
    #[error("store backend error: {0}")]
    Backend(String),
}

#[async_trait]
pub trait PayloadStore: Send + Sync {
    async fn put(&self, content: &str) -> Result<[u8; 32], StoreError>;
    async fn get(&self, hash: &[u8; 32]) -> Result<String, StoreError>;
}

/// In-memory content-addressed store. Zero external dependencies; used in tests and local dev.
/// Two puts of identical content return the same hash (deduplication).
pub struct MemoryPayloadStore {
    inner: Arc<DashMap<[u8; 32], String>>,
}

impl MemoryPayloadStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }
}

impl Default for MemoryPayloadStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PayloadStore for MemoryPayloadStore {
    async fn put(&self, content: &str) -> Result<[u8; 32], StoreError> {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash: [u8; 32] = hasher.finalize().into();
        self.inner
            .entry(hash)
            .or_insert_with(|| content.to_string());
        Ok(hash)
    }

    async fn get(&self, hash: &[u8; 32]) -> Result<String, StoreError> {
        self.inner
            .get(hash)
            .map(|v| v.value().clone())
            .ok_or(StoreError::NotFound)
    }
}

/// Resolve a `ContextPayload` to its content string.
/// `Inline` returns immediately. `Ref` fetches from the store by SHA-256 hex hash.
pub async fn resolve_context(
    payload: &ContextPayload,
    store: &dyn PayloadStore,
) -> Result<String, StoreError> {
    match payload {
        ContextPayload::Inline(s) => Ok(s.clone()),
        ContextPayload::Ref { hash, .. } => {
            let bytes = hex::decode(hash)
                .map_err(|e| StoreError::Backend(format!("invalid hash hex: {e}")))?;
            if bytes.len() != 32 {
                return Err(StoreError::Backend(format!(
                    "hash must be 32 bytes, got {}",
                    bytes.len()
                )));
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            store.get(&arr).await
        }
    }
}

/// Return `ContextPayload::Inline` when `content.len() <= threshold`, otherwise offload to
/// the store and return `ContextPayload::Ref { hash, byte_len }`.
pub async fn offload_if_large(
    content: String,
    threshold: usize,
    store: &dyn PayloadStore,
) -> Result<ContextPayload, StoreError> {
    if content.len() <= threshold {
        return Ok(ContextPayload::Inline(content));
    }
    let byte_len = content.len();
    let hash_bytes = store.put(&content).await?;
    Ok(ContextPayload::Ref {
        hash: hex::encode(hash_bytes),
        byte_len,
    })
}
