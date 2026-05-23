use async_nats::jetstream::{self, kv};
use async_trait::async_trait;
use futures::StreamExt;
use h2ai_types::config::AgentRole;
use h2ai_types::knowledge::KnowledgeNodePattern;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InductionStoreError {
    #[error("InductionStore create: {0}")]
    Create(String),
    #[error("InductionStore keys: {0}")]
    Keys(String),
    #[error("InductionStore put: {0}")]
    Put(String),
    #[error("InductionStore serialize: {0}")]
    Serialize(String),
}

#[async_trait]
pub trait KvBackend: Send + Sync {
    async fn get(&self, key: &str) -> Option<bytes::Bytes>;
    async fn put(&self, key: &str, value: bytes::Bytes) -> Result<(), InductionStoreError>;
    async fn all_keys(&self) -> Result<Vec<String>, InductionStoreError>;
}

/// In-memory KV backend for unit tests — no NATS, no I/O.
/// Stores entries in a `HashMap` protected by a `Mutex`.
#[derive(Default)]
pub struct InMemoryKvBackend {
    data: Mutex<HashMap<String, bytes::Bytes>>,
}

#[async_trait]
impl KvBackend for InMemoryKvBackend {
    async fn get(&self, key: &str) -> Option<bytes::Bytes> {
        self.data.lock().unwrap().get(key).cloned()
    }

    async fn put(&self, key: &str, value: bytes::Bytes) -> Result<(), InductionStoreError> {
        self.data.lock().unwrap().insert(key.to_string(), value);
        Ok(())
    }

    async fn all_keys(&self) -> Result<Vec<String>, InductionStoreError> {
        Ok(self.data.lock().unwrap().keys().cloned().collect())
    }
}

struct NatsKvBackend {
    kv: kv::Store,
}

#[async_trait]
impl KvBackend for NatsKvBackend {
    async fn get(&self, key: &str) -> Option<bytes::Bytes> {
        match self.kv.get(key).await {
            Ok(Some(b)) => Some(b),
            _ => None,
        }
    }

    async fn put(&self, key: &str, value: bytes::Bytes) -> Result<(), InductionStoreError> {
        self.kv
            .put(key, value)
            .await
            .map(|_| ())
            .map_err(|e| InductionStoreError::Put(e.to_string()))
    }

    async fn all_keys(&self) -> Result<Vec<String>, InductionStoreError> {
        let mut stream = self
            .kv
            .keys()
            .await
            .map_err(|e| InductionStoreError::Keys(e.to_string()))?;
        let mut keys = Vec::new();
        while let Some(result) = stream.next().await {
            if let Ok(k) = result {
                keys.push(k);
            }
        }
        Ok(keys)
    }
}

pub struct InductionStore {
    backend: Arc<dyn KvBackend>,
}

impl InductionStore {
    pub async fn create(
        nats: &async_nats::Client,
        bucket: &str,
    ) -> Result<Self, InductionStoreError> {
        let js = jetstream::new(nats.clone());
        let store = js
            .create_key_value(kv::Config {
                bucket: bucket.to_string(),
                description: "H2AI induction knowledge patterns".to_string(),
                history: 1,
                storage: jetstream::stream::StorageType::File,
                ..Default::default()
            })
            .await
            .map_err(|e| InductionStoreError::Create(e.to_string()))?;
        Ok(Self {
            backend: Arc::new(NatsKvBackend { kv: store }),
        })
    }

    /// Construct from a custom backend (used in tests).
    pub fn from_backend(backend: Arc<dyn KvBackend>) -> Self {
        Self { backend }
    }

    const fn role_str(role: &AgentRole) -> &'static str {
        match role {
            AgentRole::Coordinator => "coordinator",
            AgentRole::Executor | AgentRole::Custom { .. } => "executor",
            AgentRole::Evaluator => "evaluator",
            AgentRole::Synthesizer => "synthesizer",
        }
    }

    /// Key format: `knowledge.{node_id}.{role_str}`
    fn key(node_id: &str, role: &AgentRole) -> String {
        format!(
            "knowledge.{}.{}",
            node_id.replace('/', "-"),
            Self::role_str(role)
        )
    }

    /// Record a successful retrieval: increment `hit_rate` for each `node_id` under this role.
    pub async fn record(
        &self,
        node_ids: &[String],
        role: &AgentRole,
        domain_tags: &[String],
    ) -> Result<(), InductionStoreError> {
        for node_id in node_ids {
            let key = Self::key(node_id, role);
            let mut pattern =
                self.get_pattern(&key)
                    .await
                    .unwrap_or_else(|| KnowledgeNodePattern {
                        node_id: node_id.clone(),
                        role: role.clone(),
                        domain_tags: domain_tags.to_vec(),
                        hit_rate: 0.0,
                    });
            pattern.hit_rate += 1.0;
            for tag in domain_tags {
                if !pattern.domain_tags.contains(tag) {
                    pattern.domain_tags.push(tag.clone());
                }
            }
            let bytes: bytes::Bytes = serde_json::to_vec(&pattern)
                .map_err(|e| InductionStoreError::Serialize(e.to_string()))?
                .into();
            self.backend.put(&key, bytes).await?;
        }
        Ok(())
    }

    /// Load patterns matching role + any `domain_tag` overlap. Returns top-10 by `hit_rate`.
    pub async fn load_patterns(
        &self,
        domain_tags: &[String],
        role: &AgentRole,
    ) -> Result<Vec<KnowledgeNodePattern>, InductionStoreError> {
        let role_suffix = format!(".{}", Self::role_str(role));
        let keys = self.backend.all_keys().await?;

        let mut matched: Vec<KnowledgeNodePattern> = Vec::new();
        for key in keys {
            if !key.ends_with(&role_suffix) {
                continue;
            }
            if let Some(pattern) = self.get_pattern(&key).await {
                if domain_tags.iter().any(|t| pattern.domain_tags.contains(t)) {
                    matched.push(pattern);
                }
            }
        }
        matched.sort_by(|a, b| {
            b.hit_rate
                .partial_cmp(&a.hit_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matched.truncate(10);
        Ok(matched)
    }

    async fn get_pattern(&self, key: &str) -> Option<KnowledgeNodePattern> {
        let bytes = self.backend.get(key).await?;
        serde_json::from_slice::<KnowledgeNodePattern>(&bytes)
            .map_err(|e| {
                tracing::warn!("InductionStore: corrupt pattern at key {key}: {e}");
            })
            .ok()
    }
}
