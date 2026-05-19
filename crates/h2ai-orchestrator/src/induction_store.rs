use async_nats::jetstream::{self, kv};
use futures::StreamExt;
use h2ai_types::config::AgentRole;
use h2ai_types::knowledge::KnowledgeNodePattern;
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

pub struct InductionStore {
    kv: kv::Store,
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
        Ok(Self { kv: store })
    }

    fn role_str(role: &AgentRole) -> &'static str {
        match role {
            AgentRole::Coordinator => "coordinator",
            AgentRole::Executor | AgentRole::Custom { .. } => "executor",
            AgentRole::Evaluator => "evaluator",
            AgentRole::Synthesizer => "synthesizer",
        }
    }

    /// Key format: `knowledge.{node_id}.{role_str}`
    /// Uses dots as NATS JetStream KV hierarchy separator (not slashes — dots are the
    /// canonical separator for KV prefix/watch operations in async-nats).
    /// node_id slashes are replaced with hyphens as a safety measure.
    fn key(node_id: &str, role: &AgentRole) -> String {
        format!(
            "knowledge.{}.{}",
            node_id.replace('/', "-"),
            Self::role_str(role)
        )
    }

    /// Record a successful retrieval: increment hit_rate for each node_id under this role.
    /// Non-fatal — errors are returned to the caller; task execution is never gated on this.
    /// No CAS — hit_rate loss under concurrent writes is acceptable (additive best-effort).
    pub async fn record(
        &self,
        node_ids: &[String],
        role: &AgentRole,
        domain_tags: &[String],
    ) -> Result<(), InductionStoreError> {
        for node_id in node_ids {
            let key = Self::key(node_id, role);
            let mut pattern = self
                .get_pattern(&key)
                .await
                .unwrap_or(KnowledgeNodePattern {
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
            self.kv
                .put(&key, bytes)
                .await
                .map_err(|e| InductionStoreError::Put(e.to_string()))?;
        }
        Ok(())
    }

    /// Load patterns matching role + any domain_tag overlap. Returns top-10 by hit_rate.
    /// Empty domain_tags returns nothing (no overlap is possible).
    pub async fn load_patterns(
        &self,
        domain_tags: &[String],
        role: &AgentRole,
    ) -> Result<Vec<KnowledgeNodePattern>, InductionStoreError> {
        let role_suffix = format!(".{}", Self::role_str(role));
        let mut keys_stream = self
            .kv
            .keys()
            .await
            .map_err(|e| InductionStoreError::Keys(e.to_string()))?;

        let mut matched: Vec<KnowledgeNodePattern> = Vec::new();
        while let Some(key_result) = keys_stream.next().await {
            let key = match key_result {
                Ok(k) => k,
                Err(_) => continue,
            };
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
        match self.kv.get(key).await {
            Ok(Some(bytes)) => serde_json::from_slice::<KnowledgeNodePattern>(&bytes)
                .map_err(|e| {
                    tracing::warn!("InductionStore: corrupt pattern at key {key}: {e}");
                })
                .ok(),
            _ => None,
        }
    }
}
