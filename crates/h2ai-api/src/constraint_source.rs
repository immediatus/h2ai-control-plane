use async_trait::async_trait;
use h2ai_constraints::index::ConstraintIndex;
use h2ai_constraints::source::ConstraintError;
use h2ai_constraints::store::ConstraintStore;
use h2ai_constraints::types::{ConstraintDoc, ConstraintMeta};
use h2ai_state::nats::NatsClient;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

const TAG_INDEX_TTL: Duration = Duration::from_secs(300);

struct CachedTagIndex {
    index: HashMap<String, Vec<String>>,
    loaded_at: Instant,
}

/// NATS-backed `ConstraintIndex` — lazy, never bulk-loads the corpus.
///
/// Caches the compact tag→[id] map with a 5-minute TTL.
/// Individual constraint IDs are validated against NATS KV on demand.
pub struct NatsConstraintIndex {
    nats: Arc<NatsClient>,
    tag_cache: Arc<RwLock<Option<CachedTagIndex>>>,
}

impl NatsConstraintIndex {
    pub fn new(nats: Arc<NatsClient>) -> Self {
        Self {
            nats,
            tag_cache: Arc::new(RwLock::new(None)),
        }
    }

    async fn tag_index(&self) -> HashMap<String, Vec<String>> {
        {
            let guard = self.tag_cache.read().await;
            if let Some(c) = guard.as_ref() {
                if c.loaded_at.elapsed() < TAG_INDEX_TTL {
                    return c.index.clone();
                }
            }
        }
        let fetched = match self.nats.get_tag_index().await {
            Ok(Some(idx)) => idx,
            Ok(None) => {
                tracing::debug!(target: "h2ai.constraints", "tag index not yet bootstrapped in NATS");
                HashMap::new()
            }
            Err(e) => {
                tracing::warn!(target: "h2ai.constraints", error = %e, "failed to fetch tag index");
                HashMap::new()
            }
        };
        *self.tag_cache.write().await = Some(CachedTagIndex {
            index: fetched.clone(),
            loaded_at: Instant::now(),
        });
        fetched
    }
}

#[async_trait]
impl ConstraintIndex for NatsConstraintIndex {
    async fn find_by_ids(&self, ids: &[String]) -> Vec<String> {
        // Validate existence — fetch meta for each, keep those that exist
        let metas = match self.nats.get_constraint_metas(ids).await {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(target: "h2ai.constraints", error = %e, "id validation failed");
                return ids.to_vec();
            }
        };
        metas.into_iter().map(|m| m.id).collect()
    }

    async fn find_by_tags(&self, tags: &[String]) -> Vec<String> {
        let index = self.tag_index().await;
        let mut ids: HashSet<String> = HashSet::new();
        for tag in tags {
            if let Some(tag_ids) = index.get(tag.as_str()) {
                ids.extend(tag_ids.iter().cloned());
            }
        }
        ids.into_iter().collect()
    }

    async fn search(&self, _query: &str, _top_k: usize) -> Vec<String> {
        // Semantic search over a large corpus requires an external search service.
        // Not implemented for NATS backend — use explicit IDs or tags.
        tracing::debug!(target: "h2ai.constraints", "NATS semantic search not implemented; use explicit IDs or tags");
        vec![]
    }
}

/// NATS-backed `ConstraintStore` — fetches individual docs on demand, never bulk-loads.
pub struct NatsConstraintStore {
    nats: Arc<NatsClient>,
}

impl NatsConstraintStore {
    pub fn new(nats: Arc<NatsClient>) -> Self {
        Self { nats }
    }

    async fn load_from_nats(&self, id: &str) -> Result<ConstraintDoc, ConstraintError> {
        let meta: ConstraintMeta = self
            .nats
            .get_constraint_meta(id)
            .await
            .map_err(|e| ConstraintError::Unavailable(e.to_string()))?
            .ok_or_else(|| ConstraintError::NotFound(id.to_string()))?;

        let predicate = if let Some(inline) = meta.inline_predicate.clone() {
            inline
        } else {
            let payload = self
                .nats
                .get_constraint_payload(id, &meta.payload_version)
                .await
                .map_err(|e| ConstraintError::Unavailable(e.to_string()))?
                .ok_or_else(|| {
                    ConstraintError::NotFound(format!("{id}@{}", meta.payload_version))
                })?;
            payload.predicate
        };

        Ok(ConstraintDoc {
            id: meta.id.clone(),
            source_file: meta.source.unwrap_or_else(|| meta.id.clone()),
            description: meta.summary,
            severity: meta.severity,
            predicate,
            remediation_hint: None,
            domains: meta.domains,
            mandatory_for_tags: meta.mandatory_for_tags,
            related_to: meta.related_to,
        })
    }
}

#[async_trait]
impl ConstraintStore for NatsConstraintStore {
    async fn load(&self, id: &str) -> Result<ConstraintDoc, ConstraintError> {
        self.load_from_nats(id).await
    }
}
