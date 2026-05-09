use crate::loader::load_corpus;
use crate::types::{ConstraintDoc, ConstraintMeta, ConstraintPayload};
use crate::wiki::WikiCache;
use async_trait::async_trait;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConstraintError {
    #[error("constraint not found: {0}")]
    NotFound(String),
    #[error("store unavailable: {0}")]
    Unavailable(String),
    #[error("deserialize error: {0}")]
    Deserialize(String),
}

/// Abstraction over constraint corpus access.
///
/// `FsConstraintSource` wraps the existing flat-directory behavior (backward compat).
/// `NatsWikiConstraintSource` (in h2ai-api) reads from NATS KV + Object Store.
#[async_trait]
pub trait ConstraintSource: Send + Sync {
    /// Phase 1: resolve applicable ConstraintMeta for given task context.
    ///
    /// Resolution order:
    /// 1. Explicit IDs (`explicit_ids`) — always included, O(1) lookup
    /// 2. Tag intersection (`task_tags`) — domain/mandatory_for_tags index
    /// 3. BM25 semantic fallback — when tags match nothing, use `query_text`
    ///    to surface the most relevant constraints by keyword similarity
    async fn resolve_context(
        &self,
        task_tags: &[String],
        explicit_ids: &[String],
        query_text: &str,
    ) -> Vec<ConstraintMeta>;

    /// Phase 4: fetch full predicate payload for a specific constraint on demand.
    async fn load_payload(
        &self,
        id: &str,
        version: &str,
    ) -> Result<ConstraintPayload, ConstraintError>;

    /// NATS KV revision at cache load time — stored in ConstraintSnapshot for audit.
    fn revision(&self) -> u64;
}

/// Filesystem-backed source wrapping the existing `load_corpus` behavior.
pub struct FsConstraintSource {
    cache: WikiCache,
    docs: Vec<ConstraintDoc>,
}

impl FsConstraintSource {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let docs = load_corpus(dir)?;
        let cache = WikiCache::from_docs(&docs);
        Ok(Self { cache, docs })
    }

    pub fn from_docs(docs: Vec<ConstraintDoc>) -> Self {
        let cache = WikiCache::from_docs(&docs);
        Self { cache, docs }
    }

    pub fn all_docs(&self) -> &[ConstraintDoc] {
        &self.docs
    }
}

#[async_trait]
impl ConstraintSource for FsConstraintSource {
    /// Resolve applicable constraints using two-stage strategy:
    ///
    /// 1. **Explicit IDs** — fast-path: return exactly what was requested; O(1) lookup
    /// 2. **Tags ∪ BM25** — union of tag intersection and semantic search; ensures
    ///    domain-mandatory constraints (tags) are never missed while BM25 surfaces
    ///    semantically relevant constraints the tags didn't cover
    async fn resolve_context(
        &self,
        task_tags: &[String],
        explicit_ids: &[String],
        query_text: &str,
    ) -> Vec<ConstraintMeta> {
        // Fast path: explicit IDs always win — return exactly what was requested.
        if !explicit_ids.is_empty() {
            return self.cache.resolve(&[], explicit_ids);
        }

        // Union: tag-based resolution ∪ BM25 semantic search.
        // Tags ensure mandatory domain constraints (billing, GDPR) are always included.
        // BM25 surfaces additional relevant constraints the task implies but didn't tag.
        if !task_tags.is_empty() || !query_text.is_empty() {
            let resolved = self
                .cache
                .resolve_with_semantic(task_tags, &[], query_text, 20);
            if !resolved.is_empty() {
                return resolved;
            }
        }

        // Final fallback: corpus empty or no terms matched — return all.
        self.docs.iter().map(ConstraintMeta::from_doc).collect()
    }

    async fn load_payload(
        &self,
        id: &str,
        _version: &str,
    ) -> Result<ConstraintPayload, ConstraintError> {
        // FsConstraintSource holds the full ConstraintDoc in memory, so it can serve
        // any predicate tier including LlmJudge — no external fetch required.
        // load_corpus() deduplicates by ID (YAML preferred over MD), so there is at
        // most one doc per ID in self.docs.
        if let Some(doc) = self.docs.iter().find(|d| d.id == id) {
            return Ok(ConstraintPayload {
                id: id.to_string(),
                version: "v1".to_string(),
                predicate: doc.predicate.clone(),
            });
        }
        // Fall back to meta inline_predicate for any cache-only entries.
        let meta = self
            .cache
            .metas
            .get(id)
            .ok_or_else(|| ConstraintError::NotFound(id.to_string()))?;
        let predicate = meta
            .inline_predicate
            .clone()
            .ok_or_else(|| ConstraintError::NotFound(id.to_string()))?;
        Ok(ConstraintPayload {
            id: id.to_string(),
            version: meta.payload_version.clone(),
            predicate,
        })
    }

    fn revision(&self) -> u64 {
        self.cache.revision
    }
}
