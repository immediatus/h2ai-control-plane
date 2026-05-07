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
    async fn resolve_context(
        &self,
        task_tags: &[String],
        explicit_ids: &[String],
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
    async fn resolve_context(
        &self,
        task_tags: &[String],
        explicit_ids: &[String],
    ) -> Vec<ConstraintMeta> {
        if task_tags.is_empty() && explicit_ids.is_empty() {
            return self.docs.iter().map(ConstraintMeta::from_doc).collect();
        }
        let resolved = self.cache.resolve(task_tags, explicit_ids);
        // When tags are provided but the corpus has no domain metadata (no frontmatter),
        // context_map will be empty and resolve returns nothing. Fall back to all docs
        // so callers with constraint_tags set don't silently get zero constraints.
        if resolved.is_empty() && explicit_ids.is_empty() {
            return self.docs.iter().map(ConstraintMeta::from_doc).collect();
        }
        resolved
    }

    async fn load_payload(
        &self,
        id: &str,
        _version: &str,
    ) -> Result<ConstraintPayload, ConstraintError> {
        let meta = self
            .cache
            .metas
            .get(id)
            .ok_or_else(|| ConstraintError::NotFound(id.to_string()))?;
        let predicate = meta.inline_predicate.clone().ok_or_else(|| {
            ConstraintError::NotFound(format!(
                "{id}: non-static predicate requires NatsWikiConstraintSource"
            ))
        })?;
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
