use async_trait::async_trait;
use h2ai_constraints::source::{ConstraintError, ConstraintSource};
use h2ai_constraints::types::{ConstraintDoc, ConstraintMeta, ConstraintPayload};
use h2ai_constraints::wiki::WikiCache;
use h2ai_state::nats::NatsClient;
use std::sync::Arc;
use tokio::sync::RwLock;

/// NATS-backed ConstraintSource — reads from H2AI_CONSTRAINT_WIKI KV + H2AI_CONSTRAINT_PAYLOADS.
///
/// Phase 1 Bootstrap calls resolve_context — pure in-memory HashMap lookup, zero network I/O.
/// Phase 4 calls load_payload — one Object Store fetch per non-Static constraint per proposal.
pub struct NatsWikiConstraintSource {
    cache: Arc<RwLock<WikiCache>>,
    nats: Arc<NatsClient>,
}

impl NatsWikiConstraintSource {
    pub fn new(cache: Arc<RwLock<WikiCache>>, nats: Arc<NatsClient>) -> Self {
        Self { cache, nats }
    }
}

#[async_trait]
impl ConstraintSource for NatsWikiConstraintSource {
    async fn resolve_context(
        &self,
        task_tags: &[String],
        explicit_ids: &[String],
        query_text: &str,
    ) -> Vec<ConstraintMeta> {
        let cache = self.cache.read().await;
        // Fast path: explicit IDs — always return exactly what was requested.
        if !explicit_ids.is_empty() {
            return cache.resolve(&[], explicit_ids);
        }
        // Union: tag-based resolution ∪ BM25 semantic search.
        // Tags ensure mandatory domain constraints are never missed;
        // BM25 surfaces additional relevant constraints the task implies but didn't tag.
        if !task_tags.is_empty() || !query_text.is_empty() {
            let resolved = cache.resolve_with_semantic(task_tags, &[], query_text, 20);
            if !resolved.is_empty() {
                return resolved;
            }
        }
        // Return all metas for empty/unmatched context — NATS wiki is pre-filtered at load time.
        cache.metas.values().cloned().collect()
    }

    async fn load_payload(
        &self,
        id: &str,
        version: &str,
    ) -> Result<ConstraintPayload, ConstraintError> {
        // Check inline predicate in cache first (Static predicates need no fetch)
        {
            let cache = self.cache.read().await;
            if let Some(meta) = cache.metas.get(id) {
                if let Some(inline) = &meta.inline_predicate {
                    return Ok(ConstraintPayload {
                        id: id.to_string(),
                        version: meta.payload_version.clone(),
                        predicate: inline.clone(),
                    });
                }
            }
        }
        // Fetch from Object Store for LlmJudge / Oracle predicates
        self.nats
            .get_constraint_payload(id, version)
            .await
            .map_err(|e| ConstraintError::Unavailable(e.to_string()))?
            .ok_or_else(|| ConstraintError::NotFound(format!("{id}@{version}")))
    }

    fn revision(&self) -> u64 {
        self.cache.try_read().map(|c| c.revision).unwrap_or(0)
    }
}

/// Bridge function: converts resolved ConstraintMeta + lazy payload fetch into Vec<ConstraintDoc>.
///
/// Used in EngineInput construction until the engine is fully migrated to ConstraintSource
/// (Plan B). Non-Static predicates that fail to load are skipped with a warning.
pub async fn reconstruct_docs(
    metas: Vec<ConstraintMeta>,
    source: &dyn ConstraintSource,
) -> Vec<ConstraintDoc> {
    let mut docs = Vec::with_capacity(metas.len());
    for meta in metas {
        let predicate = if let Some(inline) = meta.inline_predicate.clone() {
            inline
        } else {
            match source.load_payload(&meta.id, &meta.payload_version).await {
                Ok(payload) => payload.predicate,
                Err(e) => {
                    tracing::warn!(
                        id = %meta.id,
                        "constraint payload load failed: {e}; skipping constraint"
                    );
                    continue;
                }
            }
        };
        docs.push(ConstraintDoc {
            id: meta.id.clone(),
            source_file: meta.source.unwrap_or_else(|| meta.id.clone()),
            description: meta.summary,
            severity: meta.severity,
            predicate,
            remediation_hint: None,
            domains: meta.domains,
            mandatory_for_tags: meta.mandatory_for_tags,
            related_to: meta.related_to,
        });
    }
    docs
}
