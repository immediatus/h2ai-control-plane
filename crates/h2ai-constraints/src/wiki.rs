use crate::retrieval::{ConstraintCandidate, ConstraintRetriever};
use crate::types::{ConstraintDoc, ConstraintMeta};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// In-memory wiki index — the hot path for Phase 1 Bootstrap.
///
/// Loaded from NATS KV `H2AI_CONSTRAINT_WIKI` at startup; refreshed via KV watch.
/// Analogue of `calibration` and `bandit_state` in AppState: small, in-memory, NATS-backed.
/// Full corpus is never loaded; only applicable constraint metadata is held.
///
/// ## Constraint resolution pipeline
///
/// Constraint selection uses a two-stage approach:
///
/// **Stage 1 — Exact lookup** (O(tags), μs):
/// `context_map` resolves mandatory constraints by tag intersection. This guarantees
/// that domain-specific constraints (e.g., `billing`, `eu_data`) are always included
/// when the task declares the matching domain tag. Never skipped.
///
/// **Stage 2 — BM25 semantic retrieval** (O(corpus), ms):
/// `retriever` surfaces additional constraints the task description implies but
/// didn't explicitly tag. Useful when task descriptions are rich prose rather than
/// structured tags, and essential at corpus scale (>10K constraints).
///
/// ## Relation graph
///
/// `relations` stores the explicit constraint dependency graph built from `related_to`
/// fields in YAML constraint files. Navigation is O(1) lookup: given a constraint ID,
/// return all directly related constraint IDs.
///
/// ### Navigation query patterns
///
/// ```text
/// // "What else must I satisfy if CONSTRAINT-004 applies?"
/// wiki.navigate_related("CONSTRAINT-004")  →  [CONSTRAINT-005, CONSTRAINT-007]
///
/// // "All billing-domain constraints"
/// wiki.navigate_by_domain("billing")  →  [CONSTRAINT-004, CONSTRAINT-005, CONSTRAINT-007]
///
/// // "Semantic search — top-5 constraints relevant to this task description"
/// wiki.search("atomic budget deduction idempotency redis", 5)  →  [...]
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WikiCache {
    /// tag → Vec<constraint_id>: pre-answered set-intersection query.
    /// Built from `domains` and `mandatory_for_tags` of each ConstraintMeta.
    pub context_map: HashMap<String, Vec<String>>,
    /// id → ConstraintMeta: O(1) point lookup.
    pub metas: HashMap<String, ConstraintMeta>,
    /// Explicit constraint relation graph: constraint_id → related constraint_ids.
    /// Built from the `related_to` field in YAML constraint files.
    /// Edges are unidirectional as declared; callers may traverse both directions
    /// by querying `relations[id]` and all metas where `related_to.contains(id)`.
    pub relations: HashMap<String, Vec<String>>,
    /// NATS KV revision at load time — stored in ConstraintSnapshot for audit.
    pub revision: u64,
    /// BM25 retrieval index for semantic search over the constraint corpus.
    /// Indexed text: constraint id + description + rubric terms.
    #[serde(skip)]
    pub retriever: Option<ConstraintRetriever>,
}

impl WikiCache {
    /// Build a WikiCache from a slice of ConstraintDoc (backward compat path).
    ///
    /// Used by FsConstraintSource to bootstrap from the flat directory.
    /// Builds the BM25 retriever alongside the tag index.
    pub fn from_docs(docs: &[ConstraintDoc]) -> Self {
        let mut cache = WikiCache::default();
        for doc in docs {
            let meta = ConstraintMeta::from_doc(doc);
            for domain in &meta.domains {
                cache
                    .context_map
                    .entry(domain.clone())
                    .or_default()
                    .push(meta.id.clone());
            }
            for tag in &meta.mandatory_for_tags {
                cache
                    .context_map
                    .entry(tag.clone())
                    .or_default()
                    .push(meta.id.clone());
            }
            // Build relation graph from explicit cross-references.
            if !meta.related_to.is_empty() {
                cache
                    .relations
                    .insert(meta.id.clone(), meta.related_to.clone());
            }
            cache.metas.insert(meta.id.clone(), meta);
        }
        cache.retriever = Some(ConstraintRetriever::from_docs(docs));
        cache
    }

    /// Resolve applicable ConstraintMeta for the given task context.
    ///
    /// Returns the union of constraints matched by tag lookup and explicit ID override.
    /// Deduplicates: a constraint matched by both tag and explicit ID appears once.
    pub fn resolve(&self, task_tags: &[String], explicit_ids: &[String]) -> Vec<ConstraintMeta> {
        let mut ids: HashSet<String> = explicit_ids.iter().cloned().collect();
        for tag in task_tags {
            if let Some(constraint_ids) = self.context_map.get(tag.as_str()) {
                ids.extend(constraint_ids.iter().cloned());
            }
        }
        ids.into_iter()
            .filter_map(|id| self.metas.get(&id).cloned())
            .collect()
    }

    /// BM25 semantic search over the indexed constraint corpus.
    ///
    /// Returns up to `top_k` constraints ranked by BM25 relevance score.
    /// Useful for surfaces constraints relevant to a task description when tags are
    /// sparse or absent. Complements `resolve()` for the two-stage retrieval pipeline.
    ///
    /// Returns an empty vec if the retriever is not built (e.g. deserialized from NATS KV
    /// without reconstruction — call `rebuild_retriever()` first).
    pub fn search(&self, query_text: &str, top_k: usize) -> Vec<ConstraintCandidate> {
        match &self.retriever {
            Some(r) => r.query(query_text, top_k),
            None => vec![],
        }
    }

    /// Two-stage resolution: exact tag lookup ∪ BM25 semantic search, deduplicated.
    ///
    /// Use this when the task carries both structured tags (domain, mandatory constraints)
    /// and free-text description. The union ensures:
    /// - Domain-critical constraints (billing, GDPR) are never missed via tags
    /// - Semantically relevant constraints surface even without explicit tagging
    pub fn resolve_with_semantic(
        &self,
        task_tags: &[String],
        explicit_ids: &[String],
        query_text: &str,
        semantic_top_k: usize,
    ) -> Vec<ConstraintMeta> {
        let mut ids: HashSet<String> = HashSet::new();

        // Stage 1: exact tag/id lookup
        for id in explicit_ids {
            ids.insert(id.clone());
        }
        for tag in task_tags {
            if let Some(constraint_ids) = self.context_map.get(tag.as_str()) {
                ids.extend(constraint_ids.iter().cloned());
            }
        }

        // Stage 2: BM25 semantic retrieval
        if !query_text.is_empty() {
            let semantic_hits = self.search(query_text, semantic_top_k);
            for hit in semantic_hits {
                ids.insert(hit.id);
            }
        }

        ids.into_iter()
            .filter_map(|id| self.metas.get(&id).cloned())
            .collect()
    }

    /// Navigate the explicit constraint relation graph.
    ///
    /// Returns the directly related constraints for the given constraint ID.
    /// Only outgoing edges declared in `related_to` are returned. Returns an empty
    /// vec if the constraint has no declared relations or the ID is not found.
    ///
    /// Example: `navigate_related("CONSTRAINT-004")` → constraints referenced in its
    /// `related_to` field (e.g. CONSTRAINT-005, CONSTRAINT-007 for budget pacing).
    pub fn navigate_related(&self, id: &str) -> Vec<ConstraintMeta> {
        let related_ids = match self.relations.get(id) {
            Some(ids) => ids,
            None => return vec![],
        };
        related_ids
            .iter()
            .filter_map(|related_id| self.metas.get(related_id).cloned())
            .collect()
    }

    /// Return all constraints in a given domain.
    ///
    /// Equivalent to `resolve(tags: [domain], explicit_ids: [])` but returns the metas
    /// directly without the fallback-to-all logic in `FsConstraintSource`.
    pub fn navigate_by_domain(&self, domain: &str) -> Vec<ConstraintMeta> {
        let ids = match self.context_map.get(domain) {
            Some(ids) => ids,
            None => return vec![],
        };
        ids.iter()
            .filter_map(|id| self.metas.get(id).cloned())
            .collect()
    }

    /// Rebuild the BM25 retriever from the current metas.
    ///
    /// Required after deserialization from NATS KV, since the retriever is not serialized.
    /// The docs parameter must match the metas in this cache.
    pub fn rebuild_retriever(&mut self, docs: &[ConstraintDoc]) {
        self.retriever = Some(ConstraintRetriever::from_docs(docs));
    }
}
