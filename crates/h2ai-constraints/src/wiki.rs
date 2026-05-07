use crate::types::{ConstraintDoc, ConstraintMeta};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// In-memory wiki index — the hot path for Phase 1 Bootstrap.
///
/// Loaded from NATS KV `H2AI_CONSTRAINT_WIKI` at startup; refreshed via KV watch.
/// Analogue of `calibration` and `bandit_state` in AppState: small, in-memory, NATS-backed.
/// Full corpus is never loaded; only applicable constraint metadata is held.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WikiCache {
    /// tag → Vec<constraint_id>: pre-answered set-intersection query.
    /// Built from `domains` and `mandatory_for_tags` of each ConstraintMeta.
    pub context_map: HashMap<String, Vec<String>>,
    /// id → ConstraintMeta: point lookup in O(1).
    pub metas: HashMap<String, ConstraintMeta>,
    /// NATS KV revision at load time — stored in ConstraintSnapshot for audit.
    pub revision: u64,
}

impl WikiCache {
    /// Build a WikiCache from a slice of ConstraintDoc (backward compat path).
    ///
    /// Used by FsConstraintSource to bootstrap from the flat directory.
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
            cache.metas.insert(meta.id.clone(), meta);
        }
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
}
