use crate::index::ConstraintIndex;
use crate::loader::load_corpus;
use crate::retrieval::ConstraintRetriever;
use crate::store::ConstraintStore;
use crate::types::{ConstraintDoc, ConstraintMeta};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
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

/// Filesystem-backed `ConstraintIndex` + `ConstraintStore`.
///
/// Loads all docs from a directory once at construction (small local corpus only).
/// For large corpora use the NATS-backed implementations in `h2ai-api`.
pub struct FsConstraintIndex {
    /// tag/domain → constraint IDs
    tag_map: HashMap<String, Vec<String>>,
    /// all known IDs (for find_by_ids validation)
    all_ids: HashSet<String>,
    retriever: ConstraintRetriever,
}

impl FsConstraintIndex {
    pub fn from_docs(docs: &[ConstraintDoc]) -> Self {
        let mut tag_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut all_ids = HashSet::new();
        for doc in docs {
            all_ids.insert(doc.id.clone());
            for domain in &doc.domains {
                tag_map
                    .entry(domain.clone())
                    .or_default()
                    .push(doc.id.clone());
            }
            for tag in &doc.mandatory_for_tags {
                tag_map.entry(tag.clone()).or_default().push(doc.id.clone());
            }
        }
        Self {
            tag_map,
            all_ids,
            retriever: ConstraintRetriever::from_docs(docs),
        }
    }
}

#[async_trait]
impl ConstraintIndex for FsConstraintIndex {
    async fn find_by_ids(&self, ids: &[String]) -> Vec<String> {
        ids.iter()
            .filter(|id| self.all_ids.contains(*id))
            .cloned()
            .collect()
    }

    async fn find_by_tags(&self, tags: &[String]) -> Vec<String> {
        let mut ids: HashSet<String> = HashSet::new();
        for tag in tags {
            if let Some(tag_ids) = self.tag_map.get(tag.as_str()) {
                ids.extend(tag_ids.iter().cloned());
            }
        }
        ids.into_iter().collect()
    }

    async fn search(&self, query: &str, top_k: usize) -> Vec<String> {
        self.retriever
            .query(query, top_k)
            .into_iter()
            .map(|c| c.id)
            .collect()
    }
}

/// Filesystem-backed `ConstraintStore`.
///
/// Keeps all docs in memory — suitable only for small local corpora.
pub struct FsConstraintStore {
    docs: HashMap<String, ConstraintDoc>,
}

impl FsConstraintStore {
    pub fn from_docs(docs: Vec<ConstraintDoc>) -> Self {
        Self {
            docs: docs.into_iter().map(|d| (d.id.clone(), d)).collect(),
        }
    }

    /// Load a corpus directory and build both index and store.
    pub fn load(dir: impl AsRef<Path>) -> Result<(FsConstraintIndex, Self), std::io::Error> {
        let docs = load_corpus(dir)?;
        let index = FsConstraintIndex::from_docs(&docs);
        let store = Self::from_docs(docs);
        Ok((index, store))
    }

    /// Expose all docs (e.g. for decomposition agent's corpus parameter).
    pub fn all_docs(&self) -> Vec<&ConstraintDoc> {
        self.docs.values().collect()
    }

    /// Return all docs as a sorted vec for deterministic ordering.
    pub fn all_docs_sorted(&self) -> Vec<ConstraintDoc> {
        let mut v: Vec<ConstraintDoc> = self.docs.values().cloned().collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }
}

#[async_trait]
impl ConstraintStore for FsConstraintStore {
    async fn load(&self, id: &str) -> Result<ConstraintDoc, ConstraintError> {
        self.docs
            .get(id)
            .cloned()
            .ok_or_else(|| ConstraintError::NotFound(id.to_string()))
    }
}

/// Build a `ConstraintMeta` vec from a store (used by decomposition agent).
pub fn metas_from_store(store: &FsConstraintStore) -> Vec<ConstraintMeta> {
    store
        .all_docs_sorted()
        .into_iter()
        .map(|d| ConstraintMeta::from_doc(&d))
        .collect()
}
