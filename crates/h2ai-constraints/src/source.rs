use crate::index::ConstraintIndex;
use crate::retrieval::ConstraintRetriever;
use crate::spec::SemanticSpec;
use crate::store::ConstraintStore;
use crate::types::{ConstraintDoc, ConstraintMeta};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConstraintError {
    #[error("constraint not found: {0}")]
    NotFound(String),
    #[error("store unavailable: {0}")]
    Unavailable(String),
    #[error("deserialize error: {0}")]
    Deserialize(String),
    #[error("validation error: {0}")]
    Validation(String),
}

/// Storage-agnostic source of SemanticSpec objects.
/// Implementations: YamlDirSource (filesystem) and InMemorySource (tests/code).
pub trait ConstraintSource: Send + Sync {
    fn load_all(&self) -> Result<Vec<SemanticSpec>, ConstraintError>;
}

/// In-memory source — holds SemanticSpec directly. Use in tests and programmatic construction.
pub struct InMemorySource {
    pub specs: Vec<SemanticSpec>,
}

impl ConstraintSource for InMemorySource {
    fn load_all(&self) -> Result<Vec<SemanticSpec>, ConstraintError> {
        Ok(self.specs.clone())
    }
}

/// In-memory ConstraintIndex + ConstraintStore built from any ConstraintSource.
///
/// "Runtime" prefix reflects that these types hold compiled ConstraintDoc in memory
/// regardless of how the source loaded them. The "Fs" prefix was an artifact of the
/// original filesystem-only load path.
pub struct RuntimeConstraintIndex {
    tag_map: HashMap<String, Vec<String>>,
    all_ids: HashSet<String>,
    retriever: ConstraintRetriever,
}

impl RuntimeConstraintIndex {
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
impl ConstraintIndex for RuntimeConstraintIndex {
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

pub struct RuntimeConstraintStore {
    docs: HashMap<String, ConstraintDoc>,
}

impl RuntimeConstraintStore {
    pub fn from_docs(docs: Vec<ConstraintDoc>) -> Self {
        Self {
            docs: docs.into_iter().map(|d| (d.id.clone(), d)).collect(),
        }
    }

    /// Load from any ConstraintSource — filesystem, in-memory, or future NATS KV.
    pub fn from_source(source: &dyn ConstraintSource) -> Result<Self, ConstraintError> {
        let specs = source.load_all()?;
        let docs: Vec<ConstraintDoc> = specs.into_iter().map(|s| s.into_constraint_doc()).collect();
        Ok(Self::from_docs(docs))
    }

    /// Load a corpus directory — convenience wrapper over load_corpus().
    pub fn load(
        dir: impl AsRef<std::path::Path>,
    ) -> Result<(RuntimeConstraintIndex, Self), std::io::Error> {
        use crate::loader::load_corpus;
        let docs = load_corpus(dir)?;
        let index = RuntimeConstraintIndex::from_docs(&docs);
        let store = Self::from_docs(docs);
        Ok((index, store))
    }

    pub fn all_docs(&self) -> Vec<&ConstraintDoc> {
        self.docs.values().collect()
    }

    pub fn all_docs_sorted(&self) -> Vec<ConstraintDoc> {
        let mut v: Vec<ConstraintDoc> = self.docs.values().cloned().collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }
}

#[async_trait]
impl ConstraintStore for RuntimeConstraintStore {
    async fn load(&self, id: &str) -> Result<ConstraintDoc, ConstraintError> {
        self.docs
            .get(id)
            .cloned()
            .ok_or_else(|| ConstraintError::NotFound(id.to_string()))
    }
}

/// Build a ConstraintMeta vec from a store (used by decomposition agent).
pub fn metas_from_store(store: &RuntimeConstraintStore) -> Vec<ConstraintMeta> {
    store
        .all_docs_sorted()
        .into_iter()
        .map(|d| ConstraintMeta::from_doc(&d))
        .collect()
}

// Backward-compat type aliases — kept for one release cycle.
// Remove once all callers have migrated to Runtime* names.
pub type FsConstraintIndex = RuntimeConstraintIndex;
pub type FsConstraintStore = RuntimeConstraintStore;
