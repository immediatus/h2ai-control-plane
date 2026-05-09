use crate::index::ConstraintIndex;
use crate::store::ConstraintStore;
use crate::types::ConstraintDoc;
use std::sync::Arc;

/// Per-task constraint resolver — combines index (find) and store (load).
///
/// Instantiated once per task, not held globally. Performs two steps:
/// 1. Find relevant constraint IDs via the index (cheap).
/// 2. Load only those docs from the store (lazy, parallel).
///
/// Resolution strategy:
/// - Explicit IDs: loaded directly, no tag or semantic expansion.
/// - Tags + query: UNION of tag matches and BM25 semantic results (broader coverage).
/// - Tags only: tag matches, no semantic fallback.
/// - Query only: BM25 semantic search.
pub struct ConstraintResolver {
    pub index: Arc<dyn ConstraintIndex>,
    pub store: Arc<dyn ConstraintStore>,
}

impl ConstraintResolver {
    pub fn new(index: Arc<dyn ConstraintIndex>, store: Arc<dyn ConstraintStore>) -> Self {
        Self { index, store }
    }

    /// Resolve and load all applicable constraints for a task.
    pub async fn resolve(
        &self,
        explicit_ids: &[String],
        tags: &[String],
        query: &str,
    ) -> Vec<ConstraintDoc> {
        use std::collections::HashSet;

        let ids: Vec<String> = if !explicit_ids.is_empty() {
            self.index.find_by_ids(explicit_ids).await
        } else if !tags.is_empty() {
            let tag_ids = self.index.find_by_tags(tags).await;
            if !query.is_empty() {
                // Union: structural (tag) + semantic (BM25) retrieval.
                let bm25_ids = self.index.search(query, 20).await;
                let mut seen: HashSet<String> = tag_ids.iter().cloned().collect();
                let mut union = tag_ids;
                for id in bm25_ids {
                    if seen.insert(id.clone()) {
                        union.push(id);
                    }
                }
                union
            } else {
                tag_ids
            }
        } else if !query.is_empty() {
            self.index.search(query, 20).await
        } else {
            vec![]
        };

        if ids.is_empty() {
            return vec![];
        }

        self.store.load_many(&ids).await
    }
}
