use async_trait::async_trait;

/// Thin discovery layer — resolves constraint IDs from lightweight metadata only.
///
/// Never loads full predicates or payloads. O(1) for explicit IDs,
/// O(index) for tag lookup, O(corpus) for semantic search.
#[async_trait]
pub trait ConstraintIndex: Send + Sync {
    /// Validate and return the subset of `ids` that exist in the index.
    async fn find_by_ids(&self, ids: &[String]) -> Vec<String>;

    /// Return constraint IDs whose domain tags intersect with `tags`.
    async fn find_by_tags(&self, tags: &[String]) -> Vec<String>;

    /// BM25 / semantic search — return up to `top_k` IDs ranked by relevance.
    async fn search(&self, query: &str, top_k: usize) -> Vec<String>;
}
