use crate::source::ConstraintError;
use crate::types::ConstraintDoc;
use async_trait::async_trait;

/// Content store — loads full `ConstraintDoc` by ID, on demand.
///
/// Never bulk-loads. Implementations may cache at the call-site level
/// but must not hold the full corpus in memory.
#[async_trait]
pub trait ConstraintStore: Send + Sync {
    async fn load(&self, id: &str) -> Result<ConstraintDoc, ConstraintError>;

    /// Fetch multiple docs in parallel; missing IDs are silently skipped.
    async fn load_many(&self, ids: &[String]) -> Vec<ConstraintDoc> {
        let futs: Vec<_> = ids.iter().map(|id| self.load(id)).collect();
        futures::future::join_all(futs)
            .await
            .into_iter()
            .filter_map(|r| {
                r.map_err(|e| tracing::warn!("constraint load failed: {e}"))
                    .ok()
            })
            .collect()
    }
}
