use crate::{source::ConstraintError, spec::SemanticSpec};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairProvenance {
    pub triggered_by_task: String,
    pub triggered_at_ms: u64,
    pub instability_score: f64,
    pub original_check_index: usize,
    pub original_check_text: String,
    pub simplified_check_text: String,
    pub validation_consistency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedSpec {
    pub spec: SemanticSpec,
    pub provenance: Option<RepairProvenance>,
}

#[derive(Debug, Clone)]
pub struct VersionConflictError {
    pub constraint_id: String,
    pub expected: u64,
    pub actual: u64,
}

impl std::fmt::Display for VersionConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "version conflict for {}: expected {} got {}",
            self.constraint_id, self.expected, self.actual
        )
    }
}

impl std::error::Error for VersionConflictError {}

#[async_trait]
pub trait VersionedConstraintSource: crate::source::ConstraintSource {
    async fn load_latest_versioned(&self, id: &str) -> Result<VersionedSpec, ConstraintError>;

    async fn create_next_version(
        &self,
        id: &str,
        expected_version: u64,
        spec: SemanticSpec,
        provenance: RepairProvenance,
    ) -> Result<u64, VersionConflictError>;
}
