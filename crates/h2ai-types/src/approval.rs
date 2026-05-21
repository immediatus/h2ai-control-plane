use crate::events::{ApprovalRiskLevel, ApprovalTrigger};
use crate::identity::TenantId;
use serde::{Deserialize, Serialize};

/// Snapshot of a task output held pending human review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRecord {
    pub task_id: String,
    pub tenant_id: TenantId,
    pub proposed_output: String,
    pub q_confidence: f64,
    pub triggered_by: ApprovalTrigger,
    pub created_at_ms: u64,
    pub timeout_at_ms: u64,
}

/// Operator decision on a pending approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalDecision {
    pub approved: bool,
    pub reviewer_note: Option<String>,
    /// Required — from auth header or config. "system:timeout" for auto-rejects.
    pub operator_id: String,
    pub decided_at_ms: u64,
}

/// Derive risk level from confidence and trigger type.
///
/// `Low` is never assigned — tasks reaching the gate always warrant review.
#[must_use]
pub fn compute_risk_level(triggered_by: &ApprovalTrigger, q_confidence: f64) -> ApprovalRiskLevel {
    if q_confidence < 0.3 {
        ApprovalRiskLevel::High
    } else {
        match triggered_by {
            ApprovalTrigger::ManifestFlag | ApprovalTrigger::LowConfidence => {
                ApprovalRiskLevel::Medium
            }
        }
    }
}

/// Human rating request sent to the reviewer API.
///
/// MUST NOT contain `q_confidence` or `task_id` — stored server-side to preserve
/// FUSE Triplet Conditional Independence. Showing confidence to raters causes anchoring bias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanOracleRequest {
    pub oracle_id: String,
    pub tenant_id: TenantId,
    pub task_description: String,
    pub winning_output: String,
    pub rubric: String,
    pub created_at_ms: u64,
    pub expires_at_ms: u64,
}

/// Rater submission for a `HumanOracleRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanOracleRating {
    pub oracle_id: String,
    pub passed: bool,
    pub likert_score: Option<u8>, // 1-5; caller ensures >=3 → passed=true
    pub rater_note: Option<String>,
    pub rater_id: String, // SHA-256 hashed before storage
    pub rated_at_ms: u64,
}
