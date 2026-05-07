use crate::events::{ApprovalRiskLevel, ApprovalTrigger};
use serde::{Deserialize, Serialize};

/// Snapshot of a task output held pending human review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRecord {
    pub task_id: String,
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
