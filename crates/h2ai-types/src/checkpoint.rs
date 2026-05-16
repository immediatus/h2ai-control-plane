use serde::{Deserialize, Serialize};

/// Audit record of which constraints were resolved and evaluated for a specific task.
///
/// Stored inside `TaskCheckpoint` for regulatory audit: "which constraints were active
/// at task creation time and which versions were applied?"
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintSnapshot {
    /// NATS KV revision of the wiki index at task creation time.
    pub wiki_revision: u64,
    /// Constraint IDs resolved by wiki lookup (tag + explicit IDs).
    pub resolved_ids: Vec<String>,
    /// Constraint IDs that were actually evaluated against at least one proposal.
    pub evaluated_ids: Vec<String>,
    /// Constraint IDs that fired (failed Hard or Soft threshold) on any proposal.
    pub violation_ids: Vec<String>,
}

/// Phase-output checkpoint for in-flight task crash recovery.
///
/// Written to NATS KV bucket `H2AI_TASK_CHECKPOINTS` (zstd-compressed JSON).
/// Phase is stored as a string name ("ParallelGeneration") for version stability —
/// enum discriminants shift when new variants are inserted; string names do not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskCheckpoint {
    pub task_id: String,
    /// Phase name from `TaskPhase::name_str()` — stable across binary upgrades.
    pub phase: String,
    /// Owning node identity (hostname + PID) for distributed lease.
    pub node_id: String,
    /// NATS KV revision at last write — used for optimistic concurrency.
    pub lease_seq: u64,
    /// Raw proposal strings saved after `ParallelGeneration` completes.
    pub proposals: Vec<String>,
    /// Survivor indices saved after `AuditorGate` completes.
    pub auditor_survivors: Vec<usize>,
    /// Final merged output saved after `Merging` completes.
    pub resolved_output: Option<String>,
    /// Full `TaskManifest` as JSON string for engine re-entry on recovery.
    pub manifest_json: String,
    /// When the payload exceeded 800 KB and was offloaded to Object Store,
    /// this holds the object name. `delete_task_checkpoint()` MUST delete
    /// this object before deleting the KV entry.
    pub object_store_ref: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    /// Audit snapshot of constraint resolution for this task.
    /// None for tasks submitted before wiki integration was deployed.
    #[serde(default)]
    pub constraint_snapshot: Option<ConstraintSnapshot>,
    /// Jury Efficiency computed at merge time; persisted for the HITL approval path.
    /// `None` when n_agents = 0 (Condorcet undefined) or for checkpoints predating this field.
    #[serde(default)]
    pub j_eff: Option<f64>,
}
