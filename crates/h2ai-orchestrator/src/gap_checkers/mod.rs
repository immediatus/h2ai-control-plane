pub mod coherence;
pub mod grounding;
pub mod selection_pruning;
pub mod task_context_seeder;

use async_trait::async_trait;
use std::sync::Arc;

/// Classification of what kind of quality problem was detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GapKind {
    /// A required provision is absent from the output.
    MissingProvision,
    /// Two provisions contradict each other.
    InterProvisionConflict,
    /// A provision is present but materially incomplete.
    IncompleteProvision,
    /// A provision covers a domain where the underlying law or knowledge is explicitly unsettled.
    /// Cannot be resolved by MicroExplorerResolver — always stays open as RequiresReview.
    UncertainDomain,
    /// Detected by GroundingChecker — an entity or claim in the output is not grounded in the
    /// task specification. Cannot be resolved by any resolver — always stays open, ensuring
    /// document_confidence < High.
    UngroundedContent,
}

/// Severity of a detected gap.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GapSeverity {
    Low,
    Medium,
    High,
}

/// Which checker produced this gap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GapSource {
    /// Derived from SelectionResolvedEvent.pruned_proposals reasons.
    SelectionPruning,
    /// Detected by CoherenceChecker inter-provision analysis.
    CoherenceCheck,
    /// Seeded from task slot configs that explicitly flag unsettled or evolving domains.
    TaskContextSeeding,
    /// Detected by GroundingChecker LLM or heuristic grounding analysis.
    GroundingCheck,
}

/// A single detected quality gap in the resolved output.
#[derive(Debug, Clone)]
pub struct Gap {
    /// Unique stable identifier within a task run (e.g., "g1", "g2").
    pub id: String,
    pub kind: GapKind,
    pub severity: GapSeverity,
    /// Human-readable description of what is missing or wrong.
    pub description: String,
    /// Which output provisions (headings/sections) this gap touches.
    pub affected_provisions: Vec<String>,
    /// IDs of other gaps that must be resolved before this one can be attempted.
    /// `None` means no dependencies (can run in the first concurrent batch).
    pub depends_on: Option<Vec<String>>,
    pub source: GapSource,
}

/// Context passed to every GapChecker implementation.
pub struct GapCheckContext {
    /// The verified provision labels from passing proposals (used by recovery to protect them).
    pub verified_provision_list: Vec<String>,
    /// The full constraint text from all active constraints for this task.
    pub constraint_text: String,
}

/// Context passed to GapResolver implementations.
pub struct GapResolveContext {
    /// The gap to close.
    pub gap: Gap,
    /// The exact resolved_output bytes (CRDT invariant: same allocation as NATS payload).
    pub resolved_output: Arc<String>,
    /// Human-readable list of already-verified provisions that must not be altered.
    pub verified_provision_list: Vec<String>,
    /// Full constraint text (all constraint IDs + criteria text) to keep resolver in bounds.
    pub constraint_text: String,
    /// Constraint IDs associated with this gap's affected provisions.
    pub constraint_ids: Vec<String>,
}

/// Outcome of a single gap resolution attempt.
pub struct ResolutionResult {
    pub gap_id: String,
    /// The patched text that replaces the affected section in resolved_output.
    /// `None` when recovery produced no improvement above `recovery_tau`.
    pub patched_text: Option<String>,
    /// Improvement delta (new_score - old_score). Negative means regression.
    pub score_delta: f64,
}

/// Overall outcome after all resolution passes.
pub struct RecoveryOutcome {
    /// Final resolved text after all accepted patches.
    pub resolved_output: String,
    /// Which gaps were successfully closed (score_delta >= recovery_tau).
    pub closed_gaps: Vec<String>,
    /// Which gaps remain open after all passes.
    pub open_gaps: Vec<Gap>,
}

/// Analyses the resolved output for quality gaps.
///
/// # CRDT Invariant
///
/// The `document` bytes passed here MUST be the exact same allocation that will be
/// committed to NATS JetStream. Implementations must NOT clone and mutate — operate
/// on the bytes byte-for-byte as received.
#[async_trait]
pub trait GapChecker: Send + Sync {
    async fn check(&self, document: &str, context: &GapCheckContext) -> Vec<Gap>;
}

/// Attempts to close a single gap by generating a patched section.
#[async_trait]
pub trait GapResolver: Send + Sync {
    /// Returns true when this resolver handles the given `GapKind`.
    fn handles(&self, kind: &GapKind) -> bool;
    async fn resolve(&self, context: GapResolveContext) -> ResolutionResult;
}
