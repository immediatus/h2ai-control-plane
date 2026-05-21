use crate::config::AgentRole;
use serde::{Deserialize, Serialize};

/// Retrieval mode selector for BM25 wiki provider.
/// Duplicated from `h2ai-knowledge` to keep `h2ai-types` dep-free.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalMode {
    /// Route to topic cluster first, then retrieve leaf nodes. Best for procedural depth.
    TreeTraversal,
    /// Score all RAPTOR levels simultaneously. Best for holistic orientation.
    CollapsedTree,
}

/// Per-role knowledge retrieval parameters.
/// Profiles are code constants — only change when the research reasoning changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeProfile {
    pub mode: RetrievalMode,
    pub expand_hops: u8,
    pub top_k: usize,
    /// When true: generate `topic_knowledge` by filtering result nodes on `domain_tags` overlap.
    pub domain_tag_boost: bool,
    /// Induction-injected node IDs. Empty = pure BM25 (cold start).
    pub explicit_ids: Vec<String>,
}

/// Knowledge node pattern recorded after a successful task.
/// Used by `InductionStore` to boost subsequent queries on matching `domain_tags`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNodePattern {
    pub node_id: String,
    pub role: AgentRole,
    pub domain_tags: Vec<String>,
    pub hit_rate: f32,
}

/// Returns the static knowledge profile for the given agent role.
///
/// Profile rationale (grounded in research):
/// - Coordinator: `CollapsedTree` (all RAPTOR levels simultaneously = holistic orientation), no PPR
/// - Executor: `TreeTraversal` (cluster-then-leaf = procedural depth), PPR `expand_hops=2` for
///   multi-hop traversal across constraint edges (`HippoRAG` PPR, arXiv 2405.14831)
/// - Evaluator: `TreeTraversal` (leaf-rule precision), no PPR (neighborhood adds noise here)
/// - Synthesizer: `CollapsedTree` (holistic), PPR `expand_hops=1` to surface cross-domain tensions
///   (feeds GAP-F2 `ConstraintTension` injection)
/// - Custom: inherits Executor profile
///
/// Profiles are code constants — change only when the research reasoning changes.
#[must_use]
#[allow(clippy::missing_const_for_fn)]
pub fn profile_for_role(role: &AgentRole) -> KnowledgeProfile {
    match role {
        AgentRole::Coordinator => KnowledgeProfile {
            mode: RetrievalMode::CollapsedTree,
            expand_hops: 0,
            top_k: 3,
            domain_tag_boost: false,
            explicit_ids: vec![],
        },
        AgentRole::Executor | AgentRole::Custom { .. } => KnowledgeProfile {
            mode: RetrievalMode::TreeTraversal,
            expand_hops: 2,
            top_k: 5,
            domain_tag_boost: true,
            explicit_ids: vec![],
        },
        AgentRole::Evaluator => KnowledgeProfile {
            mode: RetrievalMode::TreeTraversal,
            expand_hops: 0,
            top_k: 4,
            domain_tag_boost: true,
            explicit_ids: vec![],
        },
        AgentRole::Synthesizer => KnowledgeProfile {
            mode: RetrievalMode::CollapsedTree,
            expand_hops: 1,
            top_k: 5,
            domain_tag_boost: false,
            explicit_ids: vec![],
        },
    }
}
