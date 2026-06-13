//! Hierarchical knowledge retrieval — BM25+, Personalized PageRank, and skill
//! injection for the thinking loop.
//!
//! Two complementary retrieval mechanisms are layered here:
//!
//! 1. **`h2ai-knowledge` corpus retrieval** (`provider`, `bm25plus`, `graph`):
//!    RAPTOR dual-mode retrieval (TreeTraversal + CollapsedTree) over a
//!    YAML-backed knowledge source; optional Personalized PageRank expansion
//!    via `ConstraintGraph` for multi-hop evidence gathering.
//!
//! 2. **Skill injection** (`skill_provider`): `CompositeProvider` fetches
//!    skill nodes earned from prior task runs (Topic, Leaf, Constraint-keyed,
//!    Reason-keyed) from the NATS KV skill store and injects them into each
//!    thinking-loop iteration, enabling cross-task learning without fine-tuning.
//!
//! ## Modules
//!
//! - [`provider`] — `KnowledgeProvider` trait + `PassthroughProvider`;
//!   `Bm25WikiProvider` implements BM25+ with RAPTOR-mode selection.
//! - [`skill_provider`] — `CompositeProvider`: merges corpus retrieval with
//!   skill-store nodes; applies domain scoping and violation-based penalisation.
//! - [`graph`] — `ConstraintGraph` + Personalized PageRank for multi-hop
//!   knowledge expansion across constraint relationships.
//! - [`bm25plus`] — `Bm25PlusRetriever`: BM25+ index with Robertson-Walker IDF
//!   smoothing; K1=1.5 and B=0.75 are fixed constants (Lv & Zhai 2011 defaults).
//! - [`source`] — `KnowledgeSource` trait; `YamlDirSource` reads YAML files
//!   from a directory at startup.
//! - [`factory`] — `KnowledgeProviderFactory` + `KnowledgeConfig`; constructs
//!   the provider graph from configuration without coupling callers to concrete
//!   types.
//! - [`types`] — shared `KnowledgeNode`, `NodeDepth`, and `KnowledgeQuery` types.

pub mod bm25plus;
pub mod factory;
pub mod graph;
pub mod provider;
pub mod skill_provider;
pub mod source;
pub mod types;
