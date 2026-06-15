//! Constraint type system, corpus loader, evaluator, and versioned store.
//!
//! A [`types::ConstraintDoc`] is the unit of contractual knowledge in H2AI.
//! Each document declares Hard, Soft, or Advisory severity and one or more
//! [`types::ConstraintPredicate`] checks — `VocabularyPresence`, `NegativeKeyword`,
//! `RegexMatch`, `LlmJudge`, `SemanticPresence`, `SemanticOrdering`,
//! `SemanticExclusion`, `Composite` (AND/OR) — that the verifier evaluates
//! against every proposal.
//!
//! `compliance = hard_gate × soft_score`: any Hard predicate that scores 0.0
//! sets compliance to zero regardless of soft scores, implementing a typed veto.
//!
//! ## Modules
//!
//! - [`types`] — core `ConstraintDoc`, `ConstraintPredicate`, `ConstraintSeverity`
//!   types and the `compliance` formula.
//! - [`loader`] / [`yaml`] — load YAML files from a corpus directory into typed
//!   `ConstraintDoc` values; validates required fields at load time.
//! - [`eval`] — `eval_sync` evaluates static predicates against a proposal
//!   string; returns a `f64` score in [0.0, 1.0]. LLM-backed predicates
//!   (`LlmJudge`, Semantic*) pass through as 1.0 and must be evaluated via the
//!   async verifier path, which returns a `ComplianceResult`.
//! - [`resolver`] — resolves constraint IDs from a task manifest against the
//!   corpus; returns the subset applicable to a specific task.
//! - [`ambiguity`] — static scanner and scorecard for GAP-F8: detects
//!   structurally ambiguous predicates before the LLM ever sees them.
//! - [`wiki`] / [`store`] / [`versioned`] / [`nats_versioned`] — constraint
//!   corpus versioning; hot-reload and NATS-backed revision tracking.
//! - [`source`] — `ConstraintSource` trait; `RuntimeConstraintStore` /
//!   `RuntimeConstraintIndex` (aliased as `FsConstraintStore` / `FsConstraintIndex`)
//!   for filesystem-backed constraint loading at startup.
//! - [`spec`] — `SemanticSpec`: structured constraint definition (exclusions,
//!   requirements, orderings, rubric) that converts to `ConstraintDoc`.
//! - [`index`] / [`retrieval`] — BM25-based constraint retrieval for knowledge
//!   injection during the thinking loop.
//! - [`conflict`] / [`complexity`] — conflict-rate accumulation (CRDT) and
//!   constraint complexity scoring used by the MAPE-K controller.

pub mod ambiguity;
pub mod clustering;
pub mod complexity;
pub mod conflict;
pub mod eval;
pub mod index;
pub mod loader;
pub mod nats_versioned;
pub mod resolver;
pub mod retrieval;
pub mod source;
pub mod spec;
pub mod store;
pub mod types;
pub mod versioned;
pub mod wiki;
pub mod yaml;
