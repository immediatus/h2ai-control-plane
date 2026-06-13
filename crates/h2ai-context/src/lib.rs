//! Dark Knowledge Compiler — assembles the immutable `system_context` injected
//! into every explorer prompt before generation starts.
//!
//! The compiler fuses three sources into a single deterministic string:
//! the constraint corpus (typed predicates → prose rules), the task manifest
//! (description, Pareto weights, explorer role), and any retrieved knowledge
//! nodes from `h2ai-knowledge`. Identical inputs always produce identical
//! context strings, making generation reproducible given the same corpus revision.
//!
//! ## Modules
//!
//! - [`compiler`] — top-level `compile(manifest, corpus, include_rubric)` entry
//!   point; renders each constraint as a structured prose block and returns
//!   the final `system_context` string.
//! - [`context_chunk`] — `ContextChunk`: memory-tier-tagged content slice with
//!   Ebbinghaus decay weight; determines ensemble size requirements per chunk.
//! - [`fusion`] — Reciprocal Rank Fusion (`rrf_fuse`) and BM25 search via
//!   Tantivy (`bm25_search`); fuses multiple ranked candidate lists into one.
//! - [`compaction`] — `compact(context, CompactionConfig)`: truncates the
//!   context to fit within `max_tokens` by keeping a head+tail window and
//!   dropping the middle, then re-injects any preserved keywords that were lost.
//! - [`embedding`] — thin wrapper around the ONNX embedding model used by
//!   `h2ai-state`'s Krum and Weiszfeld selectors; produces L2-normalised f32
//!   vectors.

pub mod compaction;
pub mod compiler;
pub mod context_chunk;
pub mod embedding;
pub mod fusion;
