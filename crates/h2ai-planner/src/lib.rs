//! LLM-driven task decomposition and plan review.
//!
//! When the complexity probe rates a task at or above `decompose_threshold`,
//! the orchestrator calls [`PlanningEngine::decompose`] to break the task into
//! sub-tasks that can each be handled by a single explorer wave. The resulting
//! plan is immediately reviewed by [`PlanReviewer::evaluate`] for structural
//! soundness before any sub-task is dispatched.
//!
//! ## Structural checks performed by the reviewer
//!
//! - Empty plan (zero subtasks) is rejected immediately without an LLM call.
//! - Cycles in the dependency graph are detected locally by `detect_cycle`.
//! - If both pass, one LLM call performs the semantic soundness review.
//!
//! If the reviewer rejects the plan, the orchestrator retries decomposition
//! a configurable number of times before escalating to HITL.
//!
//! ## Modules
//!
//! - [`decomposer`] — `PlanningEngine::decompose`; calls the LLM, parses the
//!   structured plan JSON, and returns a `Vec<SubTask>`.
//! - [`reviewer`] — `PlanReviewer::evaluate`; applies structural checks and
//!   returns a [`ReviewOutcome`] with a pass/fail verdict and diagnostics.
//! - [`parsing`] — shared JSON extraction helpers used by both modules.

pub mod decomposer;
pub mod parsing;
pub mod reviewer;

pub use decomposer::{PlannerError, PlanningEngine};
pub use reviewer::{PlanReviewer, ReviewOutcome};
