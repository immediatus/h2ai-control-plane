//! # h2ai-types
//!
//! Pure types boundary for the H2AI Control Plane.
//!
//! This crate contains **zero external I/O dependencies** — no NATS, no HTTP, no FFI.
//! All other crates in the workspace depend on this crate.
//!
//! ## Modules
//!
//! - [`identity`] — `AgentId`, `TaskId`, `ExplorerId`, `SubtaskId` newtype identifiers
//! - [`config`] — `ParetoWeights`, `TopologyKind`, `AdapterKind`, `AgentRole`, `RoleSpec`,
//!   `ReviewGate`, `ExplorerConfig`, `AuditorConfig`
//! - [`memory`] — `MemoryTier` enum (Working/Episodic/Semantic/Procedural) with ρ and halflife
//! - [`sizing`] — `CoherencyCoefficients`, `RoleErrorCost`, `MergeStrategy`,
//!   `CoordinationThreshold`, `MultiplicationCondition`
//! - [`adapter`] — `IComputeAdapter` trait, `ComputeRequest`, `ComputeResponse`, `AdapterError`
//! - [`events`] — all orchestration event structs and the `H2AIEvent` enum
//! - [`agent`] — `AgentState`, `TaskPayload`, `TaskResult`, `AgentTelemetryEvent`
//! - [`plan`] — `Subtask`, `SubtaskPlan`, `SubtaskResult`, `PlanStatus`

pub mod adapter;
pub mod agent;
pub mod approval;
pub mod checkpoint;
pub mod checkpoint_delta;
pub use checkpoint_delta::{CheckpointKind, TaskCheckpointEntry};
pub mod config;
pub mod events;
pub mod identity;
pub mod manifest;
pub mod memory;
pub mod reasoning_checkpoint;
pub use reasoning_checkpoint::{
    AdapterWaveOutput, ArchetypeResult, ArchetypeSelection, CompletedWave,
    ReasoningCheckpointPhase, TaskMetaState, TaskReasoningCheckpoint,
};
pub mod plan;
pub mod prompt_variant;
pub mod prompts;
pub use prompt_variant::{AdapterOproState, PromptBanditArm, PromptVariant, PromptVariantSource};
pub mod conflict;
pub mod judge;
pub use judge::{JudgePersona, PanelDiversityKind};
pub mod sizing;
pub use conflict::{ConflictRateAccumulator, ConflictRateSample};
pub mod knowledge;
pub mod thinking;
pub use knowledge::{profile_for_role, KnowledgeNodePattern, KnowledgeProfile, RetrievalMode};
pub mod calibration;
pub mod signal;
