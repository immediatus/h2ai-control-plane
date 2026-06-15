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
//! - [`manifest`] — `TaskManifest`, `TopologyRequest`, `ExplorerRequest`,
//!   `ExplorerSlotConfig`, `CotStyle`; the primary task description submitted by callers.
//! - [`memory`] — `MemoryTier` enum (Working/Episodic/Semantic/Procedural) with ρ and halflife
//! - [`sizing`] — `CoherencyCoefficients`, `RoleErrorCost`, `MergeStrategy`,
//!   `CoordinationThreshold`, `MultiplicationCondition`
//! - [`adapter`] — `IComputeAdapter` trait, `ComputeRequest`, `ComputeResponse`, `AdapterError`
//! - [`events`] — all orchestration event structs and the `H2AIEvent` enum
//! - [`agent`] — `AgentState`, `TaskPayload`, `TaskResult`, `AgentTelemetryEvent`
//! - [`plan`] — `Subtask`, `SubtaskPlan`, `SubtaskResult`, `PlanStatus`
//! - [`approval`] — `ApprovalRecord`, `ApprovalDecision`, `HumanOracleRequest`,
//!   `HumanOracleRating`; HITL approval flow types.
//! - [`checkpoint`] — `ConstraintSnapshot`, `TaskCheckpoint`; per-task state
//!   snapshots persisted to NATS KV for crash recovery.
//! - [`calibration`] — `CalibrationRecord`, `AuditorHealth`, `CalibrationDriftWarning`,
//!   `CalibrationChangepoint`, `AuditorCircuitState`, `ProbeSource`.
//! - [`signal`] — `ResumeSignal`, `ApproveSignal`, `WaveContinueSignal`,
//!   `SignalPayload`; inter-component control messages.
//! - [`thinking`] — `ThinkingReport`, `ArchetypeSpec`, `ArchetypeOutput`, `ModelTier`.
//! - [`gap_i1`] — `KnowledgeGapRecord`, `DomainSynthesis`; knowledge gap evidence types.
//! - [`knowledge`] — `KnowledgeProfile`, `KnowledgeNodePattern`, `RetrievalMode`,
//!   `profile_for_role`; retrieval configuration per agent role.
//! - [`prompts`] — default system prompt builders (e.g. `auditor_system_prompt_default`).
//! - [`conflict`] — `ConflictRateAccumulator`, `ConflictRateSample` (also re-exported
//!   at crate root).
//! - [`judge`] — `JudgePersona`, `PanelDiversityKind` (also re-exported at crate root).
//! - [`prompt_variant`] — `PromptVariant`, `PromptBanditArm`, `PromptVariantSource`,
//!   `AdapterOproState` (also re-exported at crate root).
//! - [`reasoning_checkpoint`] — `TaskReasoningCheckpoint`, `CompletedWave`,
//!   `AdapterWaveOutput`, `ArchetypeResult`, `ArchetypeSelection`,
//!   `ReasoningCheckpointPhase`, `TaskMetaState` (also re-exported at crate root).

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
pub mod gap_i1;
pub mod judge;
pub use judge::{JudgePersona, PanelDiversityKind};
pub mod sizing;
pub use conflict::{ConflictRateAccumulator, ConflictRateSample};
pub mod knowledge;
pub mod thinking;
pub use knowledge::{profile_for_role, KnowledgeNodePattern, KnowledgeProfile, RetrievalMode};
pub mod calibration;
pub mod chain;
pub mod signal;
