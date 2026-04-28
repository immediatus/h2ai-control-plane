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
//! - [`physics`] — `CoherencyCoefficients`, `RoleErrorCost`, `MergeStrategy`,
//!   `CoordinationThreshold`, `MultiplicationCondition`, `JeffectiveGap`
//! - [`adapter`] — `IComputeAdapter` trait, `ComputeRequest`, `ComputeResponse`, `AdapterError`
//! - [`events`] — all orchestration event structs and the `H2AIEvent` enum
//! - [`agent`] — `AgentState`, `TaskPayload`, `TaskResult`, `AgentTelemetryEvent`
//! - [`plan`] — `Subtask`, `SubtaskPlan`, `SubtaskResult`, `PlanStatus`

pub mod adapter;
pub mod agent;
pub mod config;
pub mod events;
pub mod identity;
pub mod manifest;
pub mod memory;
pub mod physics;
pub mod plan;
