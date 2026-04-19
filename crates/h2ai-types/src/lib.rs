//! # h2ai-types
//!
//! Pure types boundary for the H2AI Control Plane.
//!
//! This crate contains **zero external I/O dependencies** — no NATS, no HTTP, no FFI.
//! All other crates in the workspace depend on this crate.
//!
//! ## Modules
//!
//! - [`identity`] — `TaskId`, `ExplorerId` newtype identifiers
//! - [`config`] — `ParetoWeights`, `TopologyKind`, `AdapterKind`, `AgentRole`, `RoleSpec`,
//!   `ReviewGate`, `ExplorerConfig`, `AuditorConfig`
//! - [`physics`] — `CoherencyCoefficients`, `RoleErrorCost`, `MergeStrategy`,
//!   `CoordinationThreshold`, `MultiplicationCondition`, `JeffectiveGap`
//! - [`adapter`] — `IComputeAdapter` trait, `ComputeRequest`, `ComputeResponse`, `AdapterError`
//! - [`events`] — all 17 event structs and the `H2AIEvent` enum

pub mod adapter;
pub mod config;
pub mod events;
pub mod identity;
pub mod physics;
