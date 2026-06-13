//! Agent infrastructure abstraction for the H2AI Control Plane.
//!
//! - [`provider::AgentProvider`] — async trait for agent lifecycle management
//! - [`error::ProvisionError`] — error type for provisioning operations
//! - [`kubernetes_provider::KubernetesProvider`] — Kubernetes-backed `AgentProvider`;
//!   launches agents as pods.
//! - [`nats_provider::NatsAgentProvider`] — NATS-backed agent registry; agents
//!   self-register via `AgentRegistration` and receive tasks over NATS subjects.
//! - [`scheduling`] — `SchedulingPolicy` trait with three implementations:
//!   `LeastLoadedPolicy`, `CostAwareSpilloverPolicy`, `RoundRobinPolicy`.
//! - [`static_provider::StaticProvider`] — pre-configured agent pool with no
//!   dynamic provisioning; suitable for local development and tests.
//!
//! Agents are described by [`h2ai_types::agent::AgentDescriptor`]: a model name plus a
//! set of [`h2ai_types::agent::AgentTool`] capabilities. Providers use this descriptor
//! to launch the appropriate container/process without hard-coded agent type variants.

pub mod error;
pub mod kubernetes_provider;
pub mod nats_provider;
pub mod provider;
pub mod scheduling;
pub mod static_provider;
