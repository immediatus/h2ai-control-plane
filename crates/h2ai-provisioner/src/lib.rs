//! Agent infrastructure abstraction for the H2AI Control Plane.
//!
//! - [`provider::AgentProvider`] — async trait for agent lifecycle management
//! - [`error::ProvisionError`] — error type for provisioning operations
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
