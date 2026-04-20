//! Audit and telemetry abstraction for the H2AI Control Plane.
//!
//! - [`provider::AuditProvider`] — async trait for immutable event sourcing
//! - [`error::AuditError`] — error type for audit operations

pub mod broker_publisher;
pub mod direct_log;
pub mod error;
pub mod provider;
pub mod redaction;
