//! Audit and telemetry abstraction for the H2AI Control Plane.
//!
//! - [`provider::AuditProvider`] — async trait for immutable event sourcing
//! - [`error::AuditError`] — error type for audit operations
//! - [`broker_publisher::BrokerPublisherProvider`] — NATS-backed `AuditProvider`;
//!   `NatsPublishClient` trait abstracts the publish call for testing.
//! - [`direct_log::DirectLogProvider`] — `AuditProvider` that emits events via
//!   the `tracing` subscriber (stdout/log sink); no NATS dependency.
//! - [`redaction::redact_event`] — strips PII fields from `AgentTelemetryEvent`
//!   before the event reaches any persistent audit sink.

pub mod broker_publisher;
pub mod direct_log;
pub mod error;
pub mod provider;
pub mod redaction;
