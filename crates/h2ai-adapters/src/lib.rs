//! Concrete [`h2ai_types::adapter::IComputeAdapter`] implementations.
//!
//! - [`mock::MockAdapter`] — deterministic test double, zero I/O
//! - [`cloud::CloudGenericAdapter`] — OpenAI-compatible HTTP endpoint

pub mod cloud;
pub mod mock;
