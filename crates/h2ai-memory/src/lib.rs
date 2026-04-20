//! Memory and context state abstraction for the H2AI Control Plane.
//!
//! - [`provider::MemoryProvider`] — async trait for agent conversation history
//! - [`error::MemoryError`] — error type for memory operations

pub mod error;
pub mod in_memory;
pub mod nats_kv;
pub mod provider;
