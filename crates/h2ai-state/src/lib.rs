//! # h2ai-state
//!
//! CRDT semilattice merge engine and NATS JetStream event log.
//!
//! This is the **only** crate in the workspace that talks to NATS.
//!
//! ## Modules
//!
//! - [`semilattice`] — CRDT join over proposals, `ProposalSet`, `SemilatticeResult`
//! - [`bft`] — BFT consensus path (activated when `max(c_i) > 0.85`)
//! - [`journal`] — append-only `EventJournal` + `InMemoryBackend` for tests
//! - [`nats`] — `NatsClient` connection + stream bootstrap

pub mod bft;
pub mod journal;
pub mod nats;
pub mod semilattice;
