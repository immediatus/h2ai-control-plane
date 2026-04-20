//! # h2ai-state
//!
//! CRDT semilattice merge engine and NATS JetStream event log.
//!
//! This is the **only** crate in the workspace that talks to NATS.
//!
//! ## Modules
//!
//! - [`semilattice`] — CRDT join over proposals, `ProposalSet`, `SemilatticeResult`
//! - [`bft`] — Condorcet/ConsensusMedian path (activated when `max(c_i) > bft_threshold`).
//!   NOTE: not Byzantine-resistant. See [`krum`] for provably BFT selection.
//! - [`krum`] — Krum and Multi-Krum Byzantine-fault-tolerant selection (n ≥ 2f+3 required).
//! - [`journal`] — append-only `EventJournal` + `InMemoryBackend` for tests
//! - [`nats`] — `NatsClient` connection + stream bootstrap

pub mod bft;
pub mod journal;
pub mod krum;
pub mod nats;
pub mod semilattice;

pub use nats::NatsClient;
