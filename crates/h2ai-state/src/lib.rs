//! # h2ai-state
//!
//! CRDT semilattice merge engine and NATS `JetStream` event log.
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
//! - [`weiszfeld`] — Weiszfeld geometric median BFT selection (breakdown point 50%)

pub mod backend;
pub mod bft;
pub mod in_memory;
pub mod journal;
pub mod krum;
pub mod nats;
pub mod semilattice;
pub mod weiszfeld;

pub use backend::{
    CalibrationStore, EstimatorStore, EventPublisher, OproStore, ReasoningStore, SignalPublisher,
    SnapshotStore, StateBackend,
};
pub use in_memory::{CapturedEvent, InMemoryStateBackend};
pub use nats::NatsClient;
pub use nats::{
    apply_patches, generate_delta, should_store_base, tenant_bucket_name, CachedCheckpoint,
};
