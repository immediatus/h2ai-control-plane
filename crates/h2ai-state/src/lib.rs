//! # h2ai-state
//!
//! CRDT semilattice merge engine and NATS `JetStream` event log.
//!
//! This is the **only** crate in the workspace that talks to NATS.
//!
//! ## Modules
//!
//! - [`backend`] — all store/publisher traits: `NatsBackend` (supertrait),
//!   `StateBackend`, `EventPublisher`, `SnapshotStore`, `CalibrationStore`,
//!   `SignalPublisher`, `SignalSubscriber`, `OproStore`, `EstimatorStore`,
//!   `ReasoningStore`, `ConflictStore`, `ShadowDomainStore`, `TaskCheckpointStore`,
//!   `TaskDispatchBackend`.
//! - [`in_memory`] — `InMemoryStateBackend`: in-process `NatsBackend` impl for
//!   tests; `CapturedEvent` records events for assertion.
//! - [`semilattice`] — CRDT join over proposals, `ProposalSet`, `SemilatticeResult`
//! - [`bft`] — Condorcet/ConsensusMedian path (activated when `max(c_i) > bft_threshold`).
//!   NOTE: not Byzantine-resistant. See [`krum`] for provably BFT selection.
//! - [`krum`] — Krum and Multi-Krum Byzantine-fault-tolerant selection (n ≥ 2f+3 required).
//! - [`journal`] — append-only `EventJournal` generic over `JournalBackend`;
//!   `InMemoryBackend` implementation for test use.
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
    CalibrationStore, ConflictStore, EstimatorStore, EventPublisher, NatsBackend, OproStore,
    ReasoningStore, ShadowDomainStore, SignalPublisher, SignalSubscriber, SnapshotStore,
    StateBackend, TaskCheckpointStore, TaskDispatchBackend,
};
pub use in_memory::{CapturedEvent, InMemoryStateBackend};
pub use nats::NatsClient;
pub use nats::{
    apply_patches, generate_delta, should_store_base, tenant_bucket_name, CachedCheckpoint,
};
