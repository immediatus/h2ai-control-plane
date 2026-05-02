//! Core task orchestration pipeline for the H2AI Control Plane.
//!
//! Runs the full production harness: context compilation → topology provisioning →
//! Multiplication Condition gate → parallel TAO loop → verification → auditor gate →
//! CRDT/BFT merge. Supports two execution modes for Phase 3 (Parallel Generation):
//!
//! **Direct mode** (`EngineInput.nats_dispatch = None`): each explorer slot calls a
//! locally-held `&dyn IComputeAdapter` in-process. Used for local development and tests.
//!
//! **NATS dispatch mode** (`EngineInput.nats_dispatch = Some(NatsDispatchConfig { .. })`):
//! each explorer slot gets a [`NatsDispatchAdapter`] that (1) calls
//! `AgentProvider::select_agent` to pick a live edge agent by cost tier and capability,
//! (2) publishes a `TaskPayload` to `h2ai.tasks.ephemeral.{task_id}` over core NATS,
//! and (3) awaits a `TaskResult` on the `H2AI_RESULTS` JetStream work-queue stream.
//! The `h2ai-agent` binary on the other end executes the task and publishes the result.
//!
//! # Key modules
//!
//! - [`engine`] — `ExecutionEngine::run_offline`, `EngineInput`, `EngineOutput`
//! - [`nats_dispatch_adapter`] — `NatsDispatchAdapter`, `NatsDispatchConfig`
//! - [`tao_loop`] — iterative TAO cycle with pattern/schema verification
//! - [`verification`] — LLM-as-judge parallel scoring (Phase 3.5)
//! - [`attribution`] — `HarnessAttribution` Q_total decomposition
//! - [`self_optimizer`] — MAPE-K parameter tuning suggestions
//! - [`pipeline`] — higher-level orchestration wrapper with NATS event publishing
//! - [`scheduler`] — `SchedulingEngine`, `SubtaskExecutor` trait, topo-sort wave execution
//! - [`compound`] — `CompoundTaskEngine::run` — decompose → review → schedule pipeline

pub mod attribution;
pub mod bandit;
pub mod compound;
pub mod diagnostics;
pub mod diversity;
pub mod engine;
pub mod error;
pub mod error_class;
pub mod nats_dispatch_adapter;
pub mod output_schema;
pub mod payload_store;
pub mod pipeline;
pub mod repetition;
pub mod scheduler;
pub mod self_optimizer;
pub mod session_journal;
pub mod tao_loop;
pub mod task_store;
pub mod verification;
