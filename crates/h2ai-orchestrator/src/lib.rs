//! Core task orchestration pipeline for the H2AI Control Plane.
//!
//! Ties together memory, provisioning, and telemetry into a single task execution pipeline.
//!
//! # Pipeline
//!
//! 1. [`MemoryProvider::get_recent_history`] — assemble context for the task
//! 2. Construct [`TaskPayload`] from instructions + assembled context
//! 3. [`AgentProvider::ensure_agent_capacity`] — ready infrastructure
//! 4. Publish `TaskPayload` to NATS subject
//! 5. Listen asynchronously on telemetry and response NATS subjects
//! 6. Stream incoming logs to [`AuditProvider::record_event`]
//! 7. On [`TaskResult`], call [`MemoryProvider::commit_new_memories`]

pub mod attribution;
pub mod engine;
pub mod error;
pub mod error_class;
pub mod output_schema;
pub mod pipeline;
pub mod self_optimizer;
pub mod tao_loop;
pub mod task_store;
pub mod verification;
