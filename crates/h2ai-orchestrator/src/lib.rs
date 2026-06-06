#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::match_same_arms,
    clippy::needless_continue,
    clippy::unused_async,
    clippy::assigning_clones,
    clippy::float_cmp,
    clippy::wildcard_in_or_patterns,
    clippy::let_underscore_untyped,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::manual_let_else,
    clippy::map_unwrap_or,
    clippy::implicit_hasher,
    clippy::literal_string_with_formatting_args,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::single_match_else
)]
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
//! and (3) awaits a `TaskResult` on the `H2AI_RESULTS` `JetStream` work-queue stream.
//! The `h2ai-agent` binary on the other end executes the task and publishes the result.
//!
//! # Key modules
//!
//! - [`engine`] — `ExecutionEngine::run_offline`, `EngineInput`, `EngineOutput`
//! - [`nats_dispatch_adapter`] — `NatsDispatchAdapter`, `NatsDispatchConfig`
//! - [`tao_loop`] — iterative TAO cycle with pattern/schema verification
//! - [`verification`] — LLM-as-judge parallel scoring (Phase 3.5)
//! - [`attribution`] — `HarnessAttribution` `Q_total` decomposition
//! - [`self_optimizer`] — MAPE-K parameter tuning suggestions
//! - [`pipeline`] — higher-level orchestration wrapper with NATS event publishing
//! - [`scheduler`] — `SchedulingEngine`, `SubtaskExecutor` trait, topo-sort wave execution
//! - [`compound`] — `CompoundTaskEngine::run` — decompose → review → schedule pipeline

pub mod attribution;
pub mod bandit;
pub mod ceiling_detector;
pub mod coherence;
pub mod complexity;
pub mod compound;
pub mod context_assembler;
pub mod correlated_hallucination;
pub mod decomposition;
pub mod diagnostics;
pub mod diversity;
pub mod domain_coverage;
pub mod engine;
pub mod error;
pub mod error_class;
pub mod induction_store;
pub mod judge_panel;
pub mod leader;
pub mod mape_k;
pub mod nats_dispatch_adapter;
pub mod oracle;
pub mod oracle_gate;
pub mod output_schema;
pub mod payload_store;
pub mod phases;
pub mod pipeline;
pub mod prompts;
pub mod repetition;
pub mod scheduler;
pub mod self_optimizer;
pub mod session_journal;
pub mod skill_extractor;
pub mod signal_dispatch;
pub mod specification_grounding;
pub mod srani_gate;
pub mod srani_grounding;
pub mod synthesis;
pub mod tao_loop;
pub mod task_runner;
pub mod task_store;
pub mod thinking_loop;
pub mod verification;
