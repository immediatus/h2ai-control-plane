//! Edge-agent binary — stateless task executor dispatched over NATS JetStream.
//!
//! Each agent instance receives a single [`h2ai_types::agent::TaskPayload`] from
//! the orchestrator, runs a [`tao_agent::TaoAgent`] Thought→Action→Observation
//! loop up to `agent_max_tool_iterations` turns, and publishes a
//! [`h2ai_types::agent::TaskResult`] back over NATS.
//!
//! Agents hold no cross-task state; each instance processes exactly one
//! `TaskPayload` and terminates, so no in-process state persists between tasks.
//!
//! ## Modules
//!
//! - [`config_validation`] — fail-fast startup checks; aborts before binding if
//!   any required configuration field is missing or logically inconsistent.
//! - [`dispatch`] — NATS subscription loop; deserialises `TaskPayload` messages
//!   and drives the TAO loop to completion.
//! - [`heartbeat`] — periodic NATS keepalive so the orchestrator can distinguish
//!   a slow agent from a crashed one.
//! - [`tao_agent`] — the TAO loop: assembles tool calls, collects observations,
//!   and produces a `TaskResult`.
//! - [`tools`] — `agent_tools()`: returns the fixed tool list for this agent
//!   (`AgentTool::Shell`, `AgentTool::FileSystem`).

pub mod config_validation;
pub mod dispatch;
pub mod heartbeat;
pub mod tao_agent;
pub mod tools;
