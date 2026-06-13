//! Tool execution framework for edge agents — shell, web-search, MCP, and WASM.
//!
//! Every tool implements the [`ToolExecutor`] trait: a `schema()` method that
//! returns a JSON Schema description for LLM prompt injection, and an async
//! `execute(input)` method that runs the tool and returns a string result or
//! [`error::ToolError`].
//!
//! The [`registry::ToolRegistry`] selects which tools are available per wave
//! mode — generation waves get the full set; audit waves get a read-only subset.
//! This prevents audit nodes from side-effecting the environment during
//! Byzantine-fault detection.
//!
//! ## Modules
//!
//! - [`shell`] — `ShellExecutor`: runs a JSON-declared command list without
//!   spawning a shell interpreter; no shell injection surface.
//! - [`web_search`] — `WebSearchExecutor`: issues search queries and returns
//!   ranked result snippets; requires an external search API key.
//! - [`mcp`] — `McpExecutor`: forwards tool calls to a Model Context Protocol
//!   server over stdio; enables arbitrary tool extension without recompilation.
//! - [`wasm`] — `WasmExecutor`: runs sandboxed WASM modules via `wasmtime`;
//!   safe execution of untrusted code within the agent.
//! - [`registry`] — `ToolRegistry::for_wave(cfg, WaveMode)` returns a configured
//!   `ToolRegistry`; `WebSearch` and MCP are only registered in `Normal` mode.
//! - [`error`] — `ToolError` enum covering all failure modes across executors.

pub mod error;
pub mod mcp;
pub mod registry;
pub mod shell;
pub mod wasm;
pub mod web_search;

use async_trait::async_trait;

/// Describes a tool's interface for injection into LLM prompts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSchema {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON Schema object describing the `input` parameter accepted by `execute()`.
    pub parameters: serde_json::Value,
}

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Returns the schema describing this tool's input contract.
    ///
    /// The `name` field must be the lowercase wire identifier the LLM will use
    /// to invoke the tool (e.g., `"shell"`, `"web_search"`). It must be stable
    /// across calls and consistent with the `AgentTool` variant's serde representation.
    fn schema(&self) -> ToolSchema;
    async fn execute(&self, input: &str) -> Result<String, crate::error::ToolError>;
}
