pub mod error;
pub mod registry;
pub mod shell;

use async_trait::async_trait;

/// Describes a tool's interface for injection into LLM prompts.
#[derive(Debug, Clone, PartialEq)]
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
