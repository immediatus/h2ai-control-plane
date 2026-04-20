use h2ai_types::agent::AgentTool;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool {0:?} not registered")]
    NotRegistered(AgentTool),
    #[error("shell command failed (exit {exit_code}): {stderr}")]
    ShellFailed { exit_code: i32, stderr: String },
    #[error("tool execution timed out")]
    Timeout,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
