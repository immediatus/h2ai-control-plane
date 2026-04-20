pub mod error;
pub mod registry;
pub mod shell;

use async_trait::async_trait;

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, input: &str) -> Result<String, crate::error::ToolError>;
}
