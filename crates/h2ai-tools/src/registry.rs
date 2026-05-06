use crate::error::ToolError;
use crate::shell::ShellExecutor;
use crate::ToolExecutor;
use h2ai_config::H2AIConfig;
use h2ai_types::agent::{AgentTool, WaveMode};
use std::collections::HashMap;
use std::sync::Arc;

type ExecutorFn = Arc<dyn ToolExecutor>;

pub struct ToolRegistry {
    executors: HashMap<AgentTool, ExecutorFn>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            executors: HashMap::new(),
        }
    }

    pub fn default_with_shell() -> Self {
        let mut r = Self::new();
        r.register_shell(ShellExecutor::default());
        r
    }

    /// Constructs a fresh registry with a ShellExecutor whose allowlist is
    /// selected by wave epistemic state:
    /// - WaveMode::Normal   → cfg.shell_allowlist
    /// - WaveMode::Hardened → cfg.shell_hardened_allowlist
    pub fn for_wave(cfg: &H2AIConfig, mode: WaveMode) -> Self {
        let allowlist = match mode {
            WaveMode::Normal => cfg.shell_allowlist.clone(),
            WaveMode::Hardened => cfg.shell_hardened_allowlist.clone(),
        };
        let mut r = Self::new();
        r.register_shell(ShellExecutor::new(allowlist, cfg.shell_timeout_secs));
        r
    }

    pub fn register_shell(&mut self, executor: ShellExecutor) {
        self.executors.insert(AgentTool::Shell, Arc::new(executor));
    }

    pub fn register(&mut self, tool: AgentTool, executor: Arc<dyn ToolExecutor>) {
        self.executors.insert(tool, executor);
    }

    pub async fn execute(&self, tool: AgentTool, input: &str) -> Result<String, ToolError> {
        match self.executors.get(&tool) {
            Some(exec) => exec.execute(input).await,
            None => Err(ToolError::NotRegistered(tool)),
        }
    }

    /// Returns the schema for every registered tool, in arbitrary order.
    pub fn all_schemas(&self) -> Vec<crate::ToolSchema> {
        self.executors.values().map(|e| e.schema()).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
