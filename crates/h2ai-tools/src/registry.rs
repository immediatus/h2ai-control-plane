use crate::error::ToolError;
use crate::mcp::McpExecutor;
use crate::shell::ShellExecutor;
use crate::wasm::WasmExecutor;
use crate::web_search::WebSearchExecutor;
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

    /// Production constructor. Shell always registered. WASM registered in both modes when
    /// configured. WebSearch + MCP registered only in Normal mode when configured.
    /// Uses live backends — env vars and file paths must be valid at call time.
    pub fn for_wave(cfg: &H2AIConfig, mode: WaveMode) -> Self {
        let allowlist = match mode {
            WaveMode::Normal => cfg.shell_allowlist.clone(),
            WaveMode::Hardened => cfg.shell_hardened_allowlist.clone(),
        };
        let mut r = Self::new();
        r.register_shell(ShellExecutor::new(allowlist, cfg.shell_timeout_secs));

        if let Some(wasm_cfg) = &cfg.wasm_executor {
            #[cfg(feature = "wasm")]
            {
                match crate::wasm::RealWasmBackend::from_file(
                    &wasm_cfg.interpreter_wasm_path,
                    wasm_cfg.fuel_budget,
                ) {
                    Ok(backend) => r.register_wasm(WasmExecutor::new(Box::new(backend))),
                    Err(e) => tracing::error!(error = %e, "WasmExecutor init failed"),
                }
            }
            #[cfg(not(feature = "wasm"))]
            {
                let _ = wasm_cfg;
                tracing::warn!("wasm_executor configured but 'wasm' feature is not enabled");
            }
        }

        if mode == WaveMode::Normal {
            if let Some(ws_cfg) = &cfg.web_search {
                #[cfg(feature = "web-search")]
                {
                    let api_key = std::env::var(&ws_cfg.api_key_env).unwrap_or_default();
                    let cx = std::env::var(&ws_cfg.cx_env).unwrap_or_default();
                    let backend = crate::web_search::GoogleSearchBackend::new(api_key, cx);
                    r.register_web_search(WebSearchExecutor::new(
                        Box::new(backend),
                        ws_cfg.max_results,
                    ));
                }
                #[cfg(not(feature = "web-search"))]
                {
                    let _ = ws_cfg;
                    tracing::warn!("web_search configured but 'web-search' feature is not enabled");
                }
            }

            if let Some(mcp_cfg) = &cfg.mcp_filesystem {
                let backend = crate::mcp::StdioMcpBackend::new(
                    &mcp_cfg.command,
                    mcp_cfg.args.clone(),
                    mcp_cfg.timeout_secs,
                );
                r.register_mcp(McpExecutor::new(Box::new(backend)));
            }
        }

        r
    }

    /// Test-only constructor: identical WaveMode logic but injects mock backends.
    /// Does not touch env vars, the filesystem, or spawn subprocesses.
    pub fn for_wave_with_mocks(cfg: &H2AIConfig, mode: WaveMode) -> Self {
        use crate::mcp::MockMcpBackend;
        use crate::wasm::MockWasmBackend;
        use crate::web_search::MockSearchBackend;

        let allowlist = match mode {
            WaveMode::Normal => cfg.shell_allowlist.clone(),
            WaveMode::Hardened => cfg.shell_hardened_allowlist.clone(),
        };
        let mut r = Self::new();
        r.register_shell(ShellExecutor::new(allowlist, cfg.shell_timeout_secs));

        if cfg.wasm_executor.is_some() {
            r.register_wasm(WasmExecutor::new(Box::new(MockWasmBackend::new("mock"))));
        }

        if mode == WaveMode::Normal {
            if cfg.web_search.is_some() {
                r.register_web_search(WebSearchExecutor::new(
                    Box::new(MockSearchBackend::new("mock")),
                    3,
                ));
            }
            if cfg.mcp_filesystem.is_some() {
                r.register_mcp(McpExecutor::new(Box::new(MockMcpBackend::new(
                    HashMap::new(),
                ))));
            }
        }

        r
    }

    pub fn register_shell(&mut self, executor: ShellExecutor) {
        self.executors.insert(AgentTool::Shell, Arc::new(executor));
    }

    pub fn register_web_search(&mut self, executor: WebSearchExecutor) {
        self.executors
            .insert(AgentTool::WebSearch, Arc::new(executor));
    }

    pub fn register_mcp(&mut self, executor: McpExecutor) {
        self.executors
            .insert(AgentTool::FileSystem, Arc::new(executor));
    }

    pub fn register_wasm(&mut self, executor: WasmExecutor) {
        self.executors
            .insert(AgentTool::CodeExecution, Arc::new(executor));
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
