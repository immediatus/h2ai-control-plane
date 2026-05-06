use h2ai_config::H2AIConfig;

/// Validates tool executor configurations at node startup.
///
/// Rule: absent section → silently omit executor.
///       present section with broken config (missing env var, invalid path) → panic.
pub fn validate_tool_configs(cfg: &H2AIConfig) {
    if let Some(ws) = &cfg.web_search {
        let key = std::env::var(&ws.api_key_env).unwrap_or_default();
        if key.is_empty() {
            panic!(
                "h2ai-agent: [web_search] is configured but env var '{}' is missing or empty",
                ws.api_key_env
            );
        }
        let cx = std::env::var(&ws.cx_env).unwrap_or_default();
        if cx.is_empty() {
            panic!(
                "h2ai-agent: [web_search] is configured but env var '{}' is missing or empty",
                ws.cx_env
            );
        }
    }

    if let Some(wasm) = &cfg.wasm_executor {
        if !std::path::Path::new(&wasm.interpreter_wasm_path).exists() {
            panic!(
                "h2ai-agent: [wasm_executor] is configured but interpreter_wasm_path '{}' does not exist",
                wasm.interpreter_wasm_path
            );
        }
    }

    // mcp_filesystem: no env vars or file paths to validate at startup.
    // The subprocess command is validated when the executor is first used.
}
