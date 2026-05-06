use h2ai_config::{McpFilesystemConfig, WasmExecutorConfig, WebSearchConfig};
use h2ai_tools::registry::ToolRegistry;
use h2ai_types::agent::AgentTool;

#[tokio::test]
async fn shell_executor_runs_echo_command() {
    let registry = ToolRegistry::default_with_shell();
    let result = registry
        .execute(
            AgentTool::Shell,
            r#"{"command": "echo", "args": ["hello_tool"]}"#,
        )
        .await;
    assert!(result.is_ok(), "{:?}", result);
    assert!(result.unwrap().contains("hello_tool"));
}

#[tokio::test]
async fn shell_executor_returns_error_on_nonzero_exit() {
    let registry = ToolRegistry::default_with_shell();
    // `false` exits with code 1 — no shell interpreter needed
    let result = registry
        .execute(AgentTool::Shell, r#"{"command": "false"}"#)
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn registry_returns_err_for_unregistered_tool() {
    let registry = ToolRegistry::new();
    let result = registry.execute(AgentTool::WebSearch, "query").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn for_wave_normal_registers_all_configured_executors() {
    let cfg = h2ai_config::H2AIConfig {
        web_search: Some(WebSearchConfig {
            api_key_env: "DUMMY_KEY".into(),
            cx_env: "DUMMY_CX".into(),
            max_results: 3,
        }),
        mcp_filesystem: Some(McpFilesystemConfig {
            command: "echo".into(),
            args: vec![],
            timeout_secs: 5,
        }),
        wasm_executor: Some(WasmExecutorConfig {
            interpreter_wasm_path: "/nonexistent.wasm".into(),
            fuel_budget: 100_000,
        }),
        ..h2ai_config::H2AIConfig::default()
    };
    let registry = ToolRegistry::for_wave_with_mocks(&cfg, h2ai_types::agent::WaveMode::Normal);
    let schemas: Vec<_> = registry.all_schemas().iter().map(|s| s.name).collect();
    assert!(schemas.contains(&"shell"), "missing shell");
    assert!(schemas.contains(&"web_search"), "missing web_search");
    assert!(schemas.contains(&"file_system"), "missing file_system");
    assert!(
        schemas.contains(&"code_execution"),
        "missing code_execution"
    );
}

#[tokio::test]
async fn for_wave_hardened_only_registers_shell_and_wasm() {
    let cfg = h2ai_config::H2AIConfig {
        web_search: Some(WebSearchConfig {
            api_key_env: "DUMMY_KEY".into(),
            cx_env: "DUMMY_CX".into(),
            max_results: 3,
        }),
        mcp_filesystem: Some(McpFilesystemConfig {
            command: "echo".into(),
            args: vec![],
            timeout_secs: 5,
        }),
        wasm_executor: Some(WasmExecutorConfig {
            interpreter_wasm_path: "/nonexistent.wasm".into(),
            fuel_budget: 100_000,
        }),
        ..h2ai_config::H2AIConfig::default()
    };
    let registry = ToolRegistry::for_wave_with_mocks(&cfg, h2ai_types::agent::WaveMode::Hardened);
    let schemas: Vec<_> = registry.all_schemas().iter().map(|s| s.name).collect();
    assert!(schemas.contains(&"shell"), "missing shell");
    assert!(
        schemas.contains(&"code_execution"),
        "missing code_execution"
    );
    assert!(
        !schemas.contains(&"web_search"),
        "web_search must be absent in Hardened mode"
    );
    assert!(
        !schemas.contains(&"file_system"),
        "file_system must be absent in Hardened mode"
    );
}

#[tokio::test]
async fn for_wave_absent_config_omits_executor_silently() {
    let cfg = h2ai_config::H2AIConfig::default();
    let registry = ToolRegistry::for_wave_with_mocks(&cfg, h2ai_types::agent::WaveMode::Normal);
    let schemas: Vec<_> = registry.all_schemas().iter().map(|s| s.name).collect();
    assert!(schemas.contains(&"shell"), "shell must always be present");
    assert!(
        !schemas.contains(&"web_search"),
        "web_search must be absent without config"
    );
    assert!(
        !schemas.contains(&"file_system"),
        "file_system must be absent without config"
    );
    assert!(
        !schemas.contains(&"code_execution"),
        "code_execution must be absent without config"
    );
}
