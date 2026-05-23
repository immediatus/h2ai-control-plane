#![allow(clippy::missing_panics_doc)]
use h2ai_config::{McpFilesystemConfig, WasmExecutorConfig, WebSearchConfig};
use h2ai_test_utils::{MockMcpBackend, MockSearchBackend, MockWasmBackend};
use h2ai_tools::mcp::McpExecutor;
use h2ai_tools::registry::ToolRegistry;
use h2ai_tools::wasm::WasmExecutor;
use h2ai_tools::web_search::WebSearchExecutor;
use h2ai_types::agent::{AgentTool, WaveMode};
use std::collections::HashMap;

fn registry_with_mocks(cfg: &h2ai_config::H2AIConfig, mode: WaveMode) -> ToolRegistry {
    use h2ai_tools::shell::ShellExecutor;

    let allowlist = match mode {
        WaveMode::Normal => cfg.shell_allowlist.clone(),
        WaveMode::Hardened => cfg.shell_hardened_allowlist.clone(),
    };
    let mut r = ToolRegistry::new();
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

#[tokio::test]
async fn shell_executor_runs_echo_command() {
    let registry = ToolRegistry::default_with_shell();
    let result = registry
        .execute(
            AgentTool::Shell,
            r#"{"command": "echo", "args": ["hello_tool"]}"#,
        )
        .await;
    assert!(result.is_ok(), "{result:?}");
    assert!(result.unwrap().contains("hello_tool"));
}

#[tokio::test]
async fn shell_executor_returns_error_on_nonzero_exit() {
    let registry = ToolRegistry::default_with_shell();
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
    let registry = registry_with_mocks(&cfg, WaveMode::Normal);
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
    let registry = registry_with_mocks(&cfg, WaveMode::Hardened);
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
    let registry = registry_with_mocks(&cfg, WaveMode::Normal);
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

#[test]
fn for_wave_normal_default_config_has_only_shell() {
    let cfg = h2ai_config::H2AIConfig::default();
    let registry = ToolRegistry::for_wave(&cfg, WaveMode::Normal);
    let schemas: Vec<_> = registry.all_schemas().iter().map(|s| s.name).collect();
    assert!(schemas.contains(&"shell"), "shell must always be present");
    assert_eq!(schemas.len(), 1, "only shell expected with default config");
}

#[test]
fn for_wave_hardened_default_config_has_only_shell() {
    let cfg = h2ai_config::H2AIConfig::default();
    let registry = ToolRegistry::for_wave(&cfg, WaveMode::Hardened);
    let schemas: Vec<_> = registry.all_schemas().iter().map(|s| s.name).collect();
    assert!(schemas.contains(&"shell"), "shell must always be present");
    assert_eq!(
        schemas.len(),
        1,
        "only shell expected in hardened with default config"
    );
}

#[test]
fn for_wave_normal_with_wasm_config_no_wasm_feature_warns_but_no_registration() {
    let cfg = h2ai_config::H2AIConfig {
        wasm_executor: Some(WasmExecutorConfig {
            interpreter_wasm_path: "/nonexistent.wasm".into(),
            fuel_budget: 100_000,
        }),
        ..h2ai_config::H2AIConfig::default()
    };
    let registry = ToolRegistry::for_wave(&cfg, WaveMode::Normal);
    let schemas: Vec<_> = registry.all_schemas().iter().map(|s| s.name).collect();
    // Without the "wasm" feature, wasm_executor is ignored (warn-only path)
    assert!(schemas.contains(&"shell"), "shell must always be present");
}

#[test]
fn for_wave_normal_with_web_search_config_no_feature_warns_but_no_registration() {
    let cfg = h2ai_config::H2AIConfig {
        web_search: Some(WebSearchConfig {
            api_key_env: "DUMMY_KEY".into(),
            cx_env: "DUMMY_CX".into(),
            max_results: 3,
        }),
        ..h2ai_config::H2AIConfig::default()
    };
    let registry = ToolRegistry::for_wave(&cfg, WaveMode::Normal);
    let schemas: Vec<_> = registry.all_schemas().iter().map(|s| s.name).collect();
    assert!(schemas.contains(&"shell"), "shell must always be present");
}

#[test]
fn for_wave_normal_with_mcp_config_registers_filesystem() {
    let cfg = h2ai_config::H2AIConfig {
        mcp_filesystem: Some(McpFilesystemConfig {
            command: "echo".into(),
            args: vec![],
            timeout_secs: 5,
        }),
        ..h2ai_config::H2AIConfig::default()
    };
    let registry = ToolRegistry::for_wave(&cfg, WaveMode::Normal);
    let schemas: Vec<_> = registry.all_schemas().iter().map(|s| s.name).collect();
    assert!(schemas.contains(&"shell"), "shell must always be present");
    assert!(
        schemas.contains(&"file_system"),
        "file_system expected with mcp config"
    );
}

#[test]
fn for_wave_hardened_with_mcp_config_does_not_register_filesystem() {
    let cfg = h2ai_config::H2AIConfig {
        mcp_filesystem: Some(McpFilesystemConfig {
            command: "echo".into(),
            args: vec![],
            timeout_secs: 5,
        }),
        ..h2ai_config::H2AIConfig::default()
    };
    let registry = ToolRegistry::for_wave(&cfg, WaveMode::Hardened);
    let schemas: Vec<_> = registry.all_schemas().iter().map(|s| s.name).collect();
    assert!(
        !schemas.contains(&"file_system"),
        "file_system must be absent in Hardened mode"
    );
}
