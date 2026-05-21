use h2ai_agent::config_validation::validate_tool_configs;
use h2ai_agent::tools::agent_tools;
use h2ai_config::{H2AIConfig, McpFilesystemConfig, WasmExecutorConfig, WebSearchConfig};
use h2ai_types::agent::AgentTool;

#[test]
fn absent_sections_do_not_panic() {
    let cfg = H2AIConfig::default();
    validate_tool_configs(&cfg);
}

#[test]
#[should_panic(expected = "GOOGLE_SEARCH_API_KEY")]
fn present_web_search_config_with_missing_env_panics() {
    std::env::remove_var("GOOGLE_SEARCH_API_KEY_TEST_ONLY");
    let cfg = H2AIConfig {
        web_search: Some(WebSearchConfig {
            api_key_env: "GOOGLE_SEARCH_API_KEY_TEST_ONLY".into(),
            cx_env: "GOOGLE_SEARCH_CX_TEST_ONLY".into(),
            max_results: 3,
        }),
        ..H2AIConfig::default()
    };
    validate_tool_configs(&cfg);
}

#[test]
#[should_panic(expected = "nonexistent_quickjs.wasm")]
fn present_wasm_config_with_missing_file_panics() {
    let cfg = H2AIConfig {
        wasm_executor: Some(WasmExecutorConfig {
            interpreter_wasm_path: "nonexistent_quickjs.wasm".into(),
            fuel_budget: 100_000,
        }),
        ..H2AIConfig::default()
    };
    validate_tool_configs(&cfg);
}

#[test]
fn valid_wasm_config_with_existing_file_does_not_panic() {
    let path = "/tmp/h2ai_test_dummy.wasm";
    std::fs::write(path, b"dummy").unwrap();
    let cfg = H2AIConfig {
        wasm_executor: Some(WasmExecutorConfig {
            interpreter_wasm_path: path.into(),
            fuel_budget: 100_000,
        }),
        ..H2AIConfig::default()
    };
    validate_tool_configs(&cfg);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn mcp_filesystem_config_present_does_not_panic() {
    let cfg = H2AIConfig {
        mcp_filesystem: Some(McpFilesystemConfig {
            command: "echo".into(),
            args: vec![],
            timeout_secs: 5,
        }),
        ..H2AIConfig::default()
    };
    validate_tool_configs(&cfg);
}

#[test]
fn agent_tools_list_is_complete() {
    let tools = agent_tools();
    assert!(tools.contains(&AgentTool::Shell));
    assert!(tools.contains(&AgentTool::FileSystem));
}
