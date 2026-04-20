use h2ai_tools::registry::ToolRegistry;
use h2ai_types::agent::AgentTool;

#[tokio::test]
async fn shell_executor_runs_echo_command() {
    let registry = ToolRegistry::default_with_shell();
    let result = registry.execute(AgentTool::Shell, "echo hello_tool").await;
    assert!(result.is_ok(), "{:?}", result);
    assert!(result.unwrap().contains("hello_tool"));
}

#[tokio::test]
async fn shell_executor_returns_error_on_nonzero_exit() {
    let registry = ToolRegistry::default_with_shell();
    let result = registry.execute(AgentTool::Shell, "exit 1").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn registry_returns_err_for_unregistered_tool() {
    let registry = ToolRegistry::new();
    let result = registry.execute(AgentTool::WebSearch, "query").await;
    assert!(result.is_err());
}
