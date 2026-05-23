use h2ai_test_utils::MockMcpBackend;
use h2ai_tools::mcp::McpExecutor;
use h2ai_tools::ToolExecutor;
use std::collections::HashMap;

fn mock_backend() -> MockMcpBackend {
    let mut files = HashMap::new();
    files.insert(
        "reference.toml".to_string(),
        "agent_max_tool_iterations = 5".to_string(),
    );
    files.insert("src/".to_string(), "lib.rs\nmain.rs".to_string());
    MockMcpBackend::new(files)
}

#[tokio::test]
async fn mcp_executor_reads_known_file() {
    let executor = McpExecutor::new(Box::new(mock_backend()));
    let input = r#"{"op": "read_file", "path": "reference.toml"}"#;
    let result = executor.execute(input).await.unwrap();
    assert!(
        result.contains("agent_max_tool_iterations"),
        "got: {result}"
    );
}

#[tokio::test]
async fn mcp_executor_lists_known_directory() {
    let executor = McpExecutor::new(Box::new(mock_backend()));
    let input = r#"{"op": "list_directory", "path": "src/"}"#;
    let result = executor.execute(input).await.unwrap();
    assert!(result.contains("lib.rs"), "got: {result}");
}

#[tokio::test]
async fn mcp_executor_rejects_write_file_op() {
    let executor = McpExecutor::new(Box::new(mock_backend()));
    let input = r#"{"op": "write_file", "path": "x.txt"}"#;
    let result = executor.execute(input).await;
    assert!(result.is_err());
    let e = result.unwrap_err().to_string();
    assert!(
        e.contains("not allowed") || e.contains("permitted"),
        "got: {e}"
    );
}

#[tokio::test]
async fn mcp_executor_rejects_malformed_input() {
    let executor = McpExecutor::new(Box::new(mock_backend()));
    let result = executor.execute("not json").await;
    assert!(result.is_err());
    let e = result.unwrap_err().to_string();
    assert!(
        e.contains("malformed"),
        "expected malformed error, got: {e}"
    );
}

#[tokio::test]
async fn mcp_executor_unknown_path_returns_error() {
    let executor = McpExecutor::new(Box::new(mock_backend()));
    let input = r#"{"op": "read_file", "path": "nonexistent.rs"}"#;
    let result = executor.execute(input).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn mcp_schema_has_correct_name() {
    let executor = McpExecutor::new(Box::new(mock_backend()));
    assert_eq!(executor.schema().name, "file_system");
}
