use h2ai_test_utils::mock_mcp;
use h2ai_tools::mcp::{McpBackend as _, McpExecutor};
use h2ai_tools::ToolExecutor;

fn mock_backend() -> h2ai_test_utils::MockMcpClient {
    let mut files = std::collections::HashMap::new();
    files.insert(
        "reference.toml".to_string(),
        "agent_max_tool_iterations = 5".to_string(),
    );
    files.insert("src/".to_string(), "lib.rs\nmain.rs".to_string());
    mock_mcp(files)
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

#[tokio::test]
async fn mcp_executor_missing_path_field_returns_error() {
    let executor = McpExecutor::new(Box::new(mock_backend()));
    let input = r#"{"op": "read_file"}"#;
    let result = executor.execute(input).await;
    assert!(result.is_err());
    let e = result.unwrap_err().to_string();
    assert!(e.contains("malformed") || e.contains("missing"), "got: {e}");
}

#[tokio::test]
async fn mcp_executor_missing_op_field_returns_error() {
    let executor = McpExecutor::new(Box::new(mock_backend()));
    let input = r#"{"path": "reference.toml"}"#;
    let result = executor.execute(input).await;
    assert!(result.is_err());
    let e = result.unwrap_err().to_string();
    assert!(e.contains("malformed") || e.contains("missing"), "got: {e}");
}

#[test]
fn stdio_mcp_backend_new_constructs_without_panic() {
    use h2ai_tools::mcp::StdioMcpBackend;
    let _backend = StdioMcpBackend::new("echo", vec!["hello".into()], 5);
}

/// `StdioMcpBackend::call` — happy path.
///
/// Uses `sh -c 'echo <json>'` as the MCP server: it ignores stdin and writes
/// a valid JSON-RPC 2.0 response to stdout, which the backend parses.
#[tokio::test]
async fn stdio_mcp_backend_call_returns_content_text() {
    use h2ai_tools::mcp::StdioMcpBackend;

    // Build a JSON-RPC response the backend will parse:
    //   result.content[0].text → the extracted content string
    let response_json = r#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"hello from mock mcp"}]}}"#;
    let script = format!("echo '{response_json}'");
    let backend = StdioMcpBackend::new("sh", vec!["-c".into(), script], 5);

    let result = backend.call("read_file", "test.txt").await;
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
    assert_eq!(result.unwrap(), "hello from mock mcp");
}

/// `StdioMcpBackend::call` — error-response branch.
///
/// When the subprocess returns a JSON-RPC error object the backend maps it to
/// `ToolError::ExecutionFailed`.
#[tokio::test]
async fn stdio_mcp_backend_call_returns_execution_error_on_json_rpc_error() {
    use h2ai_tools::mcp::StdioMcpBackend;

    let error_json =
        r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"file not found"}}"#;
    let script = format!("echo '{error_json}'");
    let backend = StdioMcpBackend::new("sh", vec!["-c".into(), script], 5);

    let result = backend.call("read_file", "missing.txt").await;
    assert!(result.is_err(), "expected Err for JSON-RPC error response");
}

/// `StdioMcpBackend::call` — spawn failure branch.
///
/// When the command does not exist the backend returns `ToolError::InitializationFailed`.
#[tokio::test]
async fn stdio_mcp_backend_call_fails_when_command_not_found() {
    use h2ai_tools::mcp::StdioMcpBackend;

    let backend = StdioMcpBackend::new("__no_such_binary_xyz__", vec![], 5);
    let result = backend.call("read_file", "test.txt").await;
    assert!(result.is_err(), "expected Err when binary missing");
}

/// `StdioMcpBackend::call` — timeout branch.
///
/// When the subprocess never writes a response the call times out after 1 second
/// and returns `ToolError::Timeout`.
#[tokio::test]
async fn stdio_mcp_backend_call_times_out_when_no_response() {
    use h2ai_tools::mcp::StdioMcpBackend;

    // `sleep 2` never writes to stdout; the 1-second timeout fires first.
    let backend = StdioMcpBackend::new("sleep", vec!["2".into()], 1);
    let result = backend.call("read_file", "test.txt").await;
    assert!(result.is_err(), "expected timeout error");
}
