use h2ai_tools::wasm::{MockWasmBackend, WasmExecutor};
use h2ai_tools::ToolExecutor;

#[tokio::test]
async fn wasm_executor_returns_mock_output() {
    let executor = WasmExecutor::new(Box::new(MockWasmBackend::new("42")));
    let input = r#"{"language": "javascript", "script": "21 + 21"}"#;
    let result = executor.execute(input).await.unwrap();
    assert_eq!(result, "42");
}

#[tokio::test]
async fn wasm_executor_rejects_unsupported_language() {
    let executor = WasmExecutor::new(Box::new(MockWasmBackend::new("x")));
    let input = r#"{"language": "python", "script": "print(1)"}"#;
    let result = executor.execute(input).await;
    assert!(result.is_err());
    let e = result.unwrap_err().to_string();
    assert!(e.contains("unsupported language"), "got: {e}");
}

#[tokio::test]
async fn wasm_executor_rejects_malformed_input() {
    let executor = WasmExecutor::new(Box::new(MockWasmBackend::new("x")));
    let result = executor.execute("not json").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn wasm_executor_rejects_missing_language_field() {
    let executor = WasmExecutor::new(Box::new(MockWasmBackend::new("x")));
    let result = executor.execute(r#"{"script": "1+1"}"#).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn wasm_schema_has_correct_name() {
    let executor = WasmExecutor::new(Box::new(MockWasmBackend::new("x")));
    assert_eq!(executor.schema().name, "code_execution");
}
