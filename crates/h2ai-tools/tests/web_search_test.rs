use async_trait::async_trait;
use h2ai_tools::error::ToolError;
use h2ai_tools::web_search::{MockSearchBackend, WebSearchBackend, WebSearchExecutor};
use h2ai_tools::ToolExecutor;
use std::sync::{Arc, Mutex};

// Recording backend: captures the query forwarded by the executor.
struct RecordingBackend {
    received_query: Arc<Mutex<String>>,
}

#[async_trait]
impl WebSearchBackend for RecordingBackend {
    async fn search(&self, query: &str, _max_results: usize) -> Result<String, ToolError> {
        *self.received_query.lock().unwrap() = query.to_owned();
        Ok("recorded".into())
    }
}

#[tokio::test]
async fn web_search_executor_forwards_query_to_backend() {
    let received = Arc::new(Mutex::new(String::new()));
    let executor = WebSearchExecutor::new(
        Box::new(RecordingBackend {
            received_query: received.clone(),
        }),
        3,
    );
    executor
        .execute(r#"{"query": "rust async traits"}"#)
        .await
        .unwrap();
    assert_eq!(*received.lock().unwrap(), "rust async traits");
}

#[tokio::test]
async fn web_search_executor_returns_backend_output() {
    let executor =
        WebSearchExecutor::new(Box::new(MockSearchBackend::new("search result text")), 3);
    let result = executor.execute(r#"{"query": "anything"}"#).await.unwrap();
    assert_eq!(result, "search result text");
}

#[tokio::test]
async fn web_search_executor_rejects_malformed_input() {
    let executor = WebSearchExecutor::new(Box::new(MockSearchBackend::new("x")), 3);
    let result = executor.execute("not json").await;
    assert!(result.is_err());
    let e = result.unwrap_err().to_string();
    assert!(
        e.contains("malformed"),
        "expected malformed error, got: {e}"
    );
}

#[tokio::test]
async fn web_search_executor_rejects_missing_query_field() {
    let executor = WebSearchExecutor::new(Box::new(MockSearchBackend::new("x")), 3);
    let result = executor.execute(r#"{"q": "something"}"#).await;
    assert!(result.is_err());
    let e = result.unwrap_err().to_string();
    assert!(
        e.contains("missing") || e.contains("query") || e.contains("malformed"),
        "error must point at missing query field; got: {e}"
    );
}

#[tokio::test]
async fn web_search_schema_has_correct_name() {
    let executor = WebSearchExecutor::new(Box::new(MockSearchBackend::new("x")), 3);
    assert_eq!(executor.schema().name, "web_search");
}
