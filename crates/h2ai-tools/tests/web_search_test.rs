use async_trait::async_trait;
use h2ai_test_utils::MockSearchBackend;
use h2ai_tools::error::ToolError;
use h2ai_tools::web_search::{
    GeminiSearchBackend, WebSearchBackend, WebSearchExecutor, WikipediaSearchBackend,
};
use h2ai_tools::ToolExecutor;
use std::sync::{Arc, Mutex};

// Recording backend: captures the query forwarded by the executor.
struct RecordingBackend {
    received_query: Arc<Mutex<String>>,
}

#[async_trait]
impl WebSearchBackend for RecordingBackend {
    async fn search(&self, query: &str, _max_results: usize) -> Result<String, ToolError> {
        query.clone_into(&mut self.received_query.lock().unwrap());
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

/// Live: Wikipedia backend — free, no API key, returns real snippets.
/// Soft-skips only on network error.
#[tokio::test]
async fn live_wikipedia_search_returns_real_text() {
    let backend = WikipediaSearchBackend::new();
    match backend
        .search("Redis rate limiting sliding window", 3)
        .await
    {
        Err(e) => eprintln!("Wikipedia unreachable (skipping): {e}"),
        Ok(text) => {
            println!("Wikipedia response:\n{text}");
            assert!(
                !text.is_empty() && text != "No results found.",
                "Expected real Wikipedia snippets, got: {text}"
            );
            let lower = text.to_lowercase();
            assert!(
                lower.contains("rate limit")
                    || lower.contains("sliding")
                    || lower.contains("window")
                    || lower.contains("redis")
                    || lower.contains("counter"),
                "Expected at least one relevant keyword;\ngot:\n{text}"
            );
        }
    }
}

/// Live: Gemini backend with Google Search grounding.
/// Reads `GEMINI_API_KEY` from env; soft-skips if absent or quota exhausted.
#[tokio::test]
async fn live_gemini_search_returns_real_grounded_text() {
    let api_key = match std::env::var("GEMINI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("GEMINI_API_KEY not set — skipping live Gemini search test");
            return;
        }
    };

    let backend = GeminiSearchBackend::new(api_key);
    match backend
        .search("Redis sliding window rate limiting algorithm", 5)
        .await
    {
        Err(e) => eprintln!("Gemini API error (skipping): {e}"),
        Ok(text) => {
            println!("Gemini grounded response:\n{text}");
            assert!(
                !text.is_empty() && text != "No results found.",
                "Expected real grounded text, got: {text}"
            );
            let lower = text.to_lowercase();
            assert!(
                lower.contains("redis")
                    || lower.contains("rate limit")
                    || lower.contains("sliding")
                    || lower.contains("window")
                    || lower.contains("counter")
                    || lower.contains("token")
                    || lower.contains("lua"),
                "Expected at least one relevant technical keyword;\ngot:\n{text}"
            );
        }
    }
}
