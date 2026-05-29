use async_trait::async_trait;
use h2ai_tools::error::ToolError;
use h2ai_tools::web_search::{
    DuckDuckGoSearchBackend, GeminiSearchBackend, GoogleSearchBackend, StackOverflowSearchBackend,
    WebSearchBackend, WebSearchExecutor, WikipediaSearchBackend,
};
use h2ai_tools::ToolExecutor;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mockall::mock! {
    pub Search {}

    #[async_trait]
    impl WebSearchBackend for Search {
        async fn search(&self, query: &str, max_results: usize) -> Result<String, ToolError>;
    }
}

// ── WebSearchExecutor unit tests ──────────────────────────────────────────────

#[tokio::test]
async fn web_search_executor_forwards_query_to_backend() {
    let mut mock = MockSearch::new();
    mock.expect_search()
        .withf(|q, _| q == "rust async traits")
        .once()
        .returning(|_, _| Ok("recorded".into()));
    let executor = WebSearchExecutor::new(Box::new(mock), 3);
    executor
        .execute(r#"{"query": "rust async traits"}"#)
        .await
        .unwrap();
}

#[tokio::test]
async fn web_search_executor_returns_backend_output() {
    let mut mock = MockSearch::new();
    mock.expect_search()
        .returning(|_, _| Ok("search result text".into()));
    let executor = WebSearchExecutor::new(Box::new(mock), 3);
    let result = executor.execute(r#"{"query": "anything"}"#).await.unwrap();
    assert_eq!(result, "search result text");
}

#[tokio::test]
async fn web_search_executor_rejects_malformed_input() {
    let executor = WebSearchExecutor::new(Box::new(MockSearch::new()), 3);
    let err = executor.execute("not json").await.unwrap_err().to_string();
    assert!(
        err.contains("malformed"),
        "expected malformed error, got: {err}"
    );
}

#[tokio::test]
async fn web_search_executor_rejects_missing_query_field() {
    let executor = WebSearchExecutor::new(Box::new(MockSearch::new()), 3);
    let err = executor
        .execute(r#"{"q": "something"}"#)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("missing") || err.contains("query") || err.contains("malformed"),
        "error must point at missing query field; got: {err}"
    );
}

#[tokio::test]
async fn web_search_schema_has_correct_name() {
    let executor = WebSearchExecutor::new(Box::new(MockSearch::new()), 3);
    assert_eq!(executor.schema().name, "web_search");
}

// ── Google Search wiremock tests ──────────────────────────────────────────────

#[tokio::test]
async fn google_search_returns_formatted_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "items": [
                {"title": "Rust lang", "snippet": "Systems language", "link": "https://rust-lang.org"},
                {"title": "Async Rust", "snippet": "Futures guide", "link": "https://example.com"}
            ]
        })))
        .mount(&server)
        .await;
    let backend = GoogleSearchBackend::new("key", "cx").with_base_url(server.uri());
    let result = backend.search("rust", 3).await.unwrap();
    assert!(result.contains("[1] Rust lang"), "got: {result}");
    assert!(result.contains("Systems language"), "got: {result}");
    assert!(result.contains("[2] Async Rust"), "got: {result}");
}

#[tokio::test]
async fn google_search_non_200_returns_network_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(403))
        .mount(&server)
        .await;
    let backend = GoogleSearchBackend::new("bad-key", "cx").with_base_url(server.uri());
    let err = backend.search("rust", 3).await.unwrap_err().to_string();
    assert!(err.contains("403") || err.contains("Google"), "got: {err}");
}

#[tokio::test]
async fn google_search_empty_items_returns_no_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&server)
        .await;
    let backend = GoogleSearchBackend::new("key", "cx").with_base_url(server.uri());
    let result = backend.search("nothing", 3).await.unwrap();
    assert_eq!(result, "No results found.");
}

// ── DuckDuckGo wiremock tests ─────────────────────────────────────────────────

#[tokio::test]
async fn duckduckgo_search_returns_snippets() {
    let server = MockServer::start().await;
    let html = r#"<html><td class="result-snippet">Rust systems language</td></html>"#;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(html))
        .mount(&server)
        .await;
    let backend = DuckDuckGoSearchBackend::new().with_base_url(server.uri());
    let result = backend.search("rust", 3).await.unwrap();
    assert!(result.contains("Rust systems language"), "got: {result}");
}

#[tokio::test]
async fn duckduckgo_search_non_200_returns_network_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;
    let backend = DuckDuckGoSearchBackend::new().with_base_url(server.uri());
    let err = backend.search("rust", 3).await.unwrap_err().to_string();
    assert!(
        err.contains("503") || err.contains("DuckDuckGo"),
        "got: {err}"
    );
}

#[tokio::test]
async fn duckduckgo_search_no_snippets_returns_no_results() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string("<html><body>nothing here</body></html>"),
        )
        .mount(&server)
        .await;
    let backend = DuckDuckGoSearchBackend::new().with_base_url(server.uri());
    let result = backend.search("rust", 3).await.unwrap();
    assert_eq!(result, "No results found.");
}

// ── StackOverflow wiremock tests ──────────────────────────────────────────────

#[tokio::test]
async fn stackoverflow_search_returns_answers() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/search/advanced"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "items": [{"question_id": 42, "title": "Async in Rust", "score": 10}]
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/questions/42/answers"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "items": [{"body": "<p>Use tokio runtime</p>", "score": 5}]
        })))
        .mount(&server)
        .await;
    let backend = StackOverflowSearchBackend::new().with_base_url(server.uri());
    let result = backend.search("async rust", 2).await.unwrap();
    assert!(result.contains("Async in Rust"), "got: {result}");
    assert!(result.contains("tokio runtime"), "got: {result}");
}

#[tokio::test]
async fn stackoverflow_search_non_200_returns_network_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let backend = StackOverflowSearchBackend::new().with_base_url(server.uri());
    let err = backend.search("rust", 2).await.unwrap_err().to_string();
    assert!(
        err.contains("500") || err.contains("StackExchange"),
        "got: {err}"
    );
}

#[tokio::test]
async fn stackoverflow_search_empty_items_returns_no_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"items": []})))
        .mount(&server)
        .await;
    let backend = StackOverflowSearchBackend::new().with_base_url(server.uri());
    let result = backend.search("nothing", 2).await.unwrap();
    assert_eq!(result, "No results found.");
}

// ── Wikipedia wiremock tests ──────────────────────────────────────────────────

#[tokio::test]
async fn wikipedia_search_returns_summaries() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/w/api.php"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "query": {"search": [{"title": "Rust programming language"}]}
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/rest_v1/page/summary/Rust_programming_language"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "extract": "Rust is a systems programming language. It is memory safe. It is fast."
        })))
        .mount(&server)
        .await;
    let backend = WikipediaSearchBackend::new().with_base_url(server.uri());
    let result = backend.search("rust", 1).await.unwrap();
    assert!(result.contains("Rust"), "got: {result}");
    assert!(result.contains("systems programming"), "got: {result}");
}

#[tokio::test]
async fn wikipedia_search_non_200_returns_network_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/w/api.php"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;
    let backend = WikipediaSearchBackend::new().with_base_url(server.uri());
    let err = backend.search("rust", 1).await.unwrap_err().to_string();
    assert!(
        err.contains("503") || err.contains("Wikipedia"),
        "got: {err}"
    );
}

#[tokio::test]
async fn wikipedia_search_empty_titles_returns_no_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/w/api.php"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"query": {"search": []}})))
        .mount(&server)
        .await;
    let backend = WikipediaSearchBackend::new().with_base_url(server.uri());
    let result = backend.search("nothing", 1).await.unwrap();
    assert_eq!(result, "No results found.");
}

// ── Gemini wiremock tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_search_returns_grounded_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "Redis is an in-memory data structure store."}]}
            }]
        })))
        .mount(&server)
        .await;
    let backend = GeminiSearchBackend::new("test-key")
        .with_base_url(server.uri())
        .with_model("test-model");
    let result = backend.search("redis", 3).await.unwrap();
    assert!(result.contains("Redis"), "got: {result}");
}

#[tokio::test]
async fn gemini_search_non_200_returns_network_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_string("quota exceeded"))
        .mount(&server)
        .await;
    let backend = GeminiSearchBackend::new("test-key").with_base_url(server.uri());
    let err = backend.search("redis", 3).await.unwrap_err().to_string();
    assert!(err.contains("429") || err.contains("Gemini"), "got: {err}");
}

#[tokio::test]
async fn gemini_search_empty_text_returns_no_results() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{"content": {"parts": [{"text": "  "}]}}]
        })))
        .mount(&server)
        .await;
    let backend = GeminiSearchBackend::new("test-key").with_base_url(server.uri());
    let result = backend.search("redis", 3).await.unwrap();
    assert_eq!(result, "No results found.");
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
