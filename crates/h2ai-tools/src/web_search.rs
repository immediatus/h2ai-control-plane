use crate::error::ToolError;
use crate::{ToolExecutor, ToolSchema};
use async_trait::async_trait;
use serde_json::json;

#[async_trait]
pub trait WebSearchBackend: Send + Sync {
    async fn search(&self, query: &str, max_results: usize) -> Result<String, ToolError>;
}

// ── Mock ─────────────────────────────────────────────────────────────────────

pub struct MockSearchBackend {
    response: String,
}

impl MockSearchBackend {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

#[async_trait]
impl WebSearchBackend for MockSearchBackend {
    async fn search(&self, _query: &str, _max_results: usize) -> Result<String, ToolError> {
        Ok(self.response.clone())
    }
}

// ── Live: Google Custom Search API ───────────────────────────────────────────

#[cfg(feature = "web-search")]
pub struct GoogleSearchBackend {
    api_key: String,
    cx: String,
    client: reqwest::Client,
}

#[cfg(feature = "web-search")]
impl GoogleSearchBackend {
    pub fn new(api_key: impl Into<String>, cx: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            cx: cx.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[cfg(feature = "web-search")]
#[async_trait]
impl WebSearchBackend for GoogleSearchBackend {
    async fn search(&self, query: &str, max_results: usize) -> Result<String, ToolError> {
        // Google Custom Search API caps `num` at 10.
        let num = max_results.min(10);
        let num_str = num.to_string();
        let resp = self
            .client
            .get("https://www.googleapis.com/customsearch/v1")
            .query(&[
                ("key", self.api_key.as_str()),
                ("cx", self.cx.as_str()),
                ("q", query),
                ("num", num_str.as_str()),
            ])
            .send()
            .await
            .map_err(|e| ToolError::NetworkError(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(ToolError::NetworkError(format!(
                "Google Search API returned {}",
                resp.status()
            )));
        }

        let body: serde_json::Value = resp.json().await.map_err(|e| {
            ToolError::NetworkError(format!("failed to decode Google API response: {e}"))
        })?;

        let items = body
            .get("items")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = String::new();
        for (i, item) in items.iter().enumerate() {
            let title = item.get("title").and_then(|v| v.as_str()).unwrap_or("");
            let snippet = item.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
            let link = item.get("link").and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!("[{}] {} — {} ({})\n", i + 1, title, snippet, link));
        }
        if out.is_empty() {
            out = "No results found.".into();
        }
        Ok(out)
    }
}

// ── Executor ─────────────────────────────────────────────────────────────────

pub struct WebSearchExecutor {
    backend: Box<dyn WebSearchBackend>,
    max_results: usize,
}

impl WebSearchExecutor {
    pub fn new(backend: Box<dyn WebSearchBackend>, max_results: usize) -> Self {
        Self {
            backend,
            max_results,
        }
    }
}

#[async_trait]
impl ToolExecutor for WebSearchExecutor {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "web_search",
            description: "Search the web and return the top snippets for a query.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, input: &str) -> Result<String, ToolError> {
        let v: serde_json::Value =
            serde_json::from_str(input).map_err(|e| ToolError::MalformedInput(e.to_string()))?;
        let query = v["query"]
            .as_str()
            .ok_or_else(|| ToolError::MalformedInput("missing 'query' field".into()))?;
        self.backend.search(query, self.max_results).await
    }
}
