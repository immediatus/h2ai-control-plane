use crate::error::ToolError;
use crate::{ToolExecutor, ToolSchema};
use async_trait::async_trait;
use serde_json::json;
use std::fmt::Write as FmtWrite;

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

pub struct GoogleSearchBackend {
    api_key: String,
    cx: String,
    client: reqwest::Client,
}

impl GoogleSearchBackend {
    pub fn new(api_key: impl Into<String>, cx: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            cx: cx.into(),
            client: reqwest::Client::new(),
        }
    }
}

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
            // Link omitted: URLs waste token budget and can cause LLMs to hallucinate citations.
            let _ = link;
            writeln!(out, "[{}] {} — {}", i + 1, title, snippet).unwrap();
        }
        if out.is_empty() {
            out = "No results found.".into();
        }
        Ok(out)
    }
}

// ── Live: DuckDuckGo Instant Answer API (no key required) ────────────────────

pub struct DuckDuckGoSearchBackend {
    client: reqwest::Client,
}

impl Default for DuckDuckGoSearchBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DuckDuckGoSearchBackend {
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                // DDG blocks default reqwest UA; mimic a real browser.
                .user_agent(
                    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
                )
                .build()
                .unwrap_or_default(),
        }
    }

    /// Parses result snippets out of DDG's lite HTML response.
    fn parse_html_snippets(html: &str, max_results: usize) -> Vec<String> {
        // DDG lite wraps each result snippet in <a class="result-link"> … </a>
        // followed by the snippet text. We use simple string scanning — no dep needed.
        let mut snippets = Vec::new();
        // Each result block contains a <td class="result-snippet"> … </td>
        let marker = "class=\"result-snippet\"";
        let mut cursor = 0;
        while snippets.len() < max_results {
            let Some(start) = html[cursor..].find(marker) else {
                break;
            };
            let abs = cursor + start;
            // Find the content between > and </td>
            let Some(gt) = html[abs..].find('>') else {
                cursor = abs + 1;
                continue;
            };
            let content_start = abs + gt + 1;
            let Some(end) = html[content_start..].find("</td>") else {
                cursor = content_start;
                continue;
            };
            let raw = &html[content_start..content_start + end];
            // Strip inline tags (e.g. <b>, <wbr>)
            let snippet = Self::strip_tags(raw).trim().to_string();
            if !snippet.is_empty() {
                snippets.push(snippet);
            }
            cursor = content_start + end + 5;
        }
        snippets
    }

    fn strip_tags(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut in_tag = false;
        for c in s.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => out.push(c),
                _ => {}
            }
        }
        // Collapse HTML entities minimally
        out.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#x27;", "'")
            .replace("&nbsp;", " ")
    }
}

#[async_trait]
impl WebSearchBackend for DuckDuckGoSearchBackend {
    async fn search(&self, query: &str, max_results: usize) -> Result<String, ToolError> {
        // DDG Lite: returns plain HTML with actual web search snippets.
        let resp = self
            .client
            .post("https://lite.duckduckgo.com/lite/")
            .form(&[("q", query)])
            .send()
            .await
            .map_err(|e| ToolError::NetworkError(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(ToolError::NetworkError(format!(
                "DuckDuckGo Lite returned {}",
                resp.status()
            )));
        }

        let html = resp.text().await.map_err(|e| {
            ToolError::NetworkError(format!("failed to read DDG Lite response: {e}"))
        })?;

        let snippets = Self::parse_html_snippets(&html, max_results);
        if snippets.is_empty() {
            return Ok("No results found.".into());
        }

        let out = snippets
            .iter()
            .enumerate()
            .map(|(i, s)| format!("[{}] {s}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(out)
    }
}

// ── Shared HTML utilities ─────────────────────────────────────────────────────

/// Strip HTML tags, decode common entities, collapse whitespace.
/// Skips content inside `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`.
pub(crate) fn extract_text_from_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 4);
    let mut in_tag = false;
    let mut tag_buf = String::new();
    let mut skip_depth: u32 = 0;
    let mut prev_newline = false;

    let mut chars = html.char_indices();
    while let Some((_, ch)) = chars.next() {
        if in_tag {
            if ch == '>' {
                in_tag = false;
                let t = tag_buf.trim_start_matches('/').to_lowercase();
                let opening = !tag_buf.starts_with('/');
                let closing = tag_buf.starts_with('/');
                let skip_tags = ["script", "style", "nav", "header", "footer", "aside"];
                let block_tags = ["p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "td"];
                if opening && skip_tags.iter().any(|s| t.starts_with(s)) {
                    skip_depth += 1;
                } else if closing && skip_tags.iter().any(|s| t.starts_with(s)) {
                    skip_depth = skip_depth.saturating_sub(1);
                } else if skip_depth == 0
                    && block_tags.iter().any(|s| t.starts_with(s))
                    && !out.ends_with('\n')
                {
                    out.push('\n');
                    prev_newline = true;
                }
                tag_buf.clear();
            } else if tag_buf.len() < 16 {
                tag_buf.push(ch);
            }
            continue;
        }
        if ch == '<' {
            in_tag = true;
            tag_buf.clear();
            continue;
        }
        if skip_depth > 0 {
            continue;
        }
        // Minimal entity decoding
        if ch == '&' {
            let rest: String = chars.clone().take(8).map(|(_, c)| c).collect();
            let decoded = if rest.starts_with("amp;") {
                Some(("&", 4))
            } else if rest.starts_with("lt;") {
                Some(("<", 3))
            } else if rest.starts_with("gt;") {
                Some((">", 3))
            } else if rest.starts_with("nbsp;") || rest.starts_with("#160;") {
                Some((" ", 5))
            } else if rest.starts_with("quot;") {
                Some(("\"", 5))
            } else {
                None
            };
            if let Some((entity_str, skip)) = decoded {
                for _ in 0..skip {
                    chars.next();
                }
                out.push_str(entity_str);
                prev_newline = false;
                continue;
            }
        }
        if ch == '\n' || ch == '\r' {
            if !prev_newline {
                out.push('\n');
                prev_newline = true;
            }
        } else if ch == ' ' || ch == '\t' {
            if !out.ends_with(' ') && !out.ends_with('\n') {
                out.push(' ');
            }
        } else {
            out.push(ch);
            prev_newline = false;
        }
    }

    // Collapse consecutive blank lines, trim each line
    out.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Live: StackOverflow via StackExchange API (no key required) ───────────────

/// Searches `StackOverflow` for questions matching the query, then fetches accepted/top
/// answers. Uses the `StackExchange` API (free, 300 req/day per IP, no key needed).
///
/// Returns rich text — question title + top answer bodies — so the distiller
/// receives real implementation knowledge rather than encyclopedic intros.
pub struct StackOverflowSearchBackend {
    client: reqwest::Client,
    /// SE site to search: "stackoverflow", "softwareengineering", etc.
    site: String,
}

impl Default for StackOverflowSearchBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StackOverflowSearchBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::with_site("stackoverflow")
    }

    pub fn with_site(site: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("h2ai-control-plane/0.1 (grounding-research)")
                .timeout(std::time::Duration::from_secs(12))
                .build()
                .unwrap_or_default(),
            site: site.into(),
        }
    }

    async fn fetch_top_answers(&self, question_id: u64) -> Vec<String> {
        let url = format!(
            "https://api.stackexchange.com/2.3/questions/{question_id}/answers\
             ?site={}&sort=votes&order=desc&pagesize=2&filter=withbody",
            self.site
        );
        let Ok(resp) = self.client.get(&url).send().await else {
            return vec![];
        };
        if !resp.status().is_success() {
            return vec![];
        }
        let Ok(body) = resp.json::<serde_json::Value>().await else {
            return vec![];
        };

        body.pointer("/items")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter(|a| a["score"].as_i64().unwrap_or(0) >= 0)
                    .filter_map(|a| a["body"].as_str())
                    .map(|html| {
                        let text = extract_text_from_html(html);
                        // Cap each answer at 600 chars to keep signal dense
                        text.chars().take(600).collect::<String>()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[async_trait]
impl WebSearchBackend for StackOverflowSearchBackend {
    async fn search(&self, query: &str, max_results: usize) -> Result<String, ToolError> {
        let limit = max_results.min(5).to_string();
        let resp = self
            .client
            .get("https://api.stackexchange.com/2.3/search/advanced")
            .query(&[
                ("order", "desc"),
                ("sort", "relevance"),
                ("q", query),
                ("site", &self.site),
                ("pagesize", &limit),
            ])
            .send()
            .await
            .map_err(|e| ToolError::NetworkError(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(ToolError::NetworkError(format!(
                "StackExchange API returned {}",
                resp.status()
            )));
        }

        let body: serde_json::Value = resp.json().await.map_err(|e| {
            ToolError::NetworkError(format!("failed to decode StackExchange response: {e}"))
        })?;

        let items = match body.pointer("/items").and_then(|v| v.as_array()) {
            Some(arr) if !arr.is_empty() => arr.clone(),
            _ => return Ok("No results found.".into()),
        };

        let mut parts = Vec::new();
        for (i, item) in items.iter().enumerate().take(max_results.min(3)) {
            let title = item["title"].as_str().unwrap_or("").to_string();
            let score = item["score"].as_i64().unwrap_or(0);
            let qid = item["question_id"].as_u64().unwrap_or(0);

            let answers = self.fetch_top_answers(qid).await;
            if answers.is_empty() {
                continue;
            }

            let mut block = format!("[{}] {} (score: {})\n", i + 1, title, score);
            for (j, ans) in answers.iter().enumerate() {
                writeln!(block, "Answer {}: {}", j + 1, ans).unwrap();
            }
            parts.push(block);
        }

        if parts.is_empty() {
            return Ok("No results found.".into());
        }
        Ok(parts.join("\n"))
    }
}

// ── Live: Wikipedia Search API (general concept fallback) ─────────────────────

/// Wikipedia backend — useful only for general concept introductions.
/// For technical grounding, prefer `StackOverflowSearchBackend`.
pub struct WikipediaSearchBackend {
    client: reqwest::Client,
}

impl Default for WikipediaSearchBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WikipediaSearchBackend {
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("h2ai-control-plane/0.1 (grounding-research)")
                .build()
                .unwrap_or_default(),
        }
    }
}

#[async_trait]
impl WebSearchBackend for WikipediaSearchBackend {
    async fn search(&self, query: &str, max_results: usize) -> Result<String, ToolError> {
        let limit = max_results.min(5).to_string();
        let search_resp = self
            .client
            .get("https://en.wikipedia.org/w/api.php")
            .query(&[
                ("action", "query"),
                ("list", "search"),
                ("srsearch", query),
                ("format", "json"),
                ("srlimit", &limit),
                ("srprop", ""),
            ])
            .send()
            .await
            .map_err(|e| ToolError::NetworkError(e.to_string()))?;

        if !search_resp.status().is_success() {
            return Err(ToolError::NetworkError(format!(
                "Wikipedia search returned {}",
                search_resp.status()
            )));
        }

        let search_body: serde_json::Value = search_resp.json().await.map_err(|e| {
            ToolError::NetworkError(format!("failed to decode Wikipedia response: {e}"))
        })?;

        let titles: Vec<String> = search_body
            .pointer("/query/search")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| r["title"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        if titles.is_empty() {
            return Ok("No results found.".into());
        }

        let mut parts = Vec::new();
        for (i, title) in titles.iter().enumerate() {
            let url = format!(
                "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
                title.replace(' ', "_")
            );
            let Ok(r) = self.client.get(&url).send().await else {
                continue;
            };
            if !r.status().is_success() {
                continue;
            }
            let Ok(v) = r.json::<serde_json::Value>().await else {
                continue;
            };
            if let Some(extract) = v["extract"].as_str().filter(|s| !s.is_empty()) {
                let short: String = extract
                    .splitn(4, ". ")
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(". ");
                parts.push(format!("[{}] {} — {}", i + 1, title, short));
            }
        }

        if parts.is_empty() {
            return Ok("No results found.".into());
        }
        Ok(parts.join("\n\n"))
    }
}

// ── Live: Gemini API with Google Search grounding ────────────────────────────

/// Uses Gemini's `google_search` tool to ground a query against live web results
/// and return a concise factual summary. Requires a Gemini API key.
pub struct GeminiSearchBackend {
    api_key: String,
    client: reqwest::Client,
    model: String,
}

impl GeminiSearchBackend {
    /// `api_key`: Gemini API key (e.g. from `GEMINI_API_KEY` env var).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
            model: "gemini-2.0-flash".into(),
        }
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl WebSearchBackend for GeminiSearchBackend {
    async fn search(&self, query: &str, _max_results: usize) -> Result<String, ToolError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let body = json!({
            "contents": [{
                "parts": [{"text": query}]
            }],
            "tools": [{"google_search": {}}],
            "systemInstruction": {
                "parts": [{"text":
                    "You are a technical research assistant. \
                     Answer the query concisely using information from the search results. \
                     Focus on factual, specific technical details."
                }]
            }
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| ToolError::NetworkError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ToolError::NetworkError(format!(
                "Gemini API returned {status}: {text}"
            )));
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| {
            ToolError::NetworkError(format!("failed to decode Gemini response: {e}"))
        })?;

        // Extract the text from candidates[0].content.parts[0].text
        let text = json
            .pointer("/candidates/0/content/parts/0/text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        if text.is_empty() {
            return Ok("No results found.".into());
        }
        Ok(text)
    }
}

// ── Live: Multi-source web grounding (Reddit + HN + GitHub + page fetching) ───
//
// Sources that work without API keys and cover diverse content types:
//   • Reddit search JSON  — community discussion, practical experience
//   • HN Algolia          — curated tech articles, blog posts, papers
//   • GitHub search API   — repos with README (60 req/hr unauthenticated)
//
// For each candidate URL the backend fetches the actual page and extracts
// readable text, so the distiller receives real content, not just snippets.

/// A candidate link collected from one of the discovery sources.
struct ScoredLink {
    title: String,
    url: String,
    /// Normalised score (upvotes / stars) used for ranking.
    score: i64,
    /// When true, fetch README via GitHub API (returns clean markdown).
    is_github: bool,
}

/// General-purpose web grounding backend.
///
/// Discovers valuable links from Reddit, Hacker News, and GitHub,
/// then fetches and extracts real page content from each. No API keys required.
pub struct WebGroundingBackend {
    client: reqwest::Client,
    /// Per-page fetch timeout in seconds.
    fetch_timeout_secs: u64,
    /// Maximum characters to extract per fetched page.
    max_page_chars: usize,
}

impl Default for WebGroundingBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WebGroundingBackend {
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent(
                    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
                )
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_default(),
            fetch_timeout_secs: 8,
            max_page_chars: 2000,
        }
    }

    #[must_use]
    pub const fn with_page_chars(mut self, n: usize) -> Self {
        self.max_page_chars = n;
        self
    }

    // ── Discovery ────────────────────────────────────────────────────────────

    async fn search_reddit(&self, query: &str, max: usize) -> Vec<ScoredLink> {
        let enc: String = query
            .chars()
            .map(|c| if c == ' ' { '+' } else { c })
            .collect();
        let url = format!(
            "https://www.reddit.com/search.json?q={enc}&sort=relevance&limit={max}&type=link"
        );
        let Ok(resp) = self.client.get(&url).send().await else {
            return vec![];
        };
        if !resp.status().is_success() {
            return vec![];
        }
        let Ok(body) = resp.json::<serde_json::Value>().await else {
            return vec![];
        };

        body.pointer("/data/children")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|c| {
                        let d = &c["data"];
                        let url = d["url"].as_str()?;
                        // Skip self-posts and direct Reddit links — only external URLs
                        if url.contains("reddit.com") || url.starts_with("https://www.reddit.com") {
                            return None;
                        }
                        Some(ScoredLink {
                            title: d["title"].as_str().unwrap_or("").to_string(),
                            url: url.to_string(),
                            score: d["score"].as_i64().unwrap_or(0),
                            is_github: url.contains("github.com"),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn search_hn(&self, query: &str, max: usize) -> Vec<ScoredLink> {
        let enc: String = query
            .chars()
            .map(|c| if c == ' ' { '+' } else { c })
            .collect();
        let url = format!(
            "https://hn.algolia.com/api/v1/search?query={enc}&tags=story&hitsPerPage={max}"
        );
        let Ok(resp) = self.client.get(&url).send().await else {
            return vec![];
        };
        if !resp.status().is_success() {
            return vec![];
        }
        let Ok(body) = resp.json::<serde_json::Value>().await else {
            return vec![];
        };

        body["hits"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|h| {
                        let url = h["url"].as_str().filter(|u| !u.is_empty())?;
                        Some(ScoredLink {
                            title: h["title"].as_str().unwrap_or("").to_string(),
                            url: url.to_string(),
                            score: h["points"].as_i64().unwrap_or(0),
                            is_github: url.contains("github.com"),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn search_github(&self, query: &str, max: usize) -> Vec<ScoredLink> {
        let enc: String = query
            .chars()
            .map(|c| if c == ' ' { '+' } else { c })
            .collect();
        let url =
            format!("https://api.github.com/search/repositories?q={enc}&sort=stars&per_page={max}");
        let Ok(resp) = self
            .client
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
        else {
            return vec![];
        };
        if !resp.status().is_success() {
            return vec![];
        }
        let Ok(body) = resp.json::<serde_json::Value>().await else {
            return vec![];
        };

        body["items"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| {
                        let url = r["html_url"].as_str()?;
                        let desc = r["description"].as_str().unwrap_or("");
                        let title = format!(
                            "{} — {}",
                            r["full_name"].as_str().unwrap_or(""),
                            if desc.is_empty() {
                                "no description"
                            } else {
                                desc
                            }
                        );
                        Some(ScoredLink {
                            title,
                            url: url.to_string(),
                            score: r["stargazers_count"].as_i64().unwrap_or(0),
                            is_github: true,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    // ── Fetching ─────────────────────────────────────────────────────────────

    /// Fetch a GitHub repo README via the GitHub API (returns clean markdown).
    async fn fetch_github_readme(&self, owner: &str, repo: &str) -> Option<String> {
        use std::io::Read;
        let url = format!("https://api.github.com/repos/{owner}/{repo}/readme");
        let resp = self
            .client
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let body = resp.json::<serde_json::Value>().await.ok()?;
        let encoded = body["content"].as_str()?;
        // GitHub returns base64 with newlines
        let decoded = encoded.replace('\n', "");
        let bytes = {
            let mut buf = Vec::new();
            let mut dec = base64_decoder(decoded.as_bytes());
            dec.read_to_end(&mut buf).ok()?;
            buf
        };
        let text = String::from_utf8_lossy(&bytes).into_owned();
        // Strip markdown badges/image lines, keep readable content
        let clean: String = text
            .lines()
            .filter(|l| !l.trim_start().starts_with("[![") && !l.trim_start().starts_with("!["))
            .take(80)
            .collect::<Vec<_>>()
            .join("\n");
        Some(clean.chars().take(self.max_page_chars).collect())
    }

    /// Fetch an arbitrary URL and extract readable text from its HTML.
    async fn fetch_page(&self, url: &str) -> Option<String> {
        let resp = tokio::time::timeout(
            std::time::Duration::from_secs(self.fetch_timeout_secs),
            self.client
                .get(url)
                .header("Accept", "text/html,application/xhtml+xml,text/plain")
                .send(),
        )
        .await
        .ok()?
        .ok()?;

        if !resp.status().is_success() {
            return None;
        }

        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !ct.contains("text/html")
            && !ct.contains("text/plain")
            && !ct.contains("application/xhtml")
        {
            return None;
        }

        let html = tokio::time::timeout(std::time::Duration::from_secs(5), resp.text())
            .await
            .ok()?
            .ok()?;

        let text = extract_text_from_html(&html);
        if text.len() < 50 {
            return None;
        }
        Some(text.chars().take(self.max_page_chars).collect())
    }
}

/// Minimal base64 decoder — avoids pulling in a new dependency.
fn base64_decoder(input: &[u8]) -> impl std::io::Read + '_ {
    struct B64Reader<'a> {
        buf: &'a [u8],
        pos: usize,
    }
    impl std::io::Read for B64Reader<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            const TABLE: &[u8; 128] = b"\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x40\x3e\x40\x40\x40\x3f\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x40\x40\x40\x40\x40\x40\x40\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x40\x40\x40\x40\x40\x40\x1a\x1b\x1c\x1d\x1e\x1f\x20\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\x30\x31\x32\x33\x40\x40\x40\x40\x40";
            let mut written = 0;
            while written + 3 <= out.len() && self.pos + 4 <= self.buf.len() {
                let chunk = &self.buf[self.pos..self.pos + 4];
                if chunk[0] == b'=' {
                    break;
                }
                let v = [chunk[0], chunk[1], chunk[2], chunk[3]].map(|b| {
                    if b < 128 {
                        TABLE[b as usize]
                    } else {
                        0x40
                    }
                });
                if v.contains(&0x40) {
                    self.pos += 4;
                    continue;
                }
                out[written] = (v[0] << 2) | (v[1] >> 4);
                written += 1;
                if chunk[2] != b'=' {
                    out[written] = (v[1] << 4) | (v[2] >> 2);
                    written += 1;
                }
                if chunk[3] != b'=' {
                    out[written] = (v[2] << 6) | v[3];
                    written += 1;
                }
                self.pos += 4;
            }
            Ok(written)
        }
    }
    B64Reader { buf: input, pos: 0 }
}

/// Parse `owner` and `repo` from a GitHub URL like `https://github.com/owner/repo`.
fn parse_github_repo(url: &str) -> Option<(String, String)> {
    let path = url.strip_prefix("https://github.com/")?;
    let parts: Vec<&str> = path.splitn(3, '/').collect();
    if parts.len() >= 2 {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

#[async_trait]
impl WebSearchBackend for WebGroundingBackend {
    async fn search(&self, query: &str, max_results: usize) -> Result<String, ToolError> {
        // 1. Discover candidate URLs from all three sources concurrently.
        let (reddit, hn, github) = tokio::join!(
            self.search_reddit(query, max_results + 2),
            self.search_hn(query, max_results + 2),
            self.search_github(query, 3),
        );

        // 2. Merge and rank: deduplicate by URL prefix, sort by score.
        let mut links: Vec<ScoredLink> = reddit.into_iter().chain(hn).chain(github).collect();
        links.sort_by_key(|b| std::cmp::Reverse(b.score));
        let mut seen_urls: Vec<String> = Vec::new();
        links.retain(|l| {
            // Deduplicate: skip if a URL with the same first 60 chars already seen.
            let key = l.url.chars().take(60).collect::<String>();
            if seen_urls.contains(&key) {
                return false;
            }
            seen_urls.push(key);
            true
        });

        // 3. Fetch top N pages and extract content.
        let fetch_limit = max_results.min(4);
        let mut parts: Vec<String> = Vec::new();
        for link in links.iter().take(fetch_limit) {
            let content = if link.is_github {
                // GitHub: prefer clean README markdown over raw HTML
                let readme = if let Some((owner, repo)) = parse_github_repo(&link.url) {
                    self.fetch_github_readme(&owner, &repo).await
                } else {
                    None
                };
                readme
            } else {
                self.fetch_page(&link.url).await
            };

            if let Some(text) = content {
                parts.push(format!(
                    "[{}] {} (score: {})\n{}",
                    parts.len() + 1,
                    link.title,
                    link.score,
                    text
                ));
            }
        }

        if parts.is_empty() {
            return Ok("No results found.".into());
        }
        Ok(parts.join("\n\n---\n\n"))
    }
}

// ── Executor ─────────────────────────────────────────────────────────────────

pub struct WebSearchExecutor {
    backend: Box<dyn WebSearchBackend>,
    max_results: usize,
}

impl WebSearchExecutor {
    #[must_use]
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
