use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use regex::Regex;
use tokio::sync::RwLock;

use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthScheme {
    Bearer,
    ApiKey,
    None,
}

impl AuthScheme {
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "bearer" => Self::Bearer,
            "api_key" => Self::ApiKey,
            _ => Self::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Exponential backoff
// ---------------------------------------------------------------------------

pub struct BackoffState {
    pub current_ms: u64,
    pub max_ms: u64,
}

impl BackoffState {
    #[must_use]
    pub const fn new(initial_ms: u64, max_ms: u64) -> Self {
        Self {
            current_ms: initial_ms,
            max_ms,
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn next_backoff_interval(state: &mut BackoffState) -> Duration {
    let jitter_factor = (rand::random::<f64>() - 0.5).mul_add(0.4, 1.0); // ±20%
    let ms = ((state.current_ms as f64) * jitter_factor) as u64;
    let ms = ms.min(state.max_ms);
    state.current_ms = ((state.current_ms as f64) * 1.5) as u64;
    state.current_ms = state.current_ms.min(state.max_ms);
    Duration::from_millis(ms)
}

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

/// Returns `(header_name, header_value)` or `None` for `AuthScheme::None`.
///
/// # Errors
///
/// Currently infallible — always returns `Ok`. The `Result` wrapper is kept for
/// forward-compatibility with future auth schemes that may require validation.
pub fn build_auth_header(
    scheme: &AuthScheme,
    token: &str,
) -> Result<Option<(String, String)>, String> {
    match scheme {
        AuthScheme::Bearer => Ok(Some((
            "Authorization".to_string(),
            format!("Bearer {token}"),
        ))),
        AuthScheme::ApiKey => Ok(Some(("X-API-Key".to_string(), token.to_string()))),
        AuthScheme::None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Proposal extraction pipeline
// ---------------------------------------------------------------------------

/// Extract a clean proposal string from raw LLM output.
///
/// # Errors
///
/// Returns `Err` if the input text is empty after trimming (all stages produce no content).
///
/// # Panics
///
/// Panics if the embedded regex literals are invalid (this cannot happen with the
/// hard-coded patterns in this function).
#[allow(clippy::needless_pass_by_value)]
pub fn extract_proposal(text: &str, format: OutputFormat) -> Result<String, String> {
    let trimmed = text.trim();

    // Stage 1: direct JSON parse (no fences)
    if format == OutputFormat::Json && serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return Ok(trimmed.to_owned());
    }

    // Stage 2: strip markdown code fences
    let fence_re = Regex::new(r"```(?:\w+)?\n?([\s\S]*?)```").unwrap();
    if format == OutputFormat::Json {
        let blocks: Vec<&str> = fence_re
            .captures_iter(text)
            .map(|cap| cap.get(1).map_or("", |m| m.as_str()).trim())
            .collect();
        for inner in blocks.iter().rev() {
            if serde_json::from_str::<serde_json::Value>(inner).is_ok() {
                return Ok(inner.to_string());
            }
        }
    } else if let Some(cap) = fence_re.captures(text) {
        return Ok(cap[1].trim().to_owned());
    }

    // Stage 3: strip preamble patterns
    let preamble_re =
        Regex::new(r"(?i)^(here is|the answer is|based on|output:|result:)[^\n]*\n+").unwrap();
    let stripped = preamble_re.replace(trimmed, "").trim().to_owned();
    if !stripped.is_empty() {
        return Ok(stripped);
    }

    // Stage 4: return raw
    if trimmed.is_empty() {
        Err("empty artifact text".to_string())
    } else {
        Ok(trimmed.to_owned())
    }
}

// ---------------------------------------------------------------------------
// Agent Card cache
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct CachedCard {
    card_json: serde_json::Value,
    fetched_at: Instant,
    ttl_secs: u64,
}

impl CachedCard {
    fn is_expired(&self) -> bool {
        self.fetched_at.elapsed().as_secs() >= self.ttl_secs
    }
}

// ---------------------------------------------------------------------------
// A2aExplorerAdapter
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct A2aExplorerAdapter {
    kind: AdapterKind,
    auth_token: Option<String>,
    client: reqwest::Client,
    card_cache: Arc<RwLock<Option<CachedCard>>>,
}

impl A2aExplorerAdapter {
    /// Construct a new `A2aExplorerAdapter`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `auth_scheme` is not `"none"` and `auth_token_env` is non-empty but
    /// the named environment variable is not set.  Also returns `Err` if the underlying
    /// `reqwest::Client` cannot be built.
    pub fn new(
        endpoint: String,
        auth_scheme: String,
        auth_token_env: String,
        timeout_minutes: u64,
        poll_interval_ms: u64,
        max_poll_interval_ms: u64,
        agent_card_cache_ttl_s: u64,
    ) -> Result<Self, String> {
        let auth_token = if auth_scheme != "none" && !auth_token_env.is_empty() {
            Some(
                std::env::var(&auth_token_env).map_err(|_| {
                    format!("A2A adapter: env var `{auth_token_env}` not set (required for auth_scheme={auth_scheme})")
                })?,
            )
        } else {
            None
        };

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .tcp_keepalive(Duration::from_mins(1))
            .pool_idle_timeout(Duration::from_mins(5))
            .build()
            .map_err(|e| format!("failed to build reqwest client: {e}"))?;

        Ok(Self {
            kind: AdapterKind::A2a {
                endpoint,
                auth_scheme,
                auth_token_env,
                timeout_minutes,
                poll_interval_ms,
                max_poll_interval_ms,
                agent_card_cache_ttl_s,
            },
            auth_token,
            client,
            card_cache: Arc::new(RwLock::new(None)),
        })
    }

    fn endpoint(&self) -> &str {
        match &self.kind {
            AdapterKind::A2a { endpoint, .. } => endpoint,
            _ => unreachable!(),
        }
    }

    fn config(&self) -> (&str, u64, u64, u64, u64) {
        match &self.kind {
            AdapterKind::A2a {
                auth_scheme,
                timeout_minutes,
                poll_interval_ms,
                max_poll_interval_ms,
                agent_card_cache_ttl_s,
                ..
            } => (
                auth_scheme.as_str(),
                *timeout_minutes,
                *poll_interval_ms,
                *max_poll_interval_ms,
                *agent_card_cache_ttl_s,
            ),
            _ => unreachable!(),
        }
    }

    fn auth_header(&self) -> Option<(String, String)> {
        let (auth_scheme, ..) = self.config();
        let token = self.auth_token.as_deref().unwrap_or("");
        build_auth_header(&AuthScheme::parse(auth_scheme), token).unwrap_or(None)
    }

    async fn get_agent_card(&self) -> Result<serde_json::Value, AdapterError> {
        let (.., ttl_s) = self.config();

        {
            let guard = self.card_cache.read().await;
            if let Some(ref cached) = *guard {
                if !cached.is_expired() {
                    return Ok(cached.card_json.clone());
                }
            }
        }

        let mut guard = self.card_cache.write().await;
        if let Some(ref cached) = *guard {
            if !cached.is_expired() {
                return Ok(cached.card_json.clone());
            }
        }

        let url = format!("{}/.well-known/agent.json", self.endpoint());
        let mut req = self.client.get(&url);
        if let Some((name, value)) = self.auth_header() {
            req = req.header(name, value);
        }

        let resp = req.send().await.map_err(|e| {
            tracing::warn!(error=%e, "A2A Agent Card fetch failed");
            AdapterError::Unavailable
        })?;

        if !resp.status().is_success() {
            tracing::warn!(status=%resp.status(), "A2A Agent Card returned non-200");
            return Err(AdapterError::Unavailable);
        }

        let card_json: serde_json::Value = resp.json().await.map_err(|e| {
            tracing::warn!(error=%e, "A2A Agent Card JSON parse failed");
            AdapterError::Unavailable
        })?;

        *guard = Some(CachedCard {
            card_json: card_json.clone(),
            fetched_at: Instant::now(),
            ttl_secs: ttl_s,
        });
        drop(guard);

        Ok(card_json)
    }

    fn invalidate_card_cache_sync(&self) {
        if let Ok(mut guard) = self.card_cache.try_write() {
            *guard = None;
        }
    }

    async fn delegate(&self, prompt: &str) -> Result<String, AdapterError> {
        let (_, timeout_min, poll_ms, max_poll_ms, _) = self.config();

        let _card = self.get_agent_card().await?;

        let task_id = self.send_task(prompt).await?;

        let task_deadline = Duration::from_secs(timeout_min * 60);
        let mut backoff = BackoffState::new(poll_ms, max_poll_ms);

        let artifact_text = tokio::time::timeout(task_deadline, async {
            loop {
                tokio::time::sleep(next_backoff_interval(&mut backoff)).await;

                match self.poll_task(&task_id).await {
                    Ok(PollResult::Completed(text)) => return Ok(text),
                    Ok(PollResult::Pending) => {}
                    Ok(PollResult::Failed(reason)) => {
                        self.invalidate_card_cache_sync();
                        return Err(AdapterError::Remote(reason));
                    }
                    Ok(PollResult::Cancelled) => return Err(AdapterError::Cancelled),
                    Ok(PollResult::Rejected) => {
                        self.invalidate_card_cache_sync();
                        return Err(AdapterError::Unavailable);
                    }
                    Ok(PollResult::InputRequired) => return Err(AdapterError::Timeout),
                    Err(AdapterError::Timeout) => {
                        tracing::warn!("A2A poll timed out (15s), retrying");
                    }
                    Err(e) => return Err(e),
                }
            }
        })
        .await
        .map_err(|_| AdapterError::Timeout)??;

        let result = extract_proposal(&artifact_text, OutputFormat::Text).map_err(|e| {
            tracing::warn!(error=%e, raw=%artifact_text, "A2A extraction pipeline failed");
            AdapterError::EmptyOutput
        })?;

        if result.is_empty() {
            return Err(AdapterError::EmptyOutput);
        }

        Ok(result)
    }

    async fn send_task(&self, prompt: &str) -> Result<String, AdapterError> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": 1,
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{ "type": "text", "text": prompt }]
                },
                "configuration": {
                    "acceptedOutputModes": ["text"]
                }
            }
        });

        let mut req = self.client.post(self.endpoint()).json(&body);
        if let Some((name, value)) = self.auth_header() {
            req = req.header(name, value);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| AdapterError::NetworkError(e.to_string()))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AdapterError::NetworkError(e.to_string()))?;

        json["result"]["id"]
            .as_str()
            .map(std::string::ToString::to_string)
            .ok_or_else(|| AdapterError::NetworkError("missing task id in send response".into()))
    }

    async fn poll_task(&self, task_id: &str) -> Result<PollResult, AdapterError> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "id": 2,
            "params": { "id": task_id }
        });

        let mut req = self.client.post(self.endpoint()).json(&body);
        if let Some((name, value)) = self.auth_header() {
            req = req.header(name, value);
        }

        let resp = req.send().await.map_err(|e| {
            if e.is_timeout() {
                AdapterError::Timeout
            } else {
                AdapterError::NetworkError(e.to_string())
            }
        })?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AdapterError::NetworkError(e.to_string()))?;

        let task = &json["result"];
        let state = task["status"]["state"].as_str().unwrap_or("unknown");

        match state {
            "completed" => {
                let text = task["artifacts"][0]["parts"][0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                Ok(PollResult::Completed(text))
            }
            "failed" => {
                let reason = task["status"]["message"]
                    .as_str()
                    .unwrap_or("unknown reason")
                    .to_string();
                Ok(PollResult::Failed(reason))
            }
            "canceled" => Ok(PollResult::Cancelled),
            "rejected" | "auth_required" => Ok(PollResult::Rejected),
            "input_required" => Ok(PollResult::InputRequired),
            // "working" | "submitted" and all unknown states → poll again
            _ => Ok(PollResult::Pending),
        }
    }
}

enum PollResult {
    Completed(String),
    Pending,
    Failed(String),
    Cancelled,
    Rejected,
    InputRequired,
}

#[async_trait]
impl IComputeAdapter for A2aExplorerAdapter {
    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let prompt = format!("{}\n\n{}", request.system_context, request.task);
        let output = self.delegate(&prompt).await?;
        Ok(ComputeResponse {
            output,
            token_cost: 0,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
            reasoning_trace: None,
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}
