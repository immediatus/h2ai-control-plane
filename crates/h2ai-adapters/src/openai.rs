use async_trait::async_trait;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use serde::Deserialize;

#[derive(Debug)]
pub struct OpenAIAdapter {
    kind: AdapterKind,
    endpoint: String,
    client: reqwest::Client,
    enable_thinking: bool,
}

impl OpenAIAdapter {
    pub fn new(endpoint: String, api_key_env: String, model: String) -> Self {
        Self::with_thinking(endpoint, api_key_env, model, true)
    }

    pub fn with_thinking(
        endpoint: String,
        api_key_env: String,
        model: String,
        enable_thinking: bool,
    ) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
            kind: AdapterKind::OpenAI { api_key_env, model },
            enable_thinking,
        }
    }

    fn api_key(&self) -> Result<String, AdapterError> {
        let env_name = match &self.kind {
            AdapterKind::OpenAI { api_key_env, .. } => api_key_env,
            _ => unreachable!(),
        };
        std::env::var(env_name)
            .map_err(|_| AdapterError::NetworkError(format!("env var {env_name} not set")))
    }

    fn model(&self) -> &str {
        match &self.kind {
            AdapterKind::OpenAI { model, .. } => model,
            _ => unreachable!(),
        }
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
    /// "stop" = natural finish; "length" = max_tokens reached mid-generation.
    #[serde(default)]
    finish_reason: String,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: String,
    /// Reasoning-only models (DeepSeek R1) put their entire answer here and always
    /// leave `content` empty — use as fallback when `finish_reason == "stop"`.
    /// When `finish_reason == "length"`, the answer was never generated; return error.
    #[serde(default)]
    reasoning_content: Option<String>,
}

/// Extract the answer and optional reasoning trace from a completed choice.
///
/// - Two-phase models (e.g. future OpenAI o-series with separate reasoning field):
///   `content` holds the answer; `reasoning_content` is returned as the trace.
/// - Reasoning-only models: `content` is empty; `reasoning_content` is the full output.
///   The trace is promoted to `output`; no separate trace is returned.
/// - `finish_reason == "length"` with empty content: model exhausted tokens inside the
///   thinking phase — the answer was never generated; fail fast.
fn extract_output(choice: Choice) -> Result<(String, Option<String>), AdapterError> {
    if !choice.message.content.is_empty() {
        return Ok((choice.message.content, choice.message.reasoning_content));
    }
    if choice.finish_reason == "length" {
        return Err(AdapterError::NetworkError(
            "model hit max_tokens during thinking phase; increase max_tokens and retry".into(),
        ));
    }
    Ok((choice.message.reasoning_content.unwrap_or_default(), None))
}

#[derive(Deserialize)]
struct Usage {
    total_tokens: u64,
}

#[async_trait]
impl IComputeAdapter for OpenAIAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let api_key = self.api_key()?;
        let url = format!("{}/chat/completions", self.endpoint.trim_end_matches('/'));

        let mut body = serde_json::json!({
            "model": self.model(),
            "messages": [
                {"role": "system", "content": req.system_context},
                {"role": "user",   "content": req.task}
            ],
            "temperature": req.tau.value(),
            "max_tokens":  req.max_tokens
        });

        if !self.enable_thinking {
            body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
        }

        let http_resp = self
            .client
            .post(&url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| AdapterError::NetworkError(e.to_string()))?;

        if !http_resp.status().is_success() {
            let status = http_resp.status();
            let body = http_resp.text().await.unwrap_or_default();
            return Err(AdapterError::NetworkError(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let chat: ChatResponse = http_resp
            .json()
            .await
            .map_err(|e| AdapterError::NetworkError(e.to_string()))?;

        let choice =
            chat.choices.into_iter().next().ok_or_else(|| {
                AdapterError::NetworkError("no choices in OpenAI response".into())
            })?;
        let (output, reasoning_trace) = extract_output(choice)?;

        Ok(ComputeResponse {
            output,
            token_cost: chat.usage.total_tokens,
            adapter_kind: self.kind.clone(),
            tokens_used: Some(chat.usage.total_tokens),
            reasoning_trace,
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}
