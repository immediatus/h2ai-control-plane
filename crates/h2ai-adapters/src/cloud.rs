use async_trait::async_trait;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use serde::Deserialize;

#[derive(Debug)]
pub struct CloudGenericAdapter {
    kind: AdapterKind,
    client: reqwest::Client,
    enable_thinking: bool,
}

impl CloudGenericAdapter {
    pub fn new(endpoint: String, api_key_env: String, model: Option<String>) -> Self {
        Self::with_thinking(endpoint, api_key_env, model, true)
    }

    pub fn with_thinking(
        endpoint: String,
        api_key_env: String,
        model: Option<String>,
        enable_thinking: bool,
    ) -> Self {
        Self {
            kind: AdapterKind::CloudGeneric {
                endpoint,
                api_key_env,
                model,
            },
            client: reqwest::Client::new(),
            enable_thinking,
        }
    }

    fn endpoint(&self) -> &str {
        match &self.kind {
            AdapterKind::CloudGeneric { endpoint, .. } => endpoint,
            _ => unreachable!(),
        }
    }

    fn api_key(&self) -> Result<String, AdapterError> {
        let env_name = match &self.kind {
            AdapterKind::CloudGeneric { api_key_env, .. } => api_key_env,
            _ => unreachable!(),
        };
        if env_name.is_empty() {
            return Ok(String::new());
        }
        std::env::var(env_name)
            .map_err(|_| AdapterError::NetworkError(format!("env var {env_name} not set")))
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

/// Select the answer text from a completed choice.
///
/// - Prefer `content` when non-empty (two-phase thinking models, standard models).
/// - Fall back to `reasoning_content` only when `finish_reason == "stop"` and
///   `content` is empty — this is the reasoning-only model design (DeepSeek R1 etc.).
/// - Return an error when `finish_reason == "length"` and `content` is empty: the
///   model ran out of tokens mid-thinking; the answer was never generated.
fn extract_output(choice: Choice) -> Result<String, AdapterError> {
    if !choice.message.content.is_empty() {
        return Ok(choice.message.content);
    }
    if choice.finish_reason == "length" {
        return Err(AdapterError::NetworkError(
            "model hit max_tokens during thinking phase; increase max_tokens and retry".into(),
        ));
    }
    Ok(choice.message.reasoning_content.unwrap_or_default())
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
    /// Reasoning-only models (DeepSeek R1) put their entire answer here and
    /// always leave `content` empty — use as output when `content` is absent
    /// AND `finish_reason` is "stop".  When `finish_reason` is "length", the
    /// model ran out of tokens mid-thinking; `content` is empty but the answer
    /// was never generated — return error so callers can retry with more tokens.
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    total_tokens: u64,
}

#[async_trait]
impl IComputeAdapter for CloudGenericAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let api_key = self.api_key()?;
        let url = format!("{}/chat/completions", self.endpoint().trim_end_matches('/'));

        let model = match &self.kind {
            AdapterKind::CloudGeneric { model, .. } => model.clone(),
            _ => unreachable!(),
        };
        let mut body = serde_json::json!({
            "messages": [
                {"role": "system", "content": req.system_context},
                {"role": "user",   "content": req.task}
            ],
            "temperature": req.tau.value(),
            "max_tokens":  req.max_tokens
        });
        if let Some(m) = model {
            body["model"] = serde_json::Value::String(m);
        }
        if !self.enable_thinking {
            body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
        }

        // Retry on 429 (server busy / rate-limited) with capped exponential backoff.
        // Up to 15 attempts, delay capped at 30s — covers a full 180s local LLM slot.
        let mut delay_secs = 3u64;
        let mut attempts = 0u32;
        let http_resp = loop {
            let mut builder = self.client.post(&url).json(&body);
            if !api_key.is_empty() {
                builder = builder.bearer_auth(api_key.clone());
            }
            let resp = builder
                .send()
                .await
                .map_err(|e| AdapterError::NetworkError(e.to_string()))?;
            if resp.status().as_u16() == 429 {
                attempts += 1;
                if attempts >= 15 {
                    return Err(AdapterError::NetworkError(
                        "HTTP 429 Too Many Requests".into(),
                    ));
                }
                tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                delay_secs = (delay_secs * 2).min(30);
                continue;
            }
            break resp;
        };

        if !http_resp.status().is_success() {
            return Err(AdapterError::NetworkError(format!(
                "HTTP {}",
                http_resp.status()
            )));
        }

        let chat: ChatResponse = http_resp
            .json()
            .await
            .map_err(|e| AdapterError::NetworkError(e.to_string()))?;

        let choice = chat
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| AdapterError::NetworkError("no choices in response".into()))?;
        let output = extract_output(choice)?;
        Ok(ComputeResponse {
            output,
            token_cost: chat.usage.total_tokens,
            adapter_kind: self.kind.clone(),
            tokens_used: Some(chat.usage.total_tokens),
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}
