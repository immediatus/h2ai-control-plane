use async_trait::async_trait;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use serde::Deserialize;

#[derive(Debug)]
pub struct AnthropicAdapter {
    kind: AdapterKind,
    endpoint: String,
    client: reqwest::Client,
}

impl AnthropicAdapter {
    pub fn new(endpoint: String, api_key_env: String, model: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
            kind: AdapterKind::Anthropic { api_key_env, model },
        }
    }

    fn api_key(&self) -> Result<String, AdapterError> {
        let env_name = match &self.kind {
            AdapterKind::Anthropic { api_key_env, .. } => api_key_env,
            _ => unreachable!(),
        };
        std::env::var(env_name)
            .map_err(|_| AdapterError::NetworkError(format!("env var {env_name} not set")))
    }

    fn model(&self) -> &str {
        match &self.kind {
            AdapterKind::Anthropic { model, .. } => model,
            _ => unreachable!(),
        }
    }
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[async_trait]
impl IComputeAdapter for AnthropicAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let api_key = self.api_key()?;
        let url = format!("{}/v1/messages", self.endpoint.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": self.model(),
            "max_tokens": req.max_tokens,
            "temperature": req.tau.value(),
            "system": req.system_context,
            "messages": [{"role": "user", "content": req.task}]
        });

        let http_resp = self
            .client
            .post(&url)
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
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

        let parsed: AnthropicResponse = http_resp
            .json()
            .await
            .map_err(|e| AdapterError::NetworkError(e.to_string()))?;

        let text_blocks: Vec<String> = parsed
            .content
            .into_iter()
            .filter(|b| b.block_type == "text")
            .map(|b| b.text)
            .collect();

        if text_blocks.is_empty() {
            return Err(AdapterError::NetworkError(
                "no text content in Anthropic response".into(),
            ));
        }

        let output = text_blocks.join("");

        Ok(ComputeResponse {
            output,
            token_cost: parsed.usage.input_tokens + parsed.usage.output_tokens,
            adapter_kind: self.kind.clone(),
            tokens_used: Some(parsed.usage.input_tokens + parsed.usage.output_tokens),
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}
