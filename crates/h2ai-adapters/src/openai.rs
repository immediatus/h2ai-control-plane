use async_trait::async_trait;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use serde::Deserialize;

#[derive(Debug)]
pub struct OpenAIAdapter {
    kind: AdapterKind,
    endpoint: String,
    client: reqwest::Client,
}

impl OpenAIAdapter {
    pub fn new(endpoint: String, api_key_env: String, model: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::new(),
            kind: AdapterKind::OpenAI { api_key_env, model },
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

/// Translate tau into llama.cpp-compatible sampling parameters.
///
/// Three regimes keyed on tau (temperature):
/// - Low  (< 0.35): tight top_k=20, conservative nucleus — deterministic quality
/// - Mid  (0.35–0.65): standard top_k=40, balanced nucleus
/// - High (> 0.65): top_k disabled, Mirostat 2.0 — coherent high-entropy diversity
fn sampling_extras(tau: f64) -> serde_json::Value {
    if tau < 0.35 {
        serde_json::json!({
            "top_k": 20, "top_p": 0.85, "min_p": 0.05,
            "repeat_penalty": 1.1, "repeat_last_n": 64
        })
    } else if tau <= 0.65 {
        serde_json::json!({
            "top_k": 40, "top_p": 0.95, "min_p": 0.03,
            "repeat_penalty": 1.05, "repeat_last_n": 64
        })
    } else {
        serde_json::json!({
            "top_k": 0,
            "mirostat": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
            "repeat_penalty": 1.1, "repeat_last_n": 64
        })
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
}

#[derive(Deserialize)]
struct Message {
    content: String,
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

        let extras = sampling_extras(req.tau.value());
        body.as_object_mut()
            .unwrap()
            .extend(extras.as_object().unwrap().clone());

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

        let output = chat
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| AdapterError::NetworkError("no choices in OpenAI response".into()))?;

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

#[cfg(test)]
mod sampling_tests {
    use super::sampling_extras;

    #[test]
    fn low_tau_uses_tight_top_k() {
        let v = sampling_extras(0.2);
        assert_eq!(v["top_k"], 20);
        assert!(v.get("mirostat").is_none());
    }

    #[test]
    fn mid_tau_uses_standard_top_k() {
        let v = sampling_extras(0.5);
        assert_eq!(v["top_k"], 40);
        assert!(v.get("mirostat").is_none());
    }

    #[test]
    fn high_tau_uses_mirostat() {
        let v = sampling_extras(0.8);
        assert_eq!(v["top_k"], 0);
        assert_eq!(v["mirostat"], 2);
    }
}
