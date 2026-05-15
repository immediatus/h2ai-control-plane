use async_trait::async_trait;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use serde::Deserialize;

#[derive(Debug)]
pub struct OllamaAdapter {
    kind: AdapterKind,
    endpoint: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaAdapter {
    pub fn new(endpoint: String, model: String) -> Self {
        Self {
            kind: AdapterKind::Ollama {
                endpoint: endpoint.clone(),
                model: model.clone(),
            },
            endpoint,
            model,
            client: reqwest::Client::new(),
        }
    }
}

/// Translate tau into Ollama `options` sampling parameters.
///
/// Mirrors the OpenAI adapter's three-regime strategy; Ollama uses the same
/// parameter names inside its `options` block.
fn sampling_options(tau: f64) -> serde_json::Value {
    if tau < 0.35 {
        serde_json::json!({
            "temperature": tau,
            "top_k": 20, "top_p": 0.85, "min_p": 0.05,
            "repeat_penalty": 1.1, "repeat_last_n": 64
        })
    } else if tau <= 0.65 {
        serde_json::json!({
            "temperature": tau,
            "top_k": 40, "top_p": 0.95, "min_p": 0.03,
            "repeat_penalty": 1.05, "repeat_last_n": 64
        })
    } else {
        serde_json::json!({
            "temperature": tau,
            "top_k": 0,
            "mirostat": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
            "repeat_penalty": 1.1, "repeat_last_n": 64
        })
    }
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: OllamaMessage,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    eval_count: u64,
}

#[derive(Deserialize)]
struct OllamaMessage {
    content: String,
}

#[async_trait]
impl IComputeAdapter for OllamaAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let url = format!("{}/api/chat", self.endpoint.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": self.model,
            "stream": false,
            "messages": [
                {"role": "system", "content": req.system_context},
                {"role": "user",   "content": req.task}
            ],
            "options": sampling_options(req.tau.value())
        });

        let http_resp = self
            .client
            .post(&url)
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

        let parsed: OllamaResponse = http_resp
            .json()
            .await
            .map_err(|e| AdapterError::NetworkError(e.to_string()))?;

        Ok(ComputeResponse {
            output: parsed.message.content,
            token_cost: parsed.prompt_eval_count + parsed.eval_count,
            adapter_kind: self.kind.clone(),
            tokens_used: Some(parsed.prompt_eval_count + parsed.eval_count),
            reasoning_trace: None,
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}
