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
            "options": {
                "temperature": req.tau.value()
            }
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
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}
