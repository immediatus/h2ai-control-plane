use async_trait::async_trait;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use serde::Deserialize;

#[derive(Debug)]
pub struct CloudGenericAdapter {
    kind: AdapterKind,
    client: reqwest::Client,
}

impl CloudGenericAdapter {
    pub fn new(endpoint: String, api_key_env: String) -> Self {
        Self {
            kind: AdapterKind::CloudGeneric {
                endpoint,
                api_key_env,
            },
            client: reqwest::Client::new(),
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
        std::env::var(env_name)
            .map_err(|_| AdapterError::NetworkError(format!("env var {env_name} not set")))
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
impl IComputeAdapter for CloudGenericAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let api_key = self.api_key()?;
        let url = format!("{}/chat/completions", self.endpoint().trim_end_matches('/'));

        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": req.system_context},
                {"role": "user",   "content": req.task}
            ],
            "temperature": req.tau.value(),
            "max_tokens":  req.max_tokens
        });

        let http_resp = self
            .client
            .post(&url)
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| AdapterError::NetworkError(e.to_string()))?;

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

        Ok(ComputeResponse {
            output: chat
                .choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .unwrap_or_default(),
            token_cost: chat.usage.total_tokens,
            adapter_kind: self.kind.clone(),
            tokens_used: Some(chat.usage.total_tokens),
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}
