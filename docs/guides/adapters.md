# Adapter Development Guide

An adapter is any compute backend that implements the `IComputeAdapter` trait. The orchestrator calls adapters through this interface without knowing whether the backend is a local llama.cpp model, a cloud API, or something else entirely.

This guide covers implementing a custom adapter, testing it, and registering it with the adapter pool.

---

## The IComputeAdapter trait

Defined in `crates/h2ai-types/src/adapter.rs`:

```rust
use async_trait::async_trait;
use crate::{AdapterKind, ComputeRequest, ComputeResponse, AdapterError};

#[async_trait]
pub trait IComputeAdapter: Send + Sync + 'static {
    /// Human-readable identifier for this adapter instance.
    fn id(&self) -> &str;

    /// Whether this adapter runs locally (blocking pool) or remotely (async pool).
    fn kind(&self) -> AdapterKind;

    /// Execute one inference call. Called by the orchestrator per Explorer.
    /// Must be cancel-safe — the orchestrator wraps calls in tokio::time::timeout.
    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError>;

    /// Called by the calibration harness. Returns p_correct on a reference task set.
    /// Default implementation calls execute() on calibration tasks and measures accuracy.
    async fn calibrate(&self, tasks: &[CalibrationTask]) -> CalibrationResult {
        // default implementation provided — override for efficiency
    }
}
```

---

## ComputeRequest and ComputeResponse

```rust
pub struct ComputeRequest {
    /// Task description compiled with system_context from the bootstrap event.
    pub prompt: String,

    /// Temperature for this Explorer (τ value assigned by autonomic).
    /// Always 0.0 for the Auditor.
    pub tau: f64,

    /// Maximum tokens to generate.
    pub max_tokens: u32,

    /// The system_context string — immutable, set at TaskBootstrappedEvent.
    pub system_context: String,

    /// Explorer ID assigned by the orchestrator.
    pub explorer_id: ExplorerId,

    /// Task ID for tracing correlation.
    pub task_id: TaskId,
}

pub struct ComputeResponse {
    /// Raw text output from the model.
    pub output: String,

    /// Actual tokens consumed (prompt + completion).
    pub token_cost: u32,

    /// Wall time for the inference call.
    pub latency_ms: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("adapter timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("out of memory: {detail}")]
    OOM { detail: String },

    #[error("API error {status}: {message}")]
    ApiError { status: u16, message: String },

    #[error("rate limited, retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },

    #[error("model load failed: {reason}")]
    ModelLoad { reason: String },

    #[error("{0}")]
    Other(String),
}
```

---

## Example: Cloud HTTP adapter

```rust
// crates/adapters/src/cloud.rs

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use h2ai_types::{
    AdapterKind, AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter,
};

pub struct CloudAdapter {
    id: String,
    api_base: String,
    api_key: String,
    model: String,
    client: Client,
}

impl CloudAdapter {
    pub fn new(id: String, api_base: String, api_key: String, model: String) -> Self {
        Self {
            id,
            api_base,
            api_key,
            model,
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap(),
        }
    }
}

#[async_trait]
impl IComputeAdapter for CloudAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> AdapterKind {
        AdapterKind::Cloud
    }

    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let start = std::time::Instant::now();

        let body = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                Message { role: "system".into(), content: request.system_context },
                Message { role: "user".into(),   content: request.prompt },
            ],
            temperature: request.tau,
            max_tokens: request.max_tokens,
        };

        let resp = self.client
            .post(format!("{}/chat/completions", self.api_base))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| AdapterError::Other(e.to_string()))?;

        if resp.status() == 429 {
            let retry = resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(60);
            return Err(AdapterError::RateLimited { retry_after_secs: retry });
        }

        if !resp.status().is_success() {
            return Err(AdapterError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let chat: ChatResponse = resp.json().await
            .map_err(|e| AdapterError::Other(e.to_string()))?;

        let choice = chat.choices.into_iter().next()
            .ok_or_else(|| AdapterError::Other("empty choices".into()))?;

        Ok(ComputeResponse {
            output: choice.message.content,
            token_cost: chat.usage.total_tokens,
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }
}

// OpenAI-compatible request/response types
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f64,
    max_tokens: u32,
}

#[derive(Serialize)]
struct Message { role: String, content: String }

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Deserialize)]
struct Choice { message: Message }

#[derive(Deserialize)]
struct Usage { total_tokens: u32 }
```

---

## Example: Local llama.cpp adapter

Local adapters **must** use `tokio::task::spawn_blocking` — llama.cpp inference is CPU-bound and must not run on the async worker pool.

```rust
// crates/adapters/src/local.rs

use async_trait::async_trait;
use h2ai_types::{AdapterKind, AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};

pub struct LocalAdapter {
    id: String,
    model_path: String,
    context_size: u32,
}

#[async_trait]
impl IComputeAdapter for LocalAdapter {
    fn id(&self) -> &str { &self.id }
    fn kind(&self) -> AdapterKind { AdapterKind::Local }

    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let model_path = self.model_path.clone();
        let context_size = self.context_size;

        // REQUIRED: spawn_blocking for all CPU-bound FFI work.
        // This runs on Tokio's bounded blocking pool, never the async worker pool.
        tokio::task::spawn_blocking(move || {
            let start = std::time::Instant::now();

            // llama.cpp FFI calls go here
            let output = llama_cpp_ffi::generate(
                &model_path,
                &request.system_context,
                &request.prompt,
                request.tau as f32,
                request.max_tokens,
                context_size,
            ).map_err(|e| match e {
                llama_cpp_ffi::Error::OOM => AdapterError::OOM { detail: "model context exceeded available RAM".into() },
                llama_cpp_ffi::Error::Other(s) => AdapterError::Other(s),
            })?;

            Ok(ComputeResponse {
                token_cost: output.token_count,
                latency_ms: start.elapsed().as_millis() as u64,
                output: output.text,
            })
        })
        .await
        .map_err(|e| AdapterError::Other(format!("spawn_blocking join error: {e}")))?
    }
}
```

---

## Registering an adapter

Add the adapter instance to the pool in `adapters.toml`:

```toml
[[explorer]]
id = "my-custom-adapter"
kind = "custom"
# Custom adapters are registered by ID in the binary's adapter registry.
# See crates/adapters/src/registry.rs.
role_error_cost = 0.1
```

In `crates/adapters/src/registry.rs`, register the adapter:

```rust
pub fn build_pool(config: &AdapterConfig) -> Vec<Box<dyn IComputeAdapter>> {
    config.explorers.iter().map(|c| match c.kind.as_str() {
        "local"  => Box::new(LocalAdapter::new(...)) as Box<dyn IComputeAdapter>,
        "cloud"  => Box::new(CloudAdapter::new(...)) as Box<dyn IComputeAdapter>,
        "custom" => Box::new(MyCustomAdapter::new(...)) as Box<dyn IComputeAdapter>,
        other    => panic!("unknown adapter kind: {other}"),
    }).collect()
}
```

---

## Testing an adapter

Tests for adapters live in `crates/adapters/tests/`. The trait interface makes unit testing straightforward — no orchestrator or NATS required.

```rust
// crates/adapters/tests/cloud_adapter_test.rs

#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_types::{ComputeRequest, TaskId, ExplorerId};

    // Integration test — requires a real API key in the environment.
    // Run with: cargo nextest run --features integration-tests
    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn cloud_adapter_returns_non_empty_output() {
        let adapter = CloudAdapter::new(
            "test".into(),
            "https://api.openai.com/v1".into(),
            std::env::var("OPENAI_API_KEY").unwrap(),
            "gpt-4o-mini".into(),
        );

        let request = ComputeRequest {
            prompt: "What is 2 + 2?".into(),
            tau: 0.0,
            max_tokens: 64,
            system_context: "You are a precise calculator.".into(),
            explorer_id: ExplorerId::new(),
            task_id: TaskId::new(),
        };

        let response = adapter.execute(request).await.unwrap();
        assert!(!response.output.is_empty());
        assert!(response.token_cost > 0);
    }

    #[tokio::test]
    async fn adapter_error_on_invalid_api_key() {
        let adapter = CloudAdapter::new(
            "test".into(),
            "https://api.openai.com/v1".into(),
            "invalid-key".into(),
            "gpt-4o-mini".into(),
        );

        let request = ComputeRequest {
            prompt: "test".into(),
            tau: 0.0,
            max_tokens: 10,
            system_context: String::new(),
            explorer_id: ExplorerId::new(),
            task_id: TaskId::new(),
        };

        let err = adapter.execute(request).await.unwrap_err();
        assert!(matches!(err, AdapterError::ApiError { status: 401, .. }));
    }
}
```

---

## Auditor adapter requirements

The Auditor is a specialized adapter with additional constraints:

1. **τ is always 0.0** — deterministic, no sampling variance. The adapter must respect `request.tau = 0.0` by setting temperature to zero or using greedy decoding.
2. **`role_error_cost` must be ≥ 0.85** — this is what triggers `BftConsensus` when required. Setting it lower defeats the safety guarantee.
3. **The Auditor's system prompt includes the constraint validation rubric** — compiled automatically from the ADR corpus. Do not override `system_context` in an Auditor adapter.
4. **The Auditor must be a capable reasoning model** — it is not an Explorer draft. Use the largest, most capable model available for the Auditor role.

The Auditor adapter is identical in implementation to any cloud adapter. The distinction is in configuration (`role_error_cost = 0.9`) and routing (the orchestrator sends Auditor requests with `tau = 0.0`).
