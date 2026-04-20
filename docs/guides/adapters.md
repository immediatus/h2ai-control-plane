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
pub trait IComputeAdapter: Send + Sync + std::fmt::Debug {
    /// Execute one inference call. Called by the orchestrator per Explorer.
    /// Must be cancel-safe — the orchestrator wraps calls in tokio::time::timeout.
    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError>;

    /// Identifies which backend produced a response (for telemetry and AuditorConfig).
    fn kind(&self) -> &AdapterKind;
}
```

---

## ComputeRequest and ComputeResponse

```rust
pub struct ComputeRequest {
    /// The immutable system_context string compiled at TaskBootstrappedEvent.
    pub system_context: String,

    /// Task description fed to this Explorer.
    pub task: String,

    /// Temperature for this Explorer (τ value assigned by autonomic).
    /// Always 0.0 for the Auditor.
    pub tau: TauValue,

    /// Maximum tokens to generate.
    pub max_tokens: u64,
}

pub struct ComputeResponse {
    /// Raw text output from the model.
    pub output: String,

    /// Actual tokens consumed (prompt + completion).
    pub token_cost: u64,

    /// Which adapter kind produced this response (for telemetry / ProposalEvent).
    pub adapter_kind: AdapterKind,
}

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("adapter timed out before producing output")]
    Timeout,

    #[error("adapter OOM panic: {0}")]
    OomPanic(String),

    #[error("network error: {0}")]
    NetworkError(String),

    #[error("FFI error from llama.cpp: {0}")]
    FfiError(String),
}
```

---

## Built-in adapters

The `h2ai-adapters` crate ships five concrete adapters. All are wired through `AdapterFactory::build`.

### AnthropicAdapter

```rust
AnthropicAdapter::new(
    "https://api.anthropic.com".into(),  // endpoint
    "ANTHROPIC_API_KEY".into(),          // env var name holding the key
    "claude-3-5-sonnet-20241022".into(), // model
)
```

Sends POST `/v1/messages` with `x-api-key` and `anthropic-version: 2023-06-01` headers. Parses `content[].type == "text"` blocks. Token cost = `input_tokens + output_tokens`.

### OpenAIAdapter

```rust
OpenAIAdapter::new(
    "https://api.openai.com/v1".into(), // endpoint
    "OPENAI_API_KEY".into(),            // env var name
    "gpt-4o".into(),                    // model
)
```

Sends POST `/chat/completions` with `Authorization: Bearer`. Sends `"model"` in the request body. Token cost = `usage.total_tokens`. Returns error if `choices` is empty.

### OllamaAdapter

```rust
OllamaAdapter::new(
    "http://localhost:11434".into(), // endpoint (no trailing slash)
    "llama3.2".into(),              // model
)
```

Sends POST `/api/chat` with no auth header. Temperature is nested as `"options": {"temperature": τ}` (not top-level). `stream` is always `false`. `prompt_eval_count` and `eval_count` are `#[serde(default)]` — token cost gracefully falls back to 0 for cached responses.

### AdapterFactory

```rust
use h2ai_adapters::factory::AdapterFactory;
use h2ai_types::config::AdapterKind;

let kind = AdapterKind::Anthropic {
    api_key_env: "ANTHROPIC_API_KEY".into(),
    model: "claude-3-5-haiku-20241022".into(),
};
let adapter: Arc<dyn IComputeAdapter> = AdapterFactory::build(&kind)?;
```

`AdapterFactory::build` maps all five `AdapterKind` variants to their concrete adapter. `LocalLlamaCpp` returns `Err` — use `Ollama` with a local Ollama server for local inference until the FFI is wired.

---

## Example: Custom HTTP adapter

---

## Example: Local llama.cpp adapter

Local adapters **must** use `tokio::task::spawn_blocking` — llama.cpp inference is CPU-bound and must not run on the async worker pool.

```rust
// crates/h2ai-adapters/src/local.rs

use async_trait::async_trait;
use h2ai_types::{AdapterKind, AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};

pub struct LocalAdapter {
    model_path: std::path::PathBuf,
    n_threads: usize,
    kind: AdapterKind,
}

impl LocalAdapter {
    pub fn new(model_path: std::path::PathBuf, n_threads: usize) -> Self {
        Self {
            kind: AdapterKind::LocalLlamaCpp {
                model_path: model_path.clone(),
                n_threads,
            },
            model_path,
            n_threads,
        }
    }
}

#[async_trait]
impl IComputeAdapter for LocalAdapter {
    fn kind(&self) -> &AdapterKind { &self.kind }

    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let model_path = self.model_path.clone();
        let n_threads = self.n_threads;
        let kind = self.kind.clone();

        // REQUIRED: spawn_blocking for all CPU-bound FFI work.
        // This runs on Tokio's bounded blocking pool, never the async worker pool.
        tokio::task::spawn_blocking(move || {
            // llama.cpp FFI calls go here
            let output = llama_cpp_ffi::generate(
                &model_path,
                &request.system_context,
                &request.task,
                request.tau.value() as f32,
                request.max_tokens,
                n_threads,
            ).map_err(|e| match e {
                llama_cpp_ffi::Error::OOM => AdapterError::OomPanic("model context exceeded available RAM".into()),
                llama_cpp_ffi::Error::Other(s) => AdapterError::FfiError(s),
            })?;

            Ok(ComputeResponse {
                output: output.text,
                token_cost: output.token_count,
                adapter_kind: kind,
            })
        })
        .await
        .map_err(|e| AdapterError::FfiError(format!("spawn_blocking join error: {e}")))?
    }
}
```

---

## Selecting an adapter at runtime

The explorer and auditor adapters are selected by env var at startup. Set `H2AI_EXPLORER_PROVIDER` (and `H2AI_AUDITOR_PROVIDER`) to one of `anthropic`, `openai`, `ollama`, `cloud`, or `mock` (default).

```bash
# Anthropic explorer + auditor
H2AI_EXPLORER_PROVIDER=anthropic
H2AI_EXPLORER_MODEL=claude-3-5-sonnet-20241022
H2AI_EXPLORER_API_KEY_ENV=ANTHROPIC_API_KEY

H2AI_AUDITOR_PROVIDER=anthropic
H2AI_AUDITOR_MODEL=claude-3-5-haiku-20241022
H2AI_AUDITOR_API_KEY_ENV=ANTHROPIC_API_KEY

ANTHROPIC_API_KEY=sk-ant-...
```

See [Configuration Reference](../reference/configuration.md#llm-adapters) for the full variable list and provider defaults.

To use a custom adapter with the `AdapterFactory` dispatch path, add a new `AdapterKind` variant to `h2ai-types/src/config.rs` and add the matching arm in `h2ai-adapters/src/factory.rs`.

---

## Testing an adapter

Tests for adapters live in `crates/h2ai-adapters/tests/`. The trait interface makes unit testing straightforward — no orchestrator or NATS required.

Real LLM integration tests live in `crates/h2ai-adapters/tests/integration_test.rs` and are `#[ignore]` by default. Run them with:

```bash
ANTHROPIC_API_KEY=sk-ant-... \
cargo test -p h2ai-adapters --test integration_test -- --ignored --nocapture
```

Unit tests use `wiremock` to stand up a local HTTP server — no live credentials needed:

```rust
// crates/h2ai-adapters/tests/openai_test.rs

use h2ai_adapters::openai::OpenAIAdapter;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::physics::TauValue;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_adapter_returns_output() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"content": "4"}}],
            "usage": {"total_tokens": 12}
        })))
        .mount(&server)
        .await;

    let adapter = OpenAIAdapter::new(server.uri(), "OPENAI_API_KEY".into(), "gpt-4o-mini".into());
    let resp = adapter.execute(ComputeRequest {
        system_context: "You are a calculator.".into(),
        task: "2 + 2?".into(),
        tau: TauValue::new(0.1).unwrap(),
        max_tokens: 16,
    }).await.unwrap();

    assert_eq!(resp.output, "4");
    assert_eq!(resp.token_cost, 12);
}
```

---

## Auditor adapter requirements

The Auditor is a specialized adapter with additional constraints:

1. **τ is always 0.0** — deterministic, no sampling variance. The adapter must respect `request.tau = 0.0` by setting temperature to zero or using greedy decoding.
2. **`role_error_cost` must be ≥ 0.85** — this is what triggers `BftConsensus` when required. Setting it lower defeats the safety guarantee.
3. **The Auditor's system prompt includes the constraint validation rubric** — compiled automatically from the constraint corpus. Do not override `system_context` in an Auditor adapter.
4. **The Auditor must be a capable reasoning model** — it is not an Explorer draft. Use the largest, most capable model available for the Auditor role.

The Auditor adapter is identical in implementation to any cloud adapter. The distinction is in configuration (`role_error_cost = 0.9`) and routing (the orchestrator sends Auditor requests with `tau = 0.0`).
