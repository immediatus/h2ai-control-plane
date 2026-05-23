use async_trait::async_trait;
use h2ai_tools::error::ToolError;
use h2ai_tools::mcp::McpBackend;
use h2ai_tools::wasm::WasmBackend;
use h2ai_tools::web_search::WebSearchBackend;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use std::collections::HashMap;

// ── Tool mocks ────────────────────────────────────────────────────────────────

pub struct MockWasmBackend {
    response: String,
}

impl MockWasmBackend {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

#[async_trait]
impl WasmBackend for MockWasmBackend {
    async fn execute_script(&self, _language: &str, _script: &str) -> Result<String, ToolError> {
        Ok(self.response.clone())
    }
}

pub struct MockSearchBackend {
    response: String,
}

impl MockSearchBackend {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

#[async_trait]
impl WebSearchBackend for MockSearchBackend {
    async fn search(&self, _query: &str, _max_results: usize) -> Result<String, ToolError> {
        Ok(self.response.clone())
    }
}

pub struct MockMcpBackend {
    files: HashMap<String, String>,
}

impl MockMcpBackend {
    #[must_use]
    pub fn new(files: HashMap<String, String>) -> Self {
        Self { files }
    }
}

#[async_trait]
impl McpBackend for MockMcpBackend {
    async fn call(&self, _op: &str, path: &str) -> Result<String, ToolError> {
        self.files
            .get(path)
            .cloned()
            .ok_or_else(|| ToolError::MalformedInput(format!("path not found: {path}")))
    }
}

// ── Adapter mocks ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct MockAdapter {
    output: String,
    kind: AdapterKind,
}

impl MockAdapter {
    #[must_use]
    pub fn new(output: String) -> Self {
        Self {
            output,
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://localhost".into(),
                api_key_env: "MOCK".into(),
                model: None,
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for MockAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Ok(ComputeResponse {
            output: self.output.clone(),
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

/// Returns valid decomposition JSON for STEP3 calls (detected by "JSON formatter" in system
/// context), and `fallback_output` for all other calls. Use in e2e tests that exercise the
/// full engine pipeline — the mandatory Path C decomposition step requires valid `RawSlot` JSON.
#[derive(Debug)]
pub struct DecompositionMockAdapter {
    fallback_output: String,
    kind: AdapterKind,
}

impl DecompositionMockAdapter {
    #[must_use]
    pub fn new(fallback_output: String) -> Self {
        Self {
            fallback_output,
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://decomposition".into(),
                api_key_env: "MOCK".into(),
                model: None,
            },
        }
    }
}

const STEP3_MOCK_JSON: &str = r#"[
  {
    "role_frame": "Expert A: analyzes the task from a systems perspective",
    "cot_style": "step_by_step",
    "focus_mandate": "break down the problem into components",
    "rejection_criteria": "vague or unsupported claims",
    "constraint_domains": [],
    "search_enabled": false
  },
  {
    "role_frame": "Expert B: evaluates tradeoffs and edge cases",
    "cot_style": "devil_s_advocate",
    "focus_mandate": "surface hidden risks and counterarguments",
    "rejection_criteria": "overconfident assertions without evidence",
    "constraint_domains": [],
    "search_enabled": false
  }
]"#;

#[async_trait]
impl IComputeAdapter for DecompositionMockAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let output = if req.system_context.contains("JSON formatter") {
            STEP3_MOCK_JSON.to_string()
        } else {
            self.fallback_output.clone()
        };
        Ok(ComputeResponse {
            output,
            token_cost: 10,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
            reasoning_trace: None,
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

/// Returns responses from a fixed sequence, one per call. Exhausted sequence returns `"fallback"`.
#[derive(Debug)]
pub struct SequencedMockAdapter {
    responses: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    kind: AdapterKind,
}

impl SequencedMockAdapter {
    #[must_use]
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses: std::sync::Arc::new(std::sync::Mutex::new(responses)),
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://sequenced".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for SequencedMockAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let output = {
            let mut lock = self.responses.lock().unwrap();
            if lock.is_empty() {
                "fallback".into()
            } else {
                lock.remove(0)
            }
        };
        Ok(ComputeResponse {
            output,
            token_cost: 10,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
            reasoning_trace: None,
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

/// Always returns `Err(AdapterError::NetworkError)`. Use to exercise `adapter_failed` paths.
#[derive(Debug)]
pub struct FailingMockAdapter {
    kind: AdapterKind,
}

impl FailingMockAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://failing".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
        }
    }
}

impl Default for FailingMockAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl IComputeAdapter for FailingMockAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Err(AdapterError::NetworkError("mock network failure".into()))
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}
