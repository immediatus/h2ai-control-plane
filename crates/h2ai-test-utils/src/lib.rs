use async_trait::async_trait;
use h2ai_tools::error::ToolError;
use h2ai_tools::mcp::McpBackend;
use h2ai_tools::wasm::WasmBackend;
use h2ai_tools::web_search::WebSearchBackend;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn default_kind() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "mock://localhost".into(),
        api_key_env: "MOCK".into(),
        model: None,
        provider: Default::default(),
    }
}

fn make_response(output: impl Into<String>, cost: u64) -> ComputeResponse {
    ComputeResponse {
        output: output.into(),
        token_cost: cost,
        adapter_kind: default_kind(),
        tokens_used: None,
        reasoning_trace: None,
    }
}

// ── Mock declarations ─────────────────────────────────────────────────────────

mockall::mock! {
    #[derive(Debug)]
    pub IComputeAdapter {}

    #[async_trait]
    impl IComputeAdapter for IComputeAdapter {
        async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError>;
        fn kind(&self) -> &AdapterKind;
    }
}

mockall::mock! {
    pub WebSearch {}

    #[async_trait]
    impl WebSearchBackend for WebSearch {
        async fn search(&self, query: &str, max_results: usize) -> Result<String, ToolError>;
    }
}

mockall::mock! {
    pub WasmRunner {}

    #[async_trait]
    impl WasmBackend for WasmRunner {
        async fn execute_script(&self, language: &str, script: &str) -> Result<String, ToolError>;
    }
}

mockall::mock! {
    pub McpClient {}

    #[async_trait]
    impl McpBackend for McpClient {
        async fn call(&self, op: &str, path: &str) -> Result<String, ToolError>;
    }
}

// ── Factory helpers ───────────────────────────────────────────────────────────

/// Mock adapter that always succeeds returning `output`.
pub fn mock_adapter(output: impl Into<String>) -> MockIComputeAdapter {
    let output = output.into();
    let mut m = MockIComputeAdapter::new();
    m.expect_execute()
        .returning(move |_| Ok(make_response(output.clone(), 0)));
    m.expect_kind().return_const(default_kind()).times(0..);
    m
}

/// Mock adapter that always fails with `AdapterError::NetworkError`.
pub fn failing_adapter() -> MockIComputeAdapter {
    let mut m = MockIComputeAdapter::new();
    m.expect_execute()
        .returning(|_| Err(AdapterError::NetworkError("mock network failure".into())));
    m.expect_kind().return_const(default_kind()).times(0..);
    m
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

/// Mock adapter for the decomposition pipeline. Returns STEP3 JSON when
/// `system_context` contains `"JSON formatter"`, `fallback` otherwise.
pub fn decomposition_adapter(fallback: impl Into<String>) -> MockIComputeAdapter {
    let fallback = fallback.into();
    let mut m = MockIComputeAdapter::new();
    m.expect_execute().returning(move |req| {
        let output = if req.system_context.contains("JSON formatter") {
            STEP3_MOCK_JSON.to_string()
        } else {
            fallback.clone()
        };
        Ok(make_response(output, 10))
    });
    m.expect_kind().return_const(default_kind()).times(0..);
    m
}

/// Mock adapter returning `responses` in sequence; returns `"fallback"` when exhausted.
pub fn sequenced_adapter(responses: Vec<String>) -> MockIComputeAdapter {
    let queue = Arc::new(Mutex::new(responses));
    let mut m = MockIComputeAdapter::new();
    m.expect_execute().returning(move |_| {
        let mut lock = queue.lock().unwrap();
        let output = if lock.is_empty() {
            "fallback".into()
        } else {
            lock.drain(..1).next().unwrap()
        };
        Ok(make_response(output, 10))
    });
    m.expect_kind().return_const(default_kind()).times(0..);
    m
}

/// Mock search backend that always succeeds returning `response`.
pub fn mock_search(response: impl Into<String>) -> MockWebSearch {
    let response = response.into();
    let mut m = MockWebSearch::new();
    m.expect_search()
        .returning(move |_, _| Ok(response.clone()));
    m
}

/// Mock WASM backend that always succeeds returning `response`.
pub fn mock_wasm(response: impl Into<String>) -> MockWasmRunner {
    let response = response.into();
    let mut m = MockWasmRunner::new();
    m.expect_execute_script()
        .returning(move |_, _| Ok(response.clone()));
    m
}

/// Mock MCP backend backed by `files` map (`path → content`).
pub fn mock_mcp(files: HashMap<String, String>) -> MockMcpClient {
    let mut m = MockMcpClient::new();
    m.expect_call().returning(move |_, path| {
        files
            .get(path)
            .cloned()
            .ok_or_else(|| ToolError::MalformedInput(format!("path not found: {path}")))
    });
    m
}
