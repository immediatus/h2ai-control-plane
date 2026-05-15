use async_trait::async_trait;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;

#[derive(Debug)]
pub struct MockAdapter {
    output: String,
    kind: AdapterKind,
}

impl MockAdapter {
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

/// A test adapter that automatically returns valid decomposition JSON when called as STEP3
/// (detected by the "JSON formatter" system context), and `fallback_output` for all other calls.
///
/// Use in e2e tests that exercise the full engine pipeline — the mandatory Path C decomposition
/// step requires valid `RawSlot` JSON; a plain `MockAdapter` would cause parse failure and task failure.
#[derive(Debug)]
pub struct DecompositionMockAdapter {
    fallback_output: String,
    kind: AdapterKind,
}

impl DecompositionMockAdapter {
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

/// Minimal valid STEP3 JSON: two generic expert slots the decomposition parser accepts.
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

/// A test adapter that returns responses from a fixed sequence, one per call.
/// When the sequence is exhausted it returns `"fallback"`.
#[derive(Debug)]
pub struct SequencedMockAdapter {
    responses: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    kind: AdapterKind,
}

impl SequencedMockAdapter {
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
