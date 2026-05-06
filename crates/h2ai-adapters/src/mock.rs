use async_trait::async_trait;
use h2ai_types::adapter::{
    AdapterError, AdapterFamily, ComputeRequest, ComputeResponse, IComputeAdapter,
};
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
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }

    fn family(&self) -> AdapterFamily {
        AdapterFamily::Mock
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
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }

    fn family(&self) -> AdapterFamily {
        AdapterFamily::Mock
    }
}
