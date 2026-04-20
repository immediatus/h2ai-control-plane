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
        })
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}
