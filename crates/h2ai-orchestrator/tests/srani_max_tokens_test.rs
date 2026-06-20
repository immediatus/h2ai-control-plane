//! Verify that SRANI LLM calls use max_tokens from SraniConfig.

use async_trait::async_trait;
use h2ai_config::SraniConfig;
use h2ai_orchestrator::srani_grounding::{
    GroundingContext, GroundingProvider, LlmResearcherGrounder,
};
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::{AdapterKind, CloudProvider};
use std::sync::{Arc, Mutex};

fn cloud_kind() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "http://test".into(),
        api_key_env: "TEST".into(),
        model: None,
        provider: CloudProvider::default(),
    }
}

struct CapturingAdapter {
    requests: Mutex<Vec<u64>>,
    kind: AdapterKind,
}

impl std::fmt::Debug for CapturingAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturingAdapter").finish()
    }
}

#[async_trait]
impl IComputeAdapter for CapturingAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        self.requests.lock().unwrap().push(req.max_tokens);
        Ok(ComputeResponse {
            output: r#"{"alternatives":[],"statement":""}"#.to_string(),
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

#[tokio::test]
async fn llm_researcher_grounder_uses_config_max_tokens() {
    let adapter = Arc::new(CapturingAdapter {
        requests: Mutex::new(vec![]),
        kind: cloud_kind(),
    });
    let srani_cfg = SraniConfig {
        researcher_max_tokens: 888,
        ..Default::default()
    };

    let grounder = LlmResearcherGrounder::new(adapter.clone(), srani_cfg.researcher_max_tokens);
    let ctx = GroundingContext {
        fabricated_entities: vec!["entity".to_string()],
        task_description: "test".to_string(),
        spec_technologies: vec![],
    };
    let _ = grounder.ground(&ctx).await;

    let reqs = adapter.requests.lock().unwrap();
    assert!(
        reqs.contains(&888),
        "expected max_tokens=888, got: {:?}",
        *reqs
    );
}
