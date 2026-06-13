//! Verify that thinking loop LLM calls use max_tokens from ThinkingLoopConfig,
//! not hardcoded literals.

use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use h2ai_config::ThinkingLoopConfig;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::{AdapterKind, CloudProvider};
use h2ai_orchestrator::thinking_loop::{run, ThinkingLoopInput};

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

        // Return appropriate JSON based on the context
        let output = if req.system_context.contains("archetype") || req.system_context.contains("Archetype") {
            // For archetype selection, return a JSON array of archetypes
            r#"[{"name":"test","persona":"test persona","scope":"test scope","confidence":0.8,"tau":0.3,"model_tier":"standard","cot_style":"step_by_step"}]"#.to_string()
        } else if req.system_context.contains("synthesis") || req.system_context.contains("Synthesis") {
            // For synthesis, return a JSON object with shared_understanding
            r#"{"shared_understanding":"test understanding","tensions":[],"coverage_score":0.9}"#.to_string()
        } else {
            // For gate or other calls, return a simple YES/NO
            "YES".to_string()
        };

        Ok(ComputeResponse {
            output,
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
async fn archetype_select_uses_config_max_tokens() {
    let adapter = Arc::new(CapturingAdapter {
        requests: Mutex::new(vec![]),
        kind: cloud_kind(),
    });
    let mut cfg = ThinkingLoopConfig::default();
    cfg.enabled = true;
    cfg.max_iterations = 1;
    cfg.max_archetypes = 1;
    cfg.archetype_select_max_tokens = 999;

    let input = ThinkingLoopInput {
        task_description: "test task",
        constraint_ids: &[],
        constraint_tags: &[],
        research_context: "",
        knowledge_provider: None,
        n_archetypes: 1,
        cfg: &cfg,
        adapter: adapter.as_ref(),
        embedding_model: None,
        nats_client: None,
        task_id: "test-task-id",
    };

    let _ = run(input).await;

    let reqs = adapter.requests.lock().unwrap();
    assert!(
        reqs.iter().any(|&t| t == 999),
        "expected max_tokens=999 from cfg, got: {:?}",
        *reqs
    );
}

#[tokio::test]
async fn quality_gate_uses_config_max_tokens() {
    let adapter = Arc::new(CapturingAdapter {
        requests: Mutex::new(vec![]),
        kind: cloud_kind(),
    });
    let mut cfg = ThinkingLoopConfig::default();
    cfg.enabled = true;
    cfg.max_iterations = 2;
    cfg.max_archetypes = 1;
    cfg.quality_gate_max_tokens = 77;

    let input = ThinkingLoopInput {
        task_description: "test task",
        constraint_ids: &[],
        constraint_tags: &[],
        research_context: "",
        knowledge_provider: None,
        n_archetypes: 1,
        cfg: &cfg,
        adapter: adapter.as_ref(),
        embedding_model: None,
        nats_client: None,
        task_id: "test-task-id",
    };

    let _ = run(input).await;

    let reqs = adapter.requests.lock().unwrap();
    assert!(
        reqs.iter().any(|&t| t == 77),
        "expected max_tokens=77 from cfg, got: {:?}",
        *reqs
    );
}
