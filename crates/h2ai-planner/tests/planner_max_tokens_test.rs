use std::sync::Mutex;

use async_trait::async_trait;
use h2ai_planner::decomposer::PlanningEngine;
use h2ai_planner::reviewer::PlanReviewer;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::{AdapterKind, CloudProvider};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use h2ai_types::config::ParetoWeights;
use h2ai_types::sizing::TauValue;

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
    response: String,
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
            output: self.response.clone(),
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

fn test_manifest() -> TaskManifest {
    TaskManifest {
        description: "test task".to_string(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
        topology: TopologyRequest { kind: "auto".into(), branching_factor: None },
        explorers: ExplorerRequest {
            count: 1,
            tau_min: None,
            tau_max: None,
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![],
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: Default::default(),
    }
}

#[tokio::test]
async fn decompose_uses_provided_max_tokens() {
    let adapter = CapturingAdapter {
        requests: Mutex::new(vec![]),
        kind: cloud_kind(),
        response: r#"{"subtasks":[{"description":"step 1","depends_on":[],"role_hint":null}]}"#
            .to_string(),
    };
    let tau = TauValue::new(0.3).unwrap();

    let _ = PlanningEngine::decompose(&test_manifest(), &adapter, tau, 555).await;

    let reqs = adapter.requests.lock().unwrap();
    assert_eq!(reqs[0], 555, "decompose must use provided max_tokens");
}

#[tokio::test]
async fn evaluate_uses_provided_max_tokens() {
    let decompose_adapter = CapturingAdapter {
        requests: Mutex::new(vec![]),
        kind: cloud_kind(),
        response: r#"{"subtasks":[{"description":"step 1","depends_on":[],"role_hint":null}]}"#
            .to_string(),
    };
    let tau = TauValue::new(0.3).unwrap();
    let plan = PlanningEngine::decompose(&test_manifest(), &decompose_adapter, tau, 1024)
        .await
        .unwrap();

    let review_adapter = CapturingAdapter {
        requests: Mutex::new(vec![]),
        kind: cloud_kind(),
        response: r#"{"approved":true,"reason":"looks good"}"#.to_string(),
    };
    let _ = PlanReviewer::evaluate(&plan, "desc", &review_adapter, tau, 333).await;

    let reqs = review_adapter.requests.lock().unwrap();
    assert_eq!(reqs[0], 333, "evaluate must use provided max_tokens");
}
