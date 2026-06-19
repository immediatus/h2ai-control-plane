use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use h2ai_autonomic::spec_repair::{RepairInput, SpecRepairAdvisor};
use h2ai_config::GapK1Config;
use h2ai_constraints::{
    nats_versioned::NatsVersionedSource, source::InMemorySource, spec::SemanticSpec,
};
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::{AdapterKind, CloudProvider};

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
            output: "candidate rewrite text".to_string(),
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

fn make_spec(id: &str, check: &str) -> SemanticSpec {
    let mut s = SemanticSpec::default_for_test(id);
    s.rubric.checks = vec![check.to_owned()];
    s.rubric.pass = "good proposal text".into();
    s
}

#[tokio::test]
async fn spec_repair_uses_config_repair_max_tokens() {
    let adapter = CapturingAdapter {
        requests: Mutex::new(vec![]),
        kind: cloud_kind(),
    };
    let cfg = GapK1Config {
        repair_max_tokens: 777,
        repair_candidates: 1,
        repair_acceptance_threshold: 0.0,
        ..Default::default()
    };

    let inner = InMemorySource {
        specs: vec![make_spec("C-001", "the check text")],
    };
    let source = Arc::new(NatsVersionedSource::new_in_memory(inner));

    let input = RepairInput {
        task_id: "task-1".to_string(),
        constraint_id: "C-001".to_string(),
        check_index: 0,
        original_check_text: "the check text".to_string(),
        divergent_reasons: vec!["reason A".to_string()],
        should_pass_example: "pass example".to_string(),
        should_prune_example: None,
        current_version: 1,
    };

    let advisor = SpecRepairAdvisor::new(cfg);
    let _ = advisor.run(input, source, &adapter).await;

    let reqs = adapter.requests.lock().unwrap();
    assert!(
        reqs.contains(&777),
        "expected max_tokens=777 from cfg.repair_max_tokens, got: {:?}",
        *reqs
    );
}
