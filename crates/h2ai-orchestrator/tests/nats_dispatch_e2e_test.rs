//! End-to-end integration test for the NATS edge agent dispatch pipeline.
//!
//! Requires a running NATS server (JetStream enabled).
//! Run with:
//!   NATS_URL=nats://localhost:4222 cargo test -p h2ai-orchestrator --test nats_dispatch_e2e_test -- --ignored

use async_trait::async_trait;
use h2ai_agent::dispatch::DispatchLoop;
use h2ai_orchestrator::nats_dispatch_adapter::{NatsDispatchAdapter, NatsDispatchConfig};
use h2ai_provisioner::nats_provider::NatsAgentProvider;
use h2ai_state::NatsClient;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::agent::{AgentDescriptor, CostTier, TaskRequirements};
use h2ai_types::config::AdapterKind;
use h2ai_types::identity::AgentId;
use h2ai_types::physics::TauValue;
use std::sync::{atomic::AtomicU32, Arc};
use std::time::Duration;

/// A minimal compute adapter that returns a non-empty output and a non-zero token cost,
/// allowing the end-to-end assertion on token_cost > 0.
#[derive(Debug)]
struct FixedCostAdapter {
    output: String,
    token_cost: u64,
}

#[async_trait]
impl IComputeAdapter for FixedCostAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Ok(ComputeResponse {
            output: self.output.clone(),
            token_cost: self.token_cost,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: "fixed://localhost".into(),
                api_key_env: "NONE".into(),
            },
        })
    }

    fn kind(&self) -> &AdapterKind {
        // static reference via a thread-local to avoid a stored field lifetime issue
        static KIND: std::sync::OnceLock<AdapterKind> = std::sync::OnceLock::new();
        KIND.get_or_init(|| AdapterKind::CloudGeneric {
            endpoint: "fixed://localhost".into(),
            api_key_env: "NONE".into(),
        })
    }
}

/// Full end-to-end dispatch pipeline test:
///
/// 1. Connect to NATS, call `ensure_infrastructure`
/// 2. Spawn an in-process mock edge agent using `DispatchLoop` + `FixedCostAdapter`
/// 3. Register the mock agent with `NatsAgentProvider` via `inject_registration`
/// 4. Build `NatsDispatchAdapter` and call `adapter.execute(...)`
/// 5. Assert the response is non-empty and has token_cost > 0
#[tokio::test]
#[ignore]
async fn nats_dispatch_e2e_full_pipeline() {
    let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());

    // Step 1: Connect and ensure JetStream infrastructure.
    let nats_orchestrator = Arc::new(
        NatsClient::connect(&nats_url)
            .await
            .expect("connect orchestrator NATS client"),
    );
    nats_orchestrator
        .ensure_infrastructure()
        .await
        .expect("ensure JetStream infrastructure");

    // Step 2: Spawn in-process mock edge agent via DispatchLoop + FixedCostAdapter.
    let agent_id = AgentId::from(uuid::Uuid::new_v4().to_string());
    let agent_nats = async_nats::connect(&nats_url)
        .await
        .expect("connect agent NATS client");

    let mock_adapter: Arc<dyn IComputeAdapter> = Arc::new(FixedCostAdapter {
        output: "e2e pipeline response".into(),
        token_cost: 7,
    });
    let active_tasks = Arc::new(AtomicU32::new(0));

    let dispatch_loop = DispatchLoop::new(agent_nats, agent_id.clone(), mock_adapter, active_tasks);
    tokio::spawn(async move {
        dispatch_loop.run().await.unwrap();
    });

    // Step 3: Register the mock agent with NatsAgentProvider.
    let descriptor = AgentDescriptor {
        model: "e2e-mock-model".into(),
        tools: vec![],
        cost_tier: CostTier::Low,
    };
    let provider = Arc::new(NatsAgentProvider::new_test_only());
    provider.inject_registration(agent_id.clone(), descriptor.clone(), 0);

    // Step 4: Build NatsDispatchAdapter and execute a request.
    let adapter = NatsDispatchAdapter::new(NatsDispatchConfig {
        nats: nats_orchestrator,
        provider,
        agent_descriptor: descriptor,
        task_requirements: TaskRequirements {
            max_cost_tier: CostTier::High,
            required_tools: vec![],
        },
        task_timeout: Duration::from_secs(5),
    });

    let request = ComputeRequest {
        system_context: "integration test context".into(),
        task: "end-to-end pipeline task".into(),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 256,
    };

    let response = adapter
        .execute(request)
        .await
        .expect("NatsDispatchAdapter::execute succeeded");

    // Step 5: Assert non-empty output and token_cost > 0.
    assert!(
        !response.output.is_empty(),
        "response output must be non-empty"
    );
    assert!(
        response.token_cost > 0,
        "response token_cost must be > 0, got {}",
        response.token_cost
    );
}
