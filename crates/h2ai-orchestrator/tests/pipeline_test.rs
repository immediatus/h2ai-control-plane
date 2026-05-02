use async_trait::async_trait;
use h2ai_memory::error::MemoryError;
use h2ai_memory::provider::MemoryProvider;
use h2ai_orchestrator::pipeline::OrchestratorPipeline;
use h2ai_provisioner::error::ProvisionError;
use h2ai_provisioner::provider::AgentProvider;
use h2ai_telemetry::error::AuditError;
use h2ai_telemetry::provider::AuditProvider;
use h2ai_types::agent::TaskRequirements;
use h2ai_types::agent::{AgentDescriptor, AgentTelemetryEvent, AgentTool, CostTier, TaskResult};
use h2ai_types::identity::{AgentId, TaskId};
use h2ai_types::physics::TauValue;
use std::sync::{Arc, Mutex};

// --- Mocks ---

struct MockMemory(Arc<Mutex<Vec<serde_json::Value>>>);

#[async_trait]
impl MemoryProvider for MockMemory {
    async fn get_recent_history(
        &self,
        _s: &str,
        limit: usize,
    ) -> Result<Vec<serde_json::Value>, MemoryError> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect())
    }
    async fn commit_new_memories(
        &self,
        _s: &str,
        m: Vec<serde_json::Value>,
    ) -> Result<(), MemoryError> {
        self.0.lock().unwrap().extend(m);
        Ok(())
    }
    async fn retrieve_relevant_context(
        &self,
        _s: &str,
        _q: &str,
    ) -> Result<Vec<String>, MemoryError> {
        Ok(vec![])
    }
}

struct MockProvisioner;

#[async_trait]
impl AgentProvider for MockProvisioner {
    async fn ensure_agent_capacity(
        &self,
        _d: &AgentDescriptor,
        _l: usize,
    ) -> Result<(), ProvisionError> {
        Ok(())
    }
    async fn terminate_agent(&self, _id: &AgentId) -> Result<(), ProvisionError> {
        Ok(())
    }

    async fn select_agent(
        &self,
        requirements: &TaskRequirements,
    ) -> Result<AgentId, ProvisionError> {
        Err(ProvisionError::NoAgentsAvailable {
            max_tier: requirements.max_cost_tier.clone(),
            tools: requirements.required_tools.clone(),
        })
    }
}

struct MockAuditor(Arc<Mutex<Vec<AgentTelemetryEvent>>>);

#[async_trait]
impl AuditProvider for MockAuditor {
    async fn record_event(&self, event: AgentTelemetryEvent) -> Result<(), AuditError> {
        self.0.lock().unwrap().push(event);
        Ok(())
    }
    async fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }
}

async fn build_pipeline() -> Option<OrchestratorPipeline<MockMemory, MockProvisioner, MockAuditor>>
{
    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
    let nats = match async_nats::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return None;
        }
    };
    Some(OrchestratorPipeline::new(
        MockMemory(Arc::new(Mutex::new(vec![]))),
        MockProvisioner,
        MockAuditor(Arc::new(Mutex::new(vec![]))),
        nats,
    ))
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn pipeline_execute_dispatches_task() {
    let Some(pipeline) = build_pipeline().await else {
        return;
    };
    let agent = AgentDescriptor {
        model: "gpt-4o".into(),
        tools: vec![AgentTool::Shell],
        cost_tier: CostTier::Mid,
    };
    let task_id = pipeline
        .execute(
            "session-1",
            "summarize the doc",
            agent,
            TauValue::new(0.4).unwrap(),
            512,
        )
        .await
        .unwrap();
    assert!(!task_id.to_string().is_empty());
}

#[tokio::test]
#[ignore = "requires live NATS at localhost:4222"]
async fn pipeline_finalize_commits_to_memory() {
    let memory = Arc::new(Mutex::new(vec![]));
    let nats_url =
        std::env::var("NATS_URL").unwrap_or_else(|_| h2ai_config::H2AIConfig::default().nats_url);
    let nats = match async_nats::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return;
        }
    };
    let pipeline = OrchestratorPipeline::new(
        MockMemory(memory.clone()),
        MockProvisioner,
        MockAuditor(Arc::new(Mutex::new(vec![]))),
        nats,
    );
    let result = TaskResult {
        task_id: TaskId::new(),
        agent_id: AgentId::from("agent-1"),
        output: "The answer is 42.".into(),
        token_cost: 100,
        error: None,
    };
    pipeline.finalize("session-1", &result).await.unwrap();
    assert!(!memory.lock().unwrap().is_empty());
}

#[test]
fn orchestrator_error_display() {
    let err = h2ai_orchestrator::error::OrchestratorError::Timeout {
        task_id: "task-1".into(),
    };
    assert!(err.to_string().contains("timeout"));
}
