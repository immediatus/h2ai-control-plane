#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
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
use h2ai_types::sizing::TauValue;
use std::sync::{Arc, Mutex};

// --- Mock declarations ---

mockall::mock! {
    pub PipelineMemory {}
    #[async_trait::async_trait]
    impl MemoryProvider for PipelineMemory {
        async fn get_recent_history(&self, session_id: &str, limit: usize) -> Result<Vec<serde_json::Value>, MemoryError>;
        async fn commit_new_memories(&self, session_id: &str, memories: Vec<serde_json::Value>) -> Result<(), MemoryError>;
        async fn retrieve_relevant_context(&self, session_id: &str, query: &str) -> Result<Vec<String>, MemoryError>;
    }
}

mockall::mock! {
    pub PipelineProvisioner {}
    #[async_trait::async_trait]
    impl AgentProvider for PipelineProvisioner {
        async fn ensure_agent_capacity(&self, descriptor: &AgentDescriptor, task_load: usize) -> Result<(), ProvisionError>;
        async fn terminate_agent(&self, agent_id: &AgentId) -> Result<(), ProvisionError>;
        async fn select_agent(&self, requirements: &TaskRequirements) -> Result<AgentId, ProvisionError>;
    }
}

mockall::mock! {
    pub PipelineAuditor {}
    #[async_trait::async_trait]
    impl AuditProvider for PipelineAuditor {
        async fn record_event(&self, event: AgentTelemetryEvent) -> Result<(), AuditError>;
        async fn flush(&self) -> Result<(), AuditError>;
    }
}

// --- Factory helpers ---

fn make_memory(store: Arc<Mutex<Vec<serde_json::Value>>>) -> MockPipelineMemory {
    let commit_store = store.clone();
    let history_store = store.clone();
    let mut m = MockPipelineMemory::new();
    m.expect_commit_new_memories()
        .returning(move |_, memories| {
            commit_store.lock().unwrap().extend(memories);
            Ok(())
        });
    m.expect_get_recent_history().returning(move |_, limit| {
        let entries = history_store.lock().unwrap();
        Ok(entries.iter().rev().take(limit).cloned().collect())
    });
    m.expect_retrieve_relevant_context()
        .returning(|_, _| Ok(vec![]));
    m
}

fn make_provisioner_no_agents() -> MockPipelineProvisioner {
    let mut m = MockPipelineProvisioner::new();
    m.expect_ensure_agent_capacity().returning(|_, _| Ok(()));
    m.expect_terminate_agent().returning(|_| Ok(()));
    m.expect_select_agent().returning(|requirements| {
        Err(ProvisionError::NoAgentsAvailable {
            max_tier: requirements.max_cost_tier.clone(),
            tools: requirements.required_tools.clone(),
        })
    });
    m
}

fn make_auditor(events: Arc<Mutex<Vec<AgentTelemetryEvent>>>) -> MockPipelineAuditor {
    let record_events = events.clone();
    let mut m = MockPipelineAuditor::new();
    m.expect_record_event().returning(move |event| {
        record_events.lock().unwrap().push(event);
        Ok(())
    });
    m.expect_flush().returning(|| Ok(()));
    m
}

async fn build_pipeline(
) -> Option<OrchestratorPipeline<MockPipelineMemory, MockPipelineProvisioner, MockPipelineAuditor>>
{
    let nats_url = h2ai_config::H2AIConfig::default().nats_url;
    let nats = match async_nats::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return None;
        }
    };
    let memory_store = Arc::new(Mutex::new(vec![]));
    let events_store = Arc::new(Mutex::new(vec![]));
    Some(OrchestratorPipeline::new(
        make_memory(memory_store),
        make_provisioner_no_agents(),
        make_auditor(events_store),
        nats,
    ))
}

#[tokio::test]
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
async fn pipeline_finalize_commits_to_memory() {
    let memory_store = Arc::new(Mutex::new(vec![]));
    let nats_url = h2ai_config::H2AIConfig::default().nats_url;
    let nats = match async_nats::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return;
        }
    };
    let events_store = Arc::new(Mutex::new(vec![]));
    let pipeline = OrchestratorPipeline::new(
        make_memory(memory_store.clone()),
        make_provisioner_no_agents(),
        make_auditor(events_store),
        nats,
    );
    let result = TaskResult {
        task_id: TaskId::new(),
        agent_id: AgentId::from("agent-1"),
        output: "The answer is 42.".into(),
        token_cost: 100,
        error: None,
        tool_calls: vec![],
    };
    pipeline.finalize("session-1", &result).await.unwrap();
    assert!(!memory_store.lock().unwrap().is_empty());
}

/// Build a pipeline without live NATS by connecting to NATS and returning None if
/// unavailable. Callers skip the test body when None is returned.
async fn build_pipeline_with_shared_memory(
    memory: Arc<Mutex<Vec<serde_json::Value>>>,
    events: Arc<Mutex<Vec<AgentTelemetryEvent>>>,
) -> Option<OrchestratorPipeline<MockPipelineMemory, MockPipelineProvisioner, MockPipelineAuditor>>
{
    let nats_url = h2ai_config::H2AIConfig::default().nats_url;
    let nats = match async_nats::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return None;
        }
    };
    Some(OrchestratorPipeline::new(
        make_memory(memory),
        make_provisioner_no_agents(),
        make_auditor(events),
        nats,
    ))
}

#[tokio::test]
async fn pipeline_finalize_without_nats_usage() {
    // finalize() only calls memory.commit_new_memories + auditor.flush — no NATS I/O.
    // We still need a NatsClient for construction; skip if NATS is unavailable.
    let memory = Arc::new(Mutex::new(vec![]));
    let events = Arc::new(Mutex::new(vec![]));
    let Some(pipeline) =
        build_pipeline_with_shared_memory(Arc::clone(&memory), Arc::clone(&events)).await
    else {
        return;
    };

    let result = TaskResult {
        task_id: TaskId::new(),
        agent_id: AgentId::from("agent-x"),
        output: "finalized output".into(),
        token_cost: 42,
        error: None,
        tool_calls: vec![],
    };

    pipeline.finalize("session-fin", &result).await.unwrap();

    let mem = memory.lock().unwrap();
    assert_eq!(
        mem.len(),
        1,
        "finalize must commit exactly one memory entry"
    );
    let entry = &mem[0];
    assert_eq!(entry["role"], "assistant");
    assert_eq!(entry["content"], "finalized output");
    assert_eq!(entry["token_cost"], 42_u64);
}

#[tokio::test]
async fn pipeline_record_telemetry_stores_event() {
    // record_telemetry() redacts and records via auditor — no NATS I/O.
    let memory = Arc::new(Mutex::new(vec![]));
    let events = Arc::new(Mutex::new(vec![]));
    let Some(pipeline) =
        build_pipeline_with_shared_memory(Arc::clone(&memory), Arc::clone(&events)).await
    else {
        return;
    };

    let event = AgentTelemetryEvent::LlmResponseReceived {
        task_id: TaskId::new(),
        agent_id: AgentId::from("agent-tel"),
        response: "working on task".into(),
        token_cost: 5,
        timestamp: chrono::Utc::now(),
    };

    pipeline.record_telemetry(event).await.unwrap();

    let recorded = events.lock().unwrap();
    assert_eq!(recorded.len(), 1, "one telemetry event must be recorded");
    assert!(matches!(
        recorded[0],
        AgentTelemetryEvent::LlmResponseReceived { .. }
    ));
}

#[tokio::test]
async fn pipeline_finalize_multiple_results_accumulate() {
    // Calling finalize twice should commit two entries.
    let memory = Arc::new(Mutex::new(vec![]));
    let events = Arc::new(Mutex::new(vec![]));
    let Some(pipeline) =
        build_pipeline_with_shared_memory(Arc::clone(&memory), Arc::clone(&events)).await
    else {
        return;
    };

    for i in 0u64..2 {
        let result = TaskResult {
            task_id: TaskId::new(),
            agent_id: AgentId::from("agent-multi"),
            output: format!("output-{i}"),
            token_cost: i * 10,
            error: None,
            tool_calls: vec![],
        };
        pipeline.finalize("session-multi", &result).await.unwrap();
    }

    let mem = memory.lock().unwrap();
    assert_eq!(
        mem.len(),
        2,
        "two finalizations must produce two memory entries"
    );
    assert_eq!(mem[0]["content"], "output-0");
    assert_eq!(mem[1]["content"], "output-1");
}

#[test]
fn orchestrator_error_display() {
    let err = h2ai_orchestrator::error::OrchestratorError::Timeout {
        task_id: "task-1".into(),
    };
    assert!(err.to_string().contains("timeout"));
}
