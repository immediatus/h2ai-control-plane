//! Test helpers and `mockall`-generated mocks for the H2AI workspace.
//!
//! This crate is `cfg(test)` / dev-dependency only — it is never linked into
//! production binaries. It provides:
//!
//! - **`MockIComputeAdapter`** — deterministic LLM stub; use [`mock_adapter`]
//!   for the common case of a single fixed response.
//! - **`MockNatsBackend`** — in-memory mock of all `NatsBackend` supertrait
//!   methods; supports expectation-based assertions via `mockall`.
//! - **`MockTaskDispatchBackend`** — mock for task dispatch over NATS; use
//!   [`stub_topology_retry_event`] to inject a pre-built retry event.
//! - Tool mocks: `MockWebSearch`, `MockWasmRunner`, `MockMcpClient` — stubs for
//!   the three external-tool backends used by edge agents.
//! - **`mock_adapter`** / **`sequenced_adapter`** — convenience constructors for
//!   `MockIComputeAdapter` with one fixed response or a sequence of responses.
//!
//! ## Usage pattern
//!
//! ```rust,ignore
//! use h2ai_test_utils::{mock_adapter, MockNatsBackend};
//! let adapter = Arc::new(mock_adapter("my fixed output"));
//! let mut nats = MockNatsBackend::new();
//! nats.expect_publish_event().returning(|_, _| Ok(()));
//! ```

use async_trait::async_trait;
use h2ai_tools::error::ToolError;
use h2ai_tools::mcp::McpBackend;
use h2ai_tools::wasm::WasmBackend;
use h2ai_tools::web_search::WebSearchBackend;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

fn default_kind() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: "mock://localhost".into(),
        api_key_env: "MOCK".into(),
        model: None,
        provider: Default::default(),
    }
}

fn make_response(output: impl Into<String>, cost: u64) -> ComputeResponse {
    ComputeResponse {
        output: output.into(),
        token_cost: cost,
        adapter_kind: default_kind(),
        tokens_used: None,
        reasoning_trace: None,
    }
}

// ── Mock declarations ─────────────────────────────────────────────────────────

mockall::mock! {
    #[derive(Debug)]
    pub IComputeAdapter {}

    #[async_trait]
    impl IComputeAdapter for IComputeAdapter {
        async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError>;
        fn kind(&self) -> &AdapterKind;
    }
}

mockall::mock! {
    pub WebSearch {}

    #[async_trait]
    impl WebSearchBackend for WebSearch {
        async fn search(&self, query: &str, max_results: usize) -> Result<String, ToolError>;
    }
}

mockall::mock! {
    pub WasmRunner {}

    #[async_trait]
    impl WasmBackend for WasmRunner {
        async fn execute_script(&self, language: &str, script: &str) -> Result<String, ToolError>;
    }
}

mockall::mock! {
    pub McpClient {}

    #[async_trait]
    impl McpBackend for McpClient {
        async fn call(&self, op: &str, path: &str) -> Result<String, ToolError>;
    }
}

// ── Factory helpers ───────────────────────────────────────────────────────────

/// Mock adapter that always succeeds returning `output`.
pub fn mock_adapter(output: impl Into<String>) -> MockIComputeAdapter {
    let output = output.into();
    let mut m = MockIComputeAdapter::new();
    m.expect_execute()
        .returning(move |_| Ok(make_response(output.clone(), 0)));
    m.expect_kind().return_const(default_kind()).times(0..);
    m
}

/// Mock adapter that records every `ComputeRequest` it receives and always succeeds.
///
/// Returns the shared `Arc<Mutex<Vec<ComputeRequest>>>` so callers can inspect
/// what fields (e.g. `max_tokens`) were passed in each call.
pub fn capturing_adapter(
    output: impl Into<String>,
) -> (MockIComputeAdapter, Arc<Mutex<Vec<ComputeRequest>>>) {
    let output = output.into();
    let captured: Arc<Mutex<Vec<ComputeRequest>>> = Arc::new(Mutex::new(Vec::new()));
    let captured2 = Arc::clone(&captured);
    let mut m = MockIComputeAdapter::new();
    m.expect_execute().returning(move |req| {
        captured2.lock().unwrap().push(req);
        Ok(make_response(output.clone(), 0))
    });
    m.expect_kind().return_const(default_kind()).times(0..);
    (m, captured)
}

/// Mock adapter that always fails with `AdapterError::NetworkError`.
pub fn failing_adapter() -> MockIComputeAdapter {
    let mut m = MockIComputeAdapter::new();
    m.expect_execute()
        .returning(|_| Err(AdapterError::NetworkError("mock network failure".into())));
    m.expect_kind().return_const(default_kind()).times(0..);
    m
}

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

/// Mock adapter for the decomposition pipeline. Returns STEP3 JSON when
/// `system_context` contains `"JSON formatter"`, `fallback` otherwise.
pub fn decomposition_adapter(fallback: impl Into<String>) -> MockIComputeAdapter {
    let fallback = fallback.into();
    let mut m = MockIComputeAdapter::new();
    m.expect_execute().returning(move |req| {
        let output = if req.system_context.contains("JSON formatter") {
            STEP3_MOCK_JSON.to_string()
        } else {
            fallback.clone()
        };
        Ok(make_response(output, 10))
    });
    m.expect_kind().return_const(default_kind()).times(0..);
    m
}

/// Mock adapter returning `responses` in sequence; returns `"fallback"` when exhausted.
pub fn sequenced_adapter(responses: Vec<String>) -> MockIComputeAdapter {
    let queue = Arc::new(Mutex::new(responses));
    let mut m = MockIComputeAdapter::new();
    m.expect_execute().returning(move |_| {
        let mut lock = queue.lock().unwrap();
        let output = if lock.is_empty() {
            "fallback".into()
        } else {
            lock.drain(..1).next().unwrap()
        };
        Ok(make_response(output, 10))
    });
    m.expect_kind().return_const(default_kind()).times(0..);
    m
}

/// Mock search backend that always succeeds returning `response`.
pub fn mock_search(response: impl Into<String>) -> MockWebSearch {
    let response = response.into();
    let mut m = MockWebSearch::new();
    m.expect_search()
        .returning(move |_, _| Ok(response.clone()));
    m
}

/// Mock WASM backend that always succeeds returning `response`.
pub fn mock_wasm(response: impl Into<String>) -> MockWasmRunner {
    let response = response.into();
    let mut m = MockWasmRunner::new();
    m.expect_execute_script()
        .returning(move |_, _| Ok(response.clone()));
    m
}

/// Mock MCP backend backed by `files` map (`path → content`).
pub fn mock_mcp(files: HashMap<String, String>) -> MockMcpClient {
    let mut m = MockMcpClient::new();
    m.expect_call().returning(move |_, path| {
        files
            .get(path)
            .cloned()
            .ok_or_else(|| ToolError::MalformedInput(format!("path not found: {path}")))
    });
    m
}

// ── MockNatsBackend ──────────────────────────────────────────────────────────

use futures::stream::BoxStream;

type TaskEventStream = BoxStream<'static, Result<(u64, H2AIEvent), NatsError>>;
use h2ai_state::backend::{
    CalibrationStore, ConflictStore, EstimatorStore, EventPublisher, OproStore, ReasoningStore,
    ShadowDomainStore, SignalPublisher, SignalSubscriber, SkillStore, SnapshotStore, TailEvents,
    TaskCheckpointStore, TaskDispatchBackend,
};
use h2ai_state::nats::NatsError;
use h2ai_types::agent::{TaskPayload, TaskResult};
use h2ai_types::calibration::CalibrationRecord;
use h2ai_types::checkpoint::TaskCheckpoint;
use h2ai_types::conflict::ConflictRateAccumulator;
use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent, TaskSnapshot};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::prompt_variant::{AdapterOproState, PromptVariant};
use h2ai_types::reasoning_checkpoint::{TaskMetaState, TaskReasoningCheckpoint};
use h2ai_types::signal::ResumeSignal;
use std::collections::HashSet;

mockall::mock! {
    pub NatsBackend {}

    #[async_trait::async_trait]
    impl EventPublisher for NatsBackend {
        async fn publish_event(&self, task_id: &TaskId, event: &H2AIEvent) -> Result<(), NatsError>;
        async fn publish_to(&self, subject: &str, event: &H2AIEvent) -> Result<(), NatsError>;
        async fn publish_event_seq(&self, task_id: &TaskId, event: &H2AIEvent) -> Result<u64, NatsError>;
    }

    #[async_trait::async_trait]
    impl SnapshotStore for NatsBackend {
        async fn put_snapshot(&self, snap: &TaskSnapshot) -> Result<(), NatsError>;
        async fn get_snapshot(&self, task_id: &TaskId) -> Result<Option<TaskSnapshot>, NatsError>;
    }

    #[async_trait::async_trait]
    impl CalibrationStore for NatsBackend {
        async fn put_calibration(&self, cal: &CalibrationCompletedEvent) -> Result<(), NatsError>;
        async fn get_calibration(&self) -> Result<Option<CalibrationCompletedEvent>, NatsError>;
        async fn get_calibration_record(&self, adapter_profile: &str) -> Result<Option<CalibrationRecord>, NatsError>;
        async fn put_calibration_record(&self, record: &CalibrationRecord) -> Result<(), NatsError>;
    }

    #[async_trait::async_trait]
    impl TailEvents for NatsBackend {
        async fn tail_task_events_boxed(
            &self,
            task_id: &TaskId,
            from_seq: u64,
        ) -> Result<TaskEventStream, NatsError>;
    }

    #[async_trait::async_trait]
    impl SignalPublisher for NatsBackend {
        async fn publish_signal(&self, signal: &ResumeSignal) -> Result<(), NatsError>;
    }

    #[async_trait::async_trait]
    impl SignalSubscriber for NatsBackend {
        async fn subscribe_signals(
            &self,
            task_id: &TaskId,
            tenant_id: &TenantId,
        ) -> Result<BoxStream<'static, Result<ResumeSignal, NatsError>>, NatsError>;
        async fn delete_signal_consumer(&self, task_id: &TaskId) -> Result<(), NatsError>;
    }

    #[async_trait::async_trait]
    impl OproStore for NatsBackend {
        async fn put_prompt_variant(&self, variant: &PromptVariant) -> Result<(), NatsError>;
        async fn get_prompt_variant(&self, adapter_name: &str, prompt_key: &str, variant_id: &str) -> Result<Option<PromptVariant>, NatsError>;
        async fn get_active_variant_ptr(&self, adapter_name: &str, prompt_key: &str) -> Result<Option<String>, NatsError>;
        async fn set_active_variant_ptr(&self, adapter_name: &str, prompt_key: &str, variant_id: &str) -> Result<(), NatsError>;
        async fn get_adapter_opro_state(&self, adapter_name: &str) -> Result<Option<AdapterOproState>, NatsError>;
        async fn put_adapter_opro_state(&self, state: &AdapterOproState) -> Result<(), NatsError>;
    }

    #[async_trait::async_trait]
    impl EstimatorStore for NatsBackend {
        async fn get_tao_estimator_state(&self, tenant_id: &TenantId) -> Result<Option<(f64, usize)>, NatsError>;
        async fn put_tao_estimator_state(&self, tenant_id: &TenantId, ema: f64, count: usize) -> Result<(), NatsError>;
        async fn get_bandit_state(&self, tenant_id: &TenantId) -> Result<Option<Vec<u8>>, NatsError>;
        async fn put_bandit_state(&self, tenant_id: &TenantId, json_bytes: Vec<u8>) -> Result<(), NatsError>;
    }

    #[async_trait::async_trait]
    impl ReasoningStore for NatsBackend {
        async fn ensure_reasoning_buckets(&self, tenant_id: &TenantId, checkpoint_prefix: &str, meta_state_prefix: &str) -> Result<(), NatsError>;
        async fn put_reasoning_checkpoint(&self, checkpoint: &TaskReasoningCheckpoint, checkpoint_prefix: &str) -> Result<(), NatsError>;
        async fn get_reasoning_checkpoint(&self, task_id: &TaskId, tenant_id: &TenantId, checkpoint_prefix: &str) -> Result<Option<TaskReasoningCheckpoint>, NatsError>;
        async fn put_task_meta_state(&self, meta: &TaskMetaState, meta_state_prefix: &str) -> Result<(), NatsError>;
        async fn get_task_meta_state(&self, task_id: &TaskId, tenant_id: &TenantId, meta_state_prefix: &str) -> Result<Option<TaskMetaState>, NatsError>;
        async fn list_task_meta_states(&self, tenant_id: &TenantId, meta_state_prefix: &str, limit: usize) -> Vec<TaskMetaState>;
    }

    #[async_trait::async_trait]
    impl ConflictStore for NatsBackend {
        async fn ensure_conflict_bucket(&self, tenant_id: &TenantId, bucket_prefix: &str) -> Result<(), NatsError>;
        async fn get_conflict_accumulator(&self, tenant_id: &TenantId, bucket_prefix: &str) -> Result<Option<ConflictRateAccumulator>, NatsError>;
        async fn put_conflict_accumulator(&self, acc: &ConflictRateAccumulator, bucket_prefix: &str) -> Result<(), NatsError>;
    }

    #[async_trait::async_trait]
    impl ShadowDomainStore for NatsBackend {
        async fn put_shadow_promoted_domains(&self, domains: &HashSet<String>) -> Result<(), NatsError>;
        async fn get_shadow_promoted_domains(&self) -> Result<HashSet<String>, NatsError>;
    }

    #[async_trait::async_trait]
    impl TaskCheckpointStore for NatsBackend {
        async fn list_task_checkpoints(&self) -> Vec<TaskCheckpoint>;
        async fn put_task_checkpoint(&self, cp: &TaskCheckpoint, expected_revision: Option<u64>) -> Result<u64, NatsError>;
        async fn get_task_checkpoint(&self, task_id: &str) -> Result<Option<TaskCheckpoint>, NatsError>;
        async fn delete_task_checkpoint(&self, task_id: &str) -> Result<(), NatsError>;
    }

    #[async_trait::async_trait]
    impl SkillStore for NatsBackend {
        async fn put_skill_nodes(&self, tenant_id: &TenantId, json_bytes: Vec<u8>) -> Result<(), NatsError>;
        async fn get_skill_nodes(&self, tenant_id: &TenantId) -> Result<Vec<u8>, NatsError>;
    }
}

// ── MockTaskDispatchBackend ──────────────────────────────────────────────────

mockall::mock! {
    pub TaskDispatchBackend {}

    #[async_trait::async_trait]
    impl TaskDispatchBackend for TaskDispatchBackend {
        async fn publish_task_payload(&self, payload: &TaskPayload) -> Result<(), NatsError>;
        async fn await_task_result_once(
            &self,
            task_id: &TaskId,
            timeout: std::time::Duration,
        ) -> Result<TaskResult, NatsError>;
    }
}

// ── Stage runner mocks ────────────────────────────────────────────────────────

use h2ai_orchestrator::decomposition::DecompositionError;
use h2ai_orchestrator::engine::{EngineError, EngineOutput, EngineRunContext};
use h2ai_orchestrator::task_runner::{
    Decomposer, DecompositionArgs, EngineRunner, OwnedEngineInput, ThinkingLoopArgs,
    ThinkingLoopRunner,
};
use h2ai_types::manifest::ExplorerSlotConfig;
use h2ai_types::thinking::ThinkingReport;

mockall::mock! {
    pub ThinkingLoopRunner {}

    #[async_trait::async_trait]
    impl ThinkingLoopRunner for ThinkingLoopRunner {
        async fn run(&self, args: ThinkingLoopArgs) -> ThinkingReport;
    }
}

mockall::mock! {
    pub Decomposer {}

    #[async_trait::async_trait]
    impl Decomposer for Decomposer {
        async fn decompose(&self, args: DecompositionArgs) -> Result<Vec<ExplorerSlotConfig>, DecompositionError>;
    }
}

mod _engine_runner_mock {
    #![allow(clippy::result_large_err)]
    use super::*;
    mockall::mock! {
        pub EngineRunner {}

        #[async_trait::async_trait]
        impl EngineRunner for EngineRunner {
            async fn run(&self, input: OwnedEngineInput) -> Result<EngineOutput, (EngineError, EngineRunContext)>;
        }
    }
}
pub use _engine_runner_mock::MockEngineRunner;

pub fn stub_thinking_report() -> ThinkingReport {
    ThinkingReport {
        shared_understanding: "stub understanding".into(),
        coverage_score: 0.9,
        iteration: 1,
        ..Default::default()
    }
}

/// Build a minimal `TopologyProvisionedEvent` with `retry_count` set.
/// Only `retry_count` and `constraint_tombstone` affect `skill_from_output` —
/// all structural fields are empty/default.
pub fn stub_topology_retry_event(
    task_id: h2ai_types::identity::TaskId,
    retry_count: u32,
    constraint_tombstone: Option<String>,
) -> h2ai_types::events::TopologyProvisionedEvent {
    use h2ai_types::config::{AuditorConfig, TopologyKind};
    use h2ai_types::events::TopologyProvisionedEvent;
    use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold, MergeStrategy};
    let cc = CoherencyCoefficients {
        alpha: 0.1,
        beta_base: 0.01,
        beta_quality: None,
        cg_samples: vec![0.5],
        sample_timestamps: vec![],
    };
    TopologyProvisionedEvent {
        task_id,
        topology_kind: TopologyKind::Ensemble,
        explorer_configs: vec![],
        auditor_config: AuditorConfig::default(),
        n_max: 2.0,
        interface_n_max: None,
        beta_eff: 0.03,
        role_error_costs: vec![],
        merge_strategy: MergeStrategy::ScoreOrdered,
        coordination_threshold: CoordinationThreshold::from_calibration(&cc, 1.0),
        review_gates: vec![],
        retry_count,
        timestamp: chrono::Utc::now(),
        constraint_tombstone,
    }
}

pub fn stub_engine_output(task_id: h2ai_types::identity::TaskId) -> EngineOutput {
    use h2ai_orchestrator::attribution::HarnessAttribution;
    use h2ai_orchestrator::coherence::CoherenceState;
    use h2ai_types::events::{SelectionResolvedEvent, TaskComplexityAssessedEvent};
    use h2ai_types::sizing::{MergeStrategy, ProbeSkipReason, TaskQuadrant};

    let resolved = SelectionResolvedEvent {
        task_id: task_id.clone(),
        valid_proposals: vec![],
        pruned_proposals: vec![],
        merge_strategy: MergeStrategy::ScoreOrdered,
        timestamp: chrono::Utc::now(),
        merge_elapsed_secs: None,
        n_input_proposals: 0,
        n_failed_proposals: 0,
        merge_selection_mode: None,
    };
    let complexity = TaskComplexityAssessedEvent {
        task_id: task_id.clone(),
        tcc_structural: 0.5,
        tcc_empirical: None,
        tcc_effective: 0.5,
        n_eff_pool: None,
        task_quadrant: TaskQuadrant::Precision,
        probe_skipped: true,
        probe_skip_reason: ProbeSkipReason::None,
        heavy_fraction: 0.0,
        tcc_mismatch: false,
        probe_cost_tokens: 0,
        n_informative_static: 0,
        timestamp: chrono::Utc::now(),
    };
    let attribution = HarnessAttribution {
        baseline_quality: 0.7,
        topology_gain: 0.1,
        verification_gain: 0.0,
        tao_gain: 0.0,
        q_confidence: 0.8,
        prediction_basis: h2ai_types::sizing::PredictionBasis::Heuristic,
        q_measured: None,
        rho_adjusted: 0.7,
        case_b_flag: false,
        synthesis_gain: 0.0,
    };
    EngineOutput {
        task_id,
        resolved_output: "stub output".into(),
        selection_resolved: resolved,
        attribution,
        attribution_interval: None,
        verification_events: vec![],
        failed_proposals: vec![],
        pruned_events: vec![],
        talagrand: None,
        suggested_next_params: None,
        waste_ratio: 0.0,
        applied_optimizations: vec![],
        topology_retry_events: vec![],
        mode_collapse_count: 0,
        epistemic_yield: None,
        provenance_map: None,
        task_quadrant: Some(TaskQuadrant::Precision),
        complexity_event: complexity,
        frontier_event: None,
        adapter_correctness: vec![],
        coherence_state: CoherenceState {
            uncovered_domains: vec![],
            active_contradictions: vec![],
        },
        comparison_events: vec![],
        shadow_audit_events: vec![],
        correlated_warnings: vec![],
        researcher_grounding_events: vec![],
        diversity_degraded_event: None,
        oracle_gate_passed: None,
        leader_elected_events: vec![],
        socratic_diagnosis_events: vec![],
        consensus_agreement_rate: None,
        tokens_used: 0,
    }
}
