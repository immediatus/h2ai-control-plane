use async_trait::async_trait;
use chrono::Utc;
use h2ai_adapters::MockAdapter;
use h2ai_orchestrator::compound::{CompoundError, CompoundTaskEngine, CompoundTaskInput};
use h2ai_orchestrator::scheduler::{SchedulerError, SubtaskExecutor};
use h2ai_types::config::ParetoWeights;
use h2ai_types::identity::{SubtaskId, TaskId};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use h2ai_types::physics::TauValue;
use h2ai_types::plan::{PlanStatus, SubtaskResult};

fn manifest() -> TaskManifest {
    TaskManifest {
        description: "Build a user authentication service".into(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
        topology: TopologyRequest {
            kind: "auto".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 2,
            tau_min: None,
            tau_max: None,
            roles: vec![],
            review_gates: vec![],
        },
        constraints: vec![],
        context: None,
    }
}

struct StubExecutor;
#[async_trait]
impl SubtaskExecutor for StubExecutor {
    async fn execute(
        &self,
        id: SubtaskId,
        m: TaskManifest,
    ) -> Result<SubtaskResult, SchedulerError> {
        Ok(SubtaskResult {
            subtask_id: id,
            output: format!("stub output for: {}", m.description),
            token_cost: 1,
            timestamp: Utc::now(),
        })
    }
}

#[tokio::test]
async fn compound_engine_decomposes_reviews_and_schedules() {
    let decomp_adapter = MockAdapter::new(
        r#"{
      "subtasks": [
        {"description": "Design schema", "depends_on": [], "role_hint": null},
        {"description": "Implement service", "depends_on": [0], "role_hint": null}
      ]
    }"#
        .into(),
    );
    let review_adapter =
        MockAdapter::new(r#"{"approved": true, "reason": "Looks complete."}"#.into());

    let task_id = TaskId::new();
    let input = CompoundTaskInput {
        task_id: task_id.clone(),
        manifest: manifest(),
        planning_adapter: &decomp_adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        review_adapter: &review_adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        planning_tau: TauValue::new(0.1).unwrap(),
        executor: &StubExecutor,
    };

    let output = CompoundTaskEngine::run(input).await.unwrap();
    assert_eq!(output.subtask_results.len(), 2);
    assert_eq!(output.plan.subtasks.len(), 2);
    assert_eq!(output.plan.status, PlanStatus::Approved);
    assert_eq!(output.plan.parent_task_id, task_id);
}

#[tokio::test]
async fn compound_engine_returns_plan_rejected_error_when_review_fails() {
    let decomp_adapter = MockAdapter::new(
        r#"{
      "subtasks": [
        {"description": "Only one step", "depends_on": [], "role_hint": null}
      ]
    }"#
        .into(),
    );
    let review_adapter =
        MockAdapter::new(r#"{"approved": false, "reason": "Missing implementation step."}"#.into());

    let input = CompoundTaskInput {
        task_id: TaskId::new(),
        manifest: manifest(),
        planning_adapter: &decomp_adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        review_adapter: &review_adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        planning_tau: TauValue::new(0.1).unwrap(),
        executor: &StubExecutor,
    };

    let err = CompoundTaskEngine::run(input).await.unwrap_err();
    assert!(
        matches!(&err, CompoundError::PlanRejected { reason } if reason.contains("implementation")),
        "expected PlanRejected; got: {err:?}"
    );
}

#[tokio::test]
async fn compound_engine_returns_error_on_invalid_decomposer_json() {
    let decomp_adapter = MockAdapter::new("not valid json at all".into());
    let review_adapter = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());

    let input = CompoundTaskInput {
        task_id: TaskId::new(),
        manifest: manifest(),
        planning_adapter: &decomp_adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        review_adapter: &review_adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        planning_tau: TauValue::new(0.1).unwrap(),
        executor: &StubExecutor,
    };

    let err = CompoundTaskEngine::run(input).await.unwrap_err();
    assert!(matches!(err, CompoundError::Planning(_)));
}

#[tokio::test]
async fn compound_engine_empty_subtasks_from_decomposer_results_in_plan_rejected() {
    // LLM returns empty subtasks array → reviewer rejects plan locally (no LLM call)
    // → CompoundError::PlanRejected.
    let decomp_adapter = MockAdapter::new(r#"{"subtasks": []}"#.into());
    let review_adapter = MockAdapter::new("NOT JSON - should not be called".into());

    let input = CompoundTaskInput {
        task_id: TaskId::new(),
        manifest: manifest(),
        planning_adapter: &decomp_adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        review_adapter: &review_adapter as &dyn h2ai_types::adapter::IComputeAdapter,
        planning_tau: TauValue::new(0.1).unwrap(),
        executor: &StubExecutor,
    };

    let err = CompoundTaskEngine::run(input).await.unwrap_err();
    assert!(
        matches!(err, CompoundError::PlanRejected { .. }),
        "empty decomposition must result in PlanRejected; got: {err:?}"
    );
}
