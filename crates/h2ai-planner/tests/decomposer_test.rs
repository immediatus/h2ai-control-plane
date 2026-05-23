use async_trait::async_trait;
use h2ai_planner::decomposer::{PlannerError, PlanningEngine};
use h2ai_test_utils::MockAdapter;
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::config::{AgentRole, ParetoWeights};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use h2ai_types::plan::PlanStatus;
use h2ai_types::sizing::TauValue;

/// An adapter that always returns an error — used to exercise the Adapter error path.
#[derive(Debug)]
struct FailingAdapter;

#[async_trait]
impl IComputeAdapter for FailingAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Err(AdapterError::NetworkError(
            "simulated network failure".into(),
        ))
    }

    fn kind(&self) -> &AdapterKind {
        unimplemented!("FailingAdapter::kind not needed in tests")
    }
}

fn manifest() -> TaskManifest {
    TaskManifest {
        description: "Build a REST API for user authentication with JWT tokens".into(),
        pareto_weights: ParetoWeights::new(0.33, 0.33, 0.34).unwrap(),
        topology: TopologyRequest {
            kind: "auto".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 3,
            tau_min: None,
            tau_max: None,
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![],
            diversity_ids: vec![],
        },
        constraints: vec!["ADR-001".into()],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    }
}

#[tokio::test]
async fn decomposer_parses_llm_json_into_subtask_plan() {
    let adapter = MockAdapter::new(r#"{
      "subtasks": [
        {"description": "Design the user model and DB schema", "depends_on": [], "role_hint": "Executor"},
        {"description": "Implement JWT token generation and validation", "depends_on": [0], "role_hint": "Executor"},
        {"description": "Write integration tests for auth endpoints", "depends_on": [1], "role_hint": "Evaluator"}
      ]
    }"#.into());

    let plan = PlanningEngine::decompose(&manifest(), &adapter, TauValue::new(0.4).unwrap())
        .await
        .unwrap();

    assert_eq!(plan.subtasks.len(), 3);
    assert!(
        plan.subtasks[0].depends_on.is_empty(),
        "first subtask has no deps"
    );
    assert_eq!(plan.subtasks[1].depends_on.len(), 1);
    assert_eq!(plan.subtasks[1].depends_on[0], plan.subtasks[0].id);
    assert!(matches!(plan.status, PlanStatus::PendingReview));
}

#[tokio::test]
async fn decomposer_handles_markdown_fenced_json() {
    let adapter = MockAdapter::new(
        r#"```json
{
  "subtasks": [
    {"description": "Step one", "depends_on": [], "role_hint": null},
    {"description": "Step two", "depends_on": [0], "role_hint": null}
  ]
}
```"#
            .into(),
    );

    let plan = PlanningEngine::decompose(&manifest(), &adapter, TauValue::new(0.3).unwrap())
        .await
        .unwrap();

    assert_eq!(plan.subtasks.len(), 2);
}

#[tokio::test]
async fn decomposer_returns_error_on_invalid_json() {
    let adapter = MockAdapter::new("I cannot decompose this task.".into());
    let result =
        PlanningEngine::decompose(&manifest(), &adapter, TauValue::new(0.4).unwrap()).await;
    assert!(
        matches!(result, Err(PlannerError::ParseError(_))),
        "expected ParseError"
    );
}

#[tokio::test]
async fn decomposer_assigns_stable_subtask_ids() {
    let adapter = MockAdapter::new(
        r#"{
      "subtasks": [
        {"description": "A", "depends_on": [], "role_hint": null},
        {"description": "B", "depends_on": [0], "role_hint": null},
        {"description": "C", "depends_on": [0, 1], "role_hint": null}
      ]
    }"#
        .into(),
    );

    let plan = PlanningEngine::decompose(&manifest(), &adapter, TauValue::new(0.4).unwrap())
        .await
        .unwrap();

    let id_a = &plan.subtasks[0].id;
    let id_b = &plan.subtasks[1].id;
    let c_deps = &plan.subtasks[2].depends_on;
    assert!(c_deps.contains(id_a), "C must depend on A");
    assert!(c_deps.contains(id_b), "C must depend on B");
}

#[tokio::test]
async fn decomposer_returns_error_on_out_of_range_dependency_index() {
    // Index 5 is out of range for a 2-subtask plan.
    let adapter = MockAdapter::new(
        r#"{
          "subtasks": [
            {"description": "Step A", "depends_on": [], "role_hint": null},
            {"description": "Step B", "depends_on": [5], "role_hint": null}
          ]
        }"#
        .into(),
    );
    let result =
        PlanningEngine::decompose(&manifest(), &adapter, TauValue::new(0.4).unwrap()).await;
    assert!(
        matches!(result, Err(PlannerError::InvalidDependencyIndex { .. })),
        "expected InvalidDependencyIndex error, got: {result:?}"
    );
}

#[tokio::test]
async fn decomposer_empty_subtasks_array_produces_empty_plan() {
    // LLM returns an empty subtasks array — structurally valid JSON; reviewer will reject it.
    let adapter = MockAdapter::new(r#"{"subtasks": []}"#.into());
    let plan = PlanningEngine::decompose(&manifest(), &adapter, TauValue::new(0.4).unwrap())
        .await
        .unwrap();
    assert!(
        plan.subtasks.is_empty(),
        "empty subtasks array should yield empty plan"
    );
}

#[tokio::test]
async fn decomposer_unrecognised_role_hint_yields_none() {
    let adapter = MockAdapter::new(
        r#"{
          "subtasks": [
            {"description": "Step A", "depends_on": [], "role_hint": "UnknownRole"},
            {"description": "Step B", "depends_on": [0], "role_hint": "Executor"}
          ]
        }"#
        .into(),
    );
    let plan = PlanningEngine::decompose(&manifest(), &adapter, TauValue::new(0.4).unwrap())
        .await
        .unwrap();
    assert!(
        plan.subtasks[0].role_hint.is_none(),
        "unrecognised role_hint must be silently discarded"
    );
    assert_eq!(
        plan.subtasks[1].role_hint,
        Some(AgentRole::Executor),
        "known role_hint must be preserved"
    );
}

#[tokio::test]
async fn decomposer_recognises_synthesizer_and_coordinator_role_hints() {
    let adapter = MockAdapter::new(
        r#"{
          "subtasks": [
            {"description": "Synthesize results", "depends_on": [], "role_hint": "Synthesizer"},
            {"description": "Coordinate work", "depends_on": [0], "role_hint": "Coordinator"},
            {"description": "Evaluate output", "depends_on": [0], "role_hint": "Evaluator"}
          ]
        }"#
        .into(),
    );
    let plan = PlanningEngine::decompose(&manifest(), &adapter, TauValue::new(0.4).unwrap())
        .await
        .unwrap();
    assert_eq!(
        plan.subtasks[0].role_hint,
        Some(AgentRole::Synthesizer),
        "Synthesizer role_hint must be preserved"
    );
    assert_eq!(
        plan.subtasks[1].role_hint,
        Some(AgentRole::Coordinator),
        "Coordinator role_hint must be preserved"
    );
    assert_eq!(
        plan.subtasks[2].role_hint,
        Some(AgentRole::Evaluator),
        "Evaluator role_hint must be preserved"
    );
}

#[tokio::test]
async fn decomposer_propagates_adapter_error() {
    let result =
        PlanningEngine::decompose(&manifest(), &FailingAdapter, TauValue::new(0.4).unwrap()).await;
    assert!(
        matches!(result, Err(PlannerError::Adapter(_))),
        "adapter failure must propagate as PlannerError::Adapter, got: {result:?}"
    );
}

#[tokio::test]
async fn decomposer_handles_manifest_with_no_constraints() {
    // Exercises the `constraints_str = "none"` branch in decompose().
    let mut m = manifest();
    m.constraints = vec![];
    let adapter = MockAdapter::new(
        r#"{"subtasks": [{"description": "Only step", "depends_on": [], "role_hint": null}]}"#
            .into(),
    );
    let plan = PlanningEngine::decompose(&m, &adapter, TauValue::new(0.4).unwrap())
        .await
        .unwrap();
    assert_eq!(plan.subtasks.len(), 1);
}
