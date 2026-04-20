use async_trait::async_trait;
use chrono::Utc;
use h2ai_orchestrator::scheduler::{SchedulerError, SchedulingEngine, SubtaskExecutor};
use h2ai_types::config::ParetoWeights;
use h2ai_types::identity::{SubtaskId, TaskId};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use h2ai_types::plan::{PlanStatus, Subtask, SubtaskPlan, SubtaskResult};

struct FailExecutor;
#[async_trait]
impl SubtaskExecutor for FailExecutor {
    async fn execute(&self, id: SubtaskId, _m: TaskManifest) -> Result<SubtaskResult, SchedulerError> {
        Err(SchedulerError::ExecutionFailed {
            subtask_id: id,
            message: "intentional failure".into(),
        })
    }
}

fn base_manifest() -> TaskManifest {
    TaskManifest {
        description: "parent task".into(),
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

struct EchoExecutor;

#[async_trait]
impl SubtaskExecutor for EchoExecutor {
    async fn execute(
        &self,
        subtask_id: SubtaskId,
        manifest: TaskManifest,
    ) -> Result<SubtaskResult, SchedulerError> {
        Ok(SubtaskResult {
            subtask_id,
            output: format!("result of: {}", manifest.description),
            token_cost: 1,
            timestamp: Utc::now(),
        })
    }
}

fn linear_plan() -> (SubtaskPlan, SubtaskId, SubtaskId, SubtaskId) {
    let a = SubtaskId::new();
    let b = SubtaskId::new();
    let c = SubtaskId::new();
    let plan = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![
            Subtask {
                id: a.clone(),
                description: "step A".into(),
                depends_on: vec![],
                role_hint: None,
            },
            Subtask {
                id: b.clone(),
                description: "step B".into(),
                depends_on: vec![a.clone()],
                role_hint: None,
            },
            Subtask {
                id: c.clone(),
                description: "step C".into(),
                depends_on: vec![b.clone()],
                role_hint: None,
            },
        ],
        status: PlanStatus::Approved,
        created_at: Utc::now(),
    };
    (plan, a, b, c)
}

#[tokio::test]
async fn scheduler_executes_all_subtasks_and_returns_results() {
    let (plan, a, b, c) = linear_plan();
    let results = SchedulingEngine::execute(plan, &base_manifest(), &EchoExecutor)
        .await
        .unwrap();
    assert_eq!(results.len(), 3);
    let ids: Vec<_> = results.iter().map(|r| r.subtask_id.clone()).collect();
    assert!(ids.contains(&a));
    assert!(ids.contains(&b));
    assert!(ids.contains(&c));
}

#[tokio::test]
async fn scheduler_injects_dep_output_as_context_in_manifest() {
    use std::sync::{Arc, Mutex};
    struct ContextCapture(Arc<Mutex<Vec<Option<String>>>>);
    #[async_trait]
    impl SubtaskExecutor for ContextCapture {
        async fn execute(
            &self,
            id: SubtaskId,
            manifest: TaskManifest,
        ) -> Result<SubtaskResult, SchedulerError> {
            self.0.lock().unwrap().push(manifest.context.clone());
            Ok(SubtaskResult {
                subtask_id: id,
                output: "result of: A".into(),
                token_cost: 0,
                timestamp: Utc::now(),
            })
        }
    }

    let a = SubtaskId::new();
    let b = SubtaskId::new();
    let plan = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![
            Subtask {
                id: a.clone(),
                description: "A".into(),
                depends_on: vec![],
                role_hint: None,
            },
            Subtask {
                id: b.clone(),
                description: "B".into(),
                depends_on: vec![a.clone()],
                role_hint: None,
            },
        ],
        status: PlanStatus::Approved,
        created_at: Utc::now(),
    };

    let captured = Arc::new(Mutex::new(Vec::new()));
    let executor = ContextCapture(captured.clone());
    SchedulingEngine::execute(plan, &base_manifest(), &executor)
        .await
        .unwrap();

    let contexts = captured.lock().unwrap();
    assert_eq!(contexts.len(), 2);
    assert!(
        contexts[0].is_none(),
        "A has no deps so no injected context"
    );
    assert!(
        contexts[1]
            .as_deref()
            .unwrap_or("")
            .contains("result of: A"),
        "B must receive A's output in context; got: {:?}",
        contexts[1]
    );
}

#[tokio::test]
async fn scheduler_returns_cyclic_dependency_error() {
    let a = SubtaskId::new();
    let b = SubtaskId::new();
    let cyclic = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![
            Subtask {
                id: a.clone(),
                description: "A".into(),
                depends_on: vec![b.clone()],
                role_hint: None,
            },
            Subtask {
                id: b.clone(),
                description: "B".into(),
                depends_on: vec![a.clone()],
                role_hint: None,
            },
        ],
        status: PlanStatus::Approved,
        created_at: Utc::now(),
    };
    let result = SchedulingEngine::execute(cyclic, &base_manifest(), &EchoExecutor).await;
    assert!(matches!(result, Err(SchedulerError::CyclicDependency)));
}

#[tokio::test]
async fn scheduler_parallelises_independent_subtasks() {
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use tokio::time::sleep;

    struct SlowExecutor(Arc<Mutex<Vec<Instant>>>);
    #[async_trait]
    impl SubtaskExecutor for SlowExecutor {
        async fn execute(
            &self,
            id: SubtaskId,
            _m: TaskManifest,
        ) -> Result<SubtaskResult, SchedulerError> {
            self.0.lock().unwrap().push(Instant::now());
            sleep(Duration::from_millis(20)).await;
            Ok(SubtaskResult {
                subtask_id: id,
                output: "done".into(),
                token_cost: 0,
                timestamp: Utc::now(),
            })
        }
    }

    let a = SubtaskId::new();
    let b = SubtaskId::new();
    let plan = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![
            Subtask {
                id: a.clone(),
                description: "A".into(),
                depends_on: vec![],
                role_hint: None,
            },
            Subtask {
                id: b.clone(),
                description: "B".into(),
                depends_on: vec![],
                role_hint: None,
            },
        ],
        status: PlanStatus::Approved,
        created_at: Utc::now(),
    };

    let starts = Arc::new(Mutex::new(Vec::new()));
    let executor = SlowExecutor(starts.clone());
    let before = Instant::now();
    SchedulingEngine::execute(plan, &base_manifest(), &executor)
        .await
        .unwrap();
    let elapsed = before.elapsed();

    assert!(
        elapsed < Duration::from_millis(38),
        "independent subtasks must run in parallel; elapsed={elapsed:?}"
    );
}

#[tokio::test]
async fn scheduler_empty_plan_returns_empty_results() {
    let empty_plan = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![],
        status: PlanStatus::Approved,
        created_at: Utc::now(),
    };
    let results = SchedulingEngine::execute(empty_plan, &base_manifest(), &EchoExecutor)
        .await
        .unwrap();
    assert!(results.is_empty(), "empty plan must yield empty results");
}

#[tokio::test]
async fn scheduler_single_subtask_returns_one_result() {
    let id = SubtaskId::new();
    let plan = SubtaskPlan {
        plan_id: TaskId::new(),
        parent_task_id: TaskId::new(),
        subtasks: vec![Subtask {
            id: id.clone(),
            description: "only step".into(),
            depends_on: vec![],
            role_hint: None,
        }],
        status: PlanStatus::Approved,
        created_at: Utc::now(),
    };
    let results = SchedulingEngine::execute(plan, &base_manifest(), &EchoExecutor)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].subtask_id, id);
}

#[tokio::test]
async fn scheduler_propagates_executor_failure() {
    let (plan, _, _, _) = linear_plan();
    let result = SchedulingEngine::execute(plan, &base_manifest(), &FailExecutor).await;
    assert!(
        matches!(result, Err(SchedulerError::ExecutionFailed { .. })),
        "executor failure must propagate; got: {result:?}"
    );
}
