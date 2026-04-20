use crate::scheduler::{SchedulerError, SchedulingEngine, SubtaskExecutor};
use h2ai_planner::decomposer::{PlannerError, PlanningEngine};
use h2ai_planner::reviewer::{PlanReviewer, ReviewOutcome};
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::TaskManifest;
use h2ai_types::physics::TauValue;
use h2ai_types::plan::{PlanStatus, SubtaskPlan, SubtaskResult};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompoundError {
    #[error("planning failed: {0}")]
    Planning(#[from] PlannerError),
    #[error("plan rejected: {reason}")]
    PlanRejected { reason: String },
    #[error("scheduling failed: {0}")]
    Scheduling(#[from] SchedulerError),
}

pub struct CompoundTaskInput<'a> {
    pub task_id: TaskId,
    pub manifest: TaskManifest,
    /// Adapter used for task decomposition (typically low tau for precision).
    pub planning_adapter: &'a dyn IComputeAdapter,
    /// Adapter used for plan semantic review.
    pub review_adapter: &'a dyn IComputeAdapter,
    pub planning_tau: TauValue,
    /// Executes individual subtasks (use `EngineExecutor` in production, mock in tests).
    pub executor: &'a dyn SubtaskExecutor,
}

#[derive(Debug)]
pub struct CompoundTaskOutput {
    pub task_id: TaskId,
    pub plan: SubtaskPlan,
    pub subtask_results: Vec<SubtaskResult>,
}

pub struct CompoundTaskEngine;

impl CompoundTaskEngine {
    /// Run the decompose → review → schedule pipeline.
    ///
    /// 1. Decompose `manifest` into a `SubtaskPlan` via LLM.
    /// 2. Auto-review the plan; return `CompoundError::PlanRejected` if rejected.
    /// 3. Execute approved subtasks in topological order via `executor`.
    pub async fn run(input: CompoundTaskInput<'_>) -> Result<CompoundTaskOutput, CompoundError> {
        // Step 1: Decompose.
        let mut plan =
            PlanningEngine::decompose(&input.manifest, input.planning_adapter, input.planning_tau)
                .await?;
        plan.parent_task_id = input.task_id.clone();

        // Step 2: Review.
        let outcome = PlanReviewer::evaluate(
            &plan,
            &input.manifest.description,
            input.review_adapter,
            input.planning_tau,
        )
        .await?;

        match outcome {
            ReviewOutcome::Rejected { reason } => {
                return Err(CompoundError::PlanRejected { reason });
            }
            ReviewOutcome::Approved => {
                plan.status = PlanStatus::Approved;
            }
        }

        // Step 3: Schedule and execute.
        let subtask_results =
            SchedulingEngine::execute(plan.clone(), &input.manifest, input.executor).await?;

        Ok(CompoundTaskOutput {
            task_id: input.task_id,
            plan,
            subtask_results,
        })
    }
}
