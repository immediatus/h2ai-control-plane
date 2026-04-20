use async_trait::async_trait;
use futures::future::join_all;
use h2ai_types::identity::SubtaskId;
use h2ai_types::manifest::TaskManifest;
use h2ai_types::plan::{Subtask, SubtaskPlan, SubtaskResult};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("cyclic dependency detected in subtask plan")]
    CyclicDependency,
    #[error("subtask execution failed for {subtask_id}: {message}")]
    ExecutionFailed {
        subtask_id: SubtaskId,
        message: String,
    },
}

/// Runs a single subtask given its ID and a fully-built `TaskManifest`
/// (description + injected dependency context). Implement with `ExecutionEngine`
/// in production; use a mock in tests.
#[async_trait]
pub trait SubtaskExecutor: Send + Sync {
    async fn execute(
        &self,
        subtask_id: SubtaskId,
        manifest: TaskManifest,
    ) -> Result<SubtaskResult, SchedulerError>;
}

pub struct SchedulingEngine;

impl SchedulingEngine {
    /// Execute all subtasks in `plan` respecting dependency order.
    ///
    /// Subtasks with all dependencies satisfied form a **wave** and run in parallel
    /// via `join_all`. Results from completed subtasks are injected as `context`
    /// into the manifests of their dependents.
    ///
    /// Results are returned in unspecified order (HashMap iteration order).
    pub async fn execute(
        plan: SubtaskPlan,
        parent_manifest: &TaskManifest,
        executor: &dyn SubtaskExecutor,
    ) -> Result<Vec<SubtaskResult>, SchedulerError> {
        let waves = topo_waves(&plan.subtasks)?;
        let mut completed: HashMap<SubtaskId, SubtaskResult> = HashMap::new();

        for wave in waves {
            let futures: Vec<_> = wave
                .into_iter()
                .map(|subtask_id| {
                    let subtask = plan
                        .subtasks
                        .iter()
                        .find(|s| s.id == subtask_id)
                        .expect("wave ID must exist in plan");
                    let manifest = build_subtask_manifest(subtask, parent_manifest, &completed);
                    // `executor` is `&dyn SubtaskExecutor` — a fat pointer, which is
                    // `Copy`. Each closure captures its own copy of the reference.
                    let exec = executor;
                    async move { exec.execute(subtask_id, manifest).await }
                })
                .collect();

            let wave_results = join_all(futures).await;
            for res in wave_results {
                let result = res?;
                completed.insert(result.subtask_id.clone(), result);
            }
        }

        Ok(completed.into_values().collect())
    }
}

/// Kahn's algorithm — returns execution waves (each wave's subtasks are independent).
fn topo_waves(subtasks: &[Subtask]) -> Result<Vec<Vec<SubtaskId>>, SchedulerError> {
    let n = subtasks.len();
    let mut in_degree: Vec<usize> = subtasks.iter().map(|s| s.depends_on.len()).collect();

    let id_to_idx: HashMap<&SubtaskId, usize> = subtasks
        .iter()
        .enumerate()
        .map(|(i, s)| (&s.id, i))
        .collect();

    // adj[dep_idx] = indices of subtasks that depend on subtask at dep_idx
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, s) in subtasks.iter().enumerate() {
        for dep_id in &s.depends_on {
            if let Some(&dep_idx) = id_to_idx.get(dep_id) {
                adj[dep_idx].push(i);
            }
        }
    }

    let mut waves: Vec<Vec<SubtaskId>> = Vec::new();
    let mut ready: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut visited = 0;

    while !ready.is_empty() {
        let wave_indices = std::mem::take(&mut ready);
        visited += wave_indices.len();

        let wave_ids: Vec<SubtaskId> = wave_indices
            .iter()
            .map(|&i| subtasks[i].id.clone())
            .collect();

        for idx in &wave_indices {
            for &dep in &adj[*idx] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    ready.push(dep);
                }
            }
        }

        waves.push(wave_ids);
    }

    if visited != n {
        return Err(SchedulerError::CyclicDependency);
    }
    Ok(waves)
}

/// Build the `TaskManifest` for a subtask, injecting completed dependency outputs.
fn build_subtask_manifest(
    subtask: &Subtask,
    parent: &TaskManifest,
    completed: &HashMap<SubtaskId, SubtaskResult>,
) -> TaskManifest {
    let dep_context = subtask
        .depends_on
        .iter()
        .filter_map(|id| {
            completed
                .get(id)
                .map(|r| format!("## Subtask Result\n\n{}", r.output))
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let context = if dep_context.is_empty() {
        parent.context.clone()
    } else {
        let base = parent.context.as_deref().unwrap_or("");
        let combined = if base.is_empty() {
            dep_context
        } else {
            format!("{base}\n\n---\n\n{dep_context}")
        };
        Some(combined)
    };

    TaskManifest {
        description: subtask.description.clone(),
        pareto_weights: parent.pareto_weights.clone(),
        topology: parent.topology.clone(),
        explorers: parent.explorers.clone(),
        constraints: parent.constraints.clone(),
        context,
    }
}
