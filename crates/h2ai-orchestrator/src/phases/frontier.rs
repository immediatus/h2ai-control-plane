use crate::engine::EngineInput;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_types::events::{ConstraintFrontierEvent, ProposalEvent};
use h2ai_types::identity::TaskId;

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub task_id: &'a TaskId,
    pub synthesis_candidates: &'a [ProposalEvent],
}

/// Build the constraint satisfaction matrix and compute Pareto frontier coverage
/// (Phase 4.5).
///
/// This is a pure enrichment step — it never fails or early-exits. Returns `None`
/// when there are no synthesis candidates or no Static-tier constraints in the corpus.
#[must_use]
pub fn run(input: Input<'_>) -> Option<ConstraintFrontierEvent> {
    let static_constraints: Vec<&ConstraintDoc> = input
        .engine_input
        .constraint_corpus
        .iter()
        .filter(|d| d.tier() == h2ai_constraints::types::ConstraintTier::Static)
        .collect();

    if input.synthesis_candidates.is_empty() || static_constraints.is_empty() {
        return None;
    }

    let constraint_ids: Vec<String> = static_constraints.iter().map(|c| c.id.clone()).collect();
    let explorer_ids: Vec<h2ai_types::identity::ExplorerId> = input
        .synthesis_candidates
        .iter()
        .map(|p| p.explorer_id.clone())
        .collect();
    let satisfaction_matrix: Vec<Vec<f64>> = input
        .synthesis_candidates
        .iter()
        .map(|proposal| {
            static_constraints
                .iter()
                .map(|c| h2ai_constraints::eval::eval_sync(&c.predicate, &proposal.raw_output))
                .collect()
        })
        .collect();
    let pareto_coverage = crate::complexity::participation_ratio(&satisfaction_matrix);

    Some(ConstraintFrontierEvent {
        task_id: input.task_id.clone(),
        satisfaction_matrix,
        constraint_ids,
        explorer_ids,
        pareto_coverage,
        timestamp: chrono::Utc::now(),
    })
}
