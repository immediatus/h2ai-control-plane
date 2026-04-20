use chrono::Utc;
use h2ai_state::bft::BftConsensus;
use h2ai_state::semilattice::{ProposalSet, SemilatticeResult};
use h2ai_types::events::{
    BranchPrunedEvent, MergeResolvedEvent, SemilatticeCompiledEvent, ZeroSurvivalEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::physics::MergeStrategy;

pub enum MergeOutcome {
    Resolved {
        compiled: SemilatticeCompiledEvent,
        resolved: MergeResolvedEvent,
    },
    ZeroSurvival(ZeroSurvivalEvent),
}

pub struct MergeEngine;

impl MergeEngine {
    pub fn resolve(
        task_id: TaskId,
        proposals: ProposalSet,
        pruned: Vec<BranchPrunedEvent>,
        strategy: MergeStrategy,
        retry_count: u32,
    ) -> MergeOutcome {
        let result = SemilatticeResult::compile(task_id.clone(), proposals, pruned);

        if result.valid_proposals.is_empty() {
            return MergeOutcome::ZeroSurvival(ZeroSurvivalEvent {
                task_id,
                retry_count,
                timestamp: Utc::now(),
            });
        }

        let resolved_output = match strategy {
            MergeStrategy::BftConsensus => BftConsensus::resolve(&result.valid_proposals)
                .map(|p| p.raw_output.clone())
                .unwrap_or_default(),
            MergeStrategy::CrdtSemilattice => result
                .valid_proposals
                .first()
                .map(|p| p.raw_output.clone())
                .unwrap_or_default(),
        };

        let compiled = SemilatticeCompiledEvent {
            task_id: task_id.clone(),
            valid_proposals: result
                .valid_proposals
                .iter()
                .map(|p| p.explorer_id.clone())
                .collect(),
            pruned_proposals: result
                .pruned_proposals
                .iter()
                .map(|p| (p.explorer_id.clone(), p.reason.clone()))
                .collect(),
            merge_strategy: strategy,
            timestamp: Utc::now(),
        };

        let resolved = MergeResolvedEvent {
            task_id,
            resolved_output,
            timestamp: Utc::now(),
        };

        MergeOutcome::Resolved { compiled, resolved }
    }
}
