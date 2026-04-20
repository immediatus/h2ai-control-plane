use chrono::Utc;
use h2ai_state::bft::ConsensusMedian;
use h2ai_state::krum::{cluster_coherent, krum_select, multi_krum_select, quorum_satisfied};
use h2ai_state::semilattice::{ProposalSet, SemilatticeResult};
use h2ai_types::adapter::IComputeAdapter;
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
    pub async fn resolve(
        task_id: TaskId,
        proposals: ProposalSet,
        pruned: Vec<BranchPrunedEvent>,
        strategy: MergeStrategy,
        retry_count: u32,
        adapter: Option<&dyn IComputeAdapter>,
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
            MergeStrategy::ConsensusMedian => ConsensusMedian::resolve(&result.valid_proposals, adapter)
                .await
                .map(|p| p.raw_output.clone())
                .unwrap_or_default(),
            MergeStrategy::ScoreOrdered => result
                .valid_proposals
                .first()
                .map(|p| p.raw_output.clone())
                .unwrap_or_default(),
            MergeStrategy::Krum { f } => {
                let proposals = &result.valid_proposals;
                if quorum_satisfied(proposals.len(), f) && cluster_coherent(proposals, adapter).await {
                    krum_select(proposals, f)
                        .map(|p| p.raw_output.clone())
                        .unwrap_or_default()
                } else {
                    // Quorum not met OR cluster assumption violated (diverse stochastic outputs).
                    // ConsensusMedian handles honest divergence without requiring a cluster.
                    ConsensusMedian::resolve(proposals, adapter)
                        .await
                        .map(|p| p.raw_output.clone())
                        .unwrap_or_default()
                }
            }
            MergeStrategy::MultiKrum { f, m } => {
                let proposals = &result.valid_proposals;
                if quorum_satisfied(proposals.len(), f) && cluster_coherent(proposals, adapter).await {
                    let survivors = multi_krum_select(proposals, f, m);
                    // valid_proposals is sorted by verification score descending.
                    // Pick the survivor that appears earliest (= highest verification score).
                    proposals
                        .iter()
                        .find(|p| survivors.iter().any(|s| s.explorer_id == p.explorer_id))
                        .map(|p| p.raw_output.clone())
                        .unwrap_or_default()
                } else {
                    // Quorum not met OR cluster assumption violated — fall back to ConsensusMedian.
                    ConsensusMedian::resolve(proposals, adapter)
                        .await
                        .map(|p| p.raw_output.clone())
                        .unwrap_or_default()
                }
            }
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
