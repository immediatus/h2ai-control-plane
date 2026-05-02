use chrono::Utc;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_state::bft::ConsensusMedian;
use h2ai_state::krum::{
    cluster_coherent, krum_select_semantic, multi_krum_select_semantic, quorum_satisfied,
};
use h2ai_state::semilattice::{ProposalSet, SemilatticeResult};
use h2ai_state::weiszfeld;
use h2ai_types::events::{
    BranchPrunedEvent, MergeResolvedEvent, SemilatticeCompiledEvent, ZeroSurvivalEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::physics::MergeStrategy;
use tokio::time::Instant;

/// Result of a single `MergeEngine::resolve` call.
///
/// `Resolved` carries both the structural semilattice audit event and the
/// final merged output; `ZeroSurvival` is returned when every proposal was
/// pruned before merge could begin, signalling the orchestrator to retry.
pub enum MergeOutcome {
    /// All proposals were validated and at least one survived the semilattice compile step.
    Resolved {
        compiled: SemilatticeCompiledEvent,
        resolved: MergeResolvedEvent,
    },
    /// No proposals survived the semilattice compile step; the task should be retried.
    ZeroSurvival(ZeroSurvivalEvent),
}

/// Stateless merge coordinator that selects a consensus output from a `ProposalSet`.
///
/// Strategy dispatch, BFT quorum checks, and Weiszfeld fallback are all encapsulated
/// here so callers only need to supply a `MergeStrategy` and an optional embedding model.
pub struct MergeEngine;

impl MergeEngine {
    /// Merge a set of explorer proposals into a single consensus output.
    ///
    /// The strategy selection chain is:
    /// 1. Semilattice compile — prunes proposals that violate structural invariants.
    ///    Returns `ZeroSurvival` immediately if no proposals survive.
    /// 2. Strategy dispatch — `ConsensusMedian` and `ScoreOrdered` are applied directly.
    ///    `OutlierResistant`/`MultiOutlierResistant` first check BFT quorum and cluster
    ///    coherence; when both hold, Krum selection is used.
    /// 3. Weiszfeld fallback — when quorum or coherence fails and `embedding_model` is
    ///    present, the geometric median in embedding space is selected (breakdown point 50%).
    /// 4. `ConsensusMedian` fallback — used when quorum/coherence fails and no embedding
    ///    model is available, handling honest stochastic divergence without a cluster assumption.
    pub async fn resolve(
        task_id: TaskId,
        proposals: ProposalSet,
        pruned: Vec<BranchPrunedEvent>,
        strategy: MergeStrategy,
        retry_count: u32,
        embedding_model: Option<&dyn EmbeddingModel>,
    ) -> MergeOutcome {
        let merge_start = Instant::now();
        let n_input = proposals.len() + pruned.len();
        let result = SemilatticeResult::compile(task_id.clone(), proposals, pruned);

        if result.valid_proposals.is_empty() {
            return MergeOutcome::ZeroSurvival(ZeroSurvivalEvent {
                task_id,
                retry_count,
                timestamp: Utc::now(),
            });
        }

        let resolved_output = match strategy {
            MergeStrategy::ConsensusMedian => {
                ConsensusMedian::resolve(&result.valid_proposals, embedding_model)
                    .await
                    .map(|p| p.raw_output.clone())
                    .unwrap_or_default()
            }
            MergeStrategy::ScoreOrdered => result
                .valid_proposals
                .first()
                .map(|p| p.raw_output.clone())
                .unwrap_or_default(),
            MergeStrategy::OutlierResistant { f } => {
                let proposals = &result.valid_proposals;
                if quorum_satisfied(proposals.len(), f)
                    && cluster_coherent(proposals, embedding_model).await
                {
                    krum_select_semantic(proposals, f, embedding_model)
                        .await
                        .map(|p| p.raw_output.clone())
                        .unwrap_or_default()
                } else {
                    // Quorum not met OR cluster assumption violated (diverse stochastic outputs).
                    // With an embedding model: Weiszfeld geometric median (breakdown 50%).
                    // Without: ConsensusMedian handles honest divergence without requiring a cluster.
                    match embedding_model {
                        Some(model) => {
                            let embeddings: Vec<Vec<f32>> = proposals
                                .iter()
                                .map(|p| model.embed(&p.raw_output))
                                .collect();
                            let idx = weiszfeld::weiszfeld_select(&embeddings, 20);
                            proposals
                                .get(idx)
                                .map(|p| p.raw_output.clone())
                                .unwrap_or_default()
                        }
                        None => ConsensusMedian::resolve(proposals, embedding_model)
                            .await
                            .map(|p| p.raw_output.clone())
                            .unwrap_or_default(),
                    }
                }
            }
            MergeStrategy::MultiOutlierResistant { f, m } => {
                let proposals = &result.valid_proposals;
                if quorum_satisfied(proposals.len(), f)
                    && cluster_coherent(proposals, embedding_model).await
                {
                    let survivors =
                        multi_krum_select_semantic(proposals, f, m, embedding_model).await;
                    // valid_proposals is sorted by verification score descending.
                    // Pick the survivor that appears earliest (= highest verification score).
                    proposals
                        .iter()
                        .find(|p| survivors.iter().any(|s| s.explorer_id == p.explorer_id))
                        .map(|p| p.raw_output.clone())
                        .unwrap_or_default()
                } else {
                    // Quorum not met OR cluster assumption violated.
                    // With an embedding model: Weiszfeld geometric median (breakdown 50%).
                    // Without: ConsensusMedian handles honest stochastic divergence.
                    match embedding_model {
                        Some(model) => {
                            let embeddings: Vec<Vec<f32>> = proposals
                                .iter()
                                .map(|p| model.embed(&p.raw_output))
                                .collect();
                            let idx = weiszfeld::weiszfeld_select(&embeddings, 20);
                            proposals
                                .get(idx)
                                .map(|p| p.raw_output.clone())
                                .unwrap_or_default()
                        }
                        None => ConsensusMedian::resolve(proposals, embedding_model)
                            .await
                            .map(|p| p.raw_output.clone())
                            .unwrap_or_default(),
                    }
                }
            }
        };

        let merge_elapsed = merge_start.elapsed().as_secs_f64();
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
            merge_elapsed_secs: Some(merge_elapsed),
            n_input_proposals: n_input,
        };

        let resolved = MergeResolvedEvent {
            task_id,
            resolved_output,
            timestamp: Utc::now(),
        };

        MergeOutcome::Resolved { compiled, resolved }
    }
}
