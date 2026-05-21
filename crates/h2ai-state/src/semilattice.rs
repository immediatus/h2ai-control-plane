use h2ai_types::events::{BranchPrunedEvent, ProposalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use std::collections::{HashMap, HashSet};

/// A CRDT proposal map keyed by explorer, ordered by generation then verification score.
///
/// Implements a join-semilattice whose least-upper-bound (LUB) rule is generation-first:
/// a newer generation always supersedes an older one even if it carries a lower score,
/// because a TAO retry represents an authoritative replacement, not a concurrent alternative.
/// Ties within the same generation are resolved by preferring the higher score.
pub struct ProposalSet(HashMap<ExplorerId, (ProposalEvent, u64, f64)>);

impl ProposalSet {
    /// Create an empty proposal set.
    #[must_use]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Insert or update using generation-first LUB semantics.
    ///
    /// LUB rule: higher `generation` wins; within the same generation, higher `score` wins.
    ///
    /// This makes the semilattice correct under MAPE-K TAO retry loops where a later
    /// attempt (higher generation) must supersede an earlier one even if it scores lower.
    ///
    /// CRDT axioms satisfied (Shapiro et al. 2011):
    /// - Commutativity: join(S₁, S₂) = join(S₂, S₁)  [max is commutative]
    /// - Associativity: join(join(S₁,S₂),S₃) = join(S₁,join(S₂,S₃))  [set union]
    /// - Idempotency:   join(S, S) = S  [max(gen,score) of identical pairs = same pair]
    pub fn insert_scored(&mut self, proposal: ProposalEvent, score: f64) {
        let incoming_gen = proposal.generation;
        self.0
            .entry(proposal.explorer_id.clone())
            .and_modify(|(existing_proposal, existing_gen, existing_score)| {
                let should_replace = incoming_gen > *existing_gen
                    || (incoming_gen == *existing_gen && score > *existing_score);
                if should_replace {
                    *existing_proposal = proposal.clone();
                    *existing_gen = incoming_gen;
                    *existing_score = score;
                }
            })
            .or_insert((proposal, incoming_gen, score));
    }

    /// Merge two proposal sets, applying generation-first LUB resolution for each explorer.
    ///
    /// Prefer `join` when combining independently-accumulated sets (e.g. after a network
    /// partition or a fan-out collection step), because it satisfies the CRDT commutativity,
    /// associativity, and idempotency axioms.  Use `insert_scored` in a loop instead when
    /// appending proposals one at a time to a single accumulator.
    #[must_use]
    pub fn join(mut lhs: Self, rhs: Self) -> Self {
        for (_, (proposal, _gen, score)) in rhs.0 {
            lhs.insert_scored(proposal, score);
        }
        lhs
    }

    /// Insert a proposal without a verification score (score = 0.0).
    pub fn insert(&mut self, proposal: ProposalEvent) {
        self.insert_scored(proposal, 0.0);
    }

    /// Number of distinct explorers with a recorded proposal.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return `true` when no explorer has submitted a proposal yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return the current winning `ProposalEvent` for `explorer_id`, or `None` if the
    /// explorer has not yet submitted a proposal.
    #[must_use]
    pub fn get(&self, explorer_id: &ExplorerId) -> Option<&ProposalEvent> {
        self.0.get(explorer_id).map(|(p, _gen, _score)| p)
    }
}

impl Default for ProposalSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Output of compiling a `ProposalSet` into ranked valid, failed, and pruned partitions.
///
/// Produced by `SemilatticeResult::compile`; consumed by the merger to feed
/// the consensus selection step (Krum, `ConsensusMedian`, or Weiszfeld).
pub struct SemilatticeResult {
    /// Task this result belongs to.
    pub task_id: TaskId,
    /// Non-pruned proposals with verification score > 0, sorted descending.
    ///
    /// Index 0 holds the highest-scored proposal so `ScoreOrdered` merge needs
    /// only take `valid_proposals[0]`. Score=0 proposals are excluded to
    /// prevent synthesis contamination (GAP-D8).
    pub valid_proposals: Vec<ProposalEvent>,
    /// Verification scores for each entry in `valid_proposals` (parallel vector, same index).
    /// All values are > 0.0. Sorted descending in lock-step with `valid_proposals`.
    pub valid_proposal_scores: Vec<f64>,
    /// Non-pruned proposals that scored exactly 0.0 on every constraint.
    ///
    /// These failed verification. They are kept here for optional use as
    /// negative examples in the synthesis prompt ("anti-patterns to avoid")
    /// but must never feed the selection strategies alongside valid proposals.
    pub failed_proposals: Vec<ProposalEvent>,
    /// Proposals whose explorer branch was pruned before consensus; excluded
    /// from both `valid_proposals` and `failed_proposals`.
    pub pruned_proposals: Vec<BranchPrunedEvent>,
}

impl SemilatticeResult {
    #[must_use]
    pub fn compile(
        task_id: TaskId,
        proposals: ProposalSet,
        pruned: Vec<BranchPrunedEvent>,
    ) -> Self {
        let pruned_ids: HashSet<&ExplorerId> = pruned.iter().map(|p| &p.explorer_id).collect();

        let mut scored: Vec<(ProposalEvent, f64)> = proposals
            .0
            .into_values()
            .filter(|(p, _gen, _score)| !pruned_ids.contains(&p.explorer_id))
            .map(|(p, _gen, score)| (p, score))
            .collect();

        // Sort by score descending so ScoreOrdered merge gets the best proposal at index 0.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut valid_proposals = Vec::new();
        let mut valid_proposal_scores = Vec::new();
        let mut failed_proposals = Vec::new();
        for (proposal, score) in scored {
            if score > 0.0 {
                valid_proposals.push(proposal);
                valid_proposal_scores.push(score);
            } else {
                failed_proposals.push(proposal);
            }
        }

        Self {
            task_id,
            valid_proposals,
            valid_proposal_scores,
            failed_proposals,
            pruned_proposals: pruned,
        }
    }
}
