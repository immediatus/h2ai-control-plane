use h2ai_types::events::{BranchPrunedEvent, ProposalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use std::collections::{HashMap, HashSet};

/// A set of proposals keyed by explorer. Stores a generation counter and verification
/// score alongside each proposal. LUB rule: higher generation wins; ties broken by
/// higher score. This makes `ProposalSet` correct under TAO retry loops where a later
/// attempt may produce a lower verification score than an earlier one.
pub struct ProposalSet(HashMap<ExplorerId, (ProposalEvent, u64, f64)>);

impl ProposalSet {
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

    /// Join two proposal sets (CRDT merge).
    ///
    /// join(S₁, S₂) = S₁ ∪ S₂ with generation-first conflict resolution per explorer.
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

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return a reference to the stored `ProposalEvent` for `explorer_id`, if present.
    pub fn get(&self, explorer_id: &ExplorerId) -> Option<&ProposalEvent> {
        self.0.get(explorer_id).map(|(p, _gen, _score)| p)
    }
}

impl Default for ProposalSet {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SemilatticeResult {
    pub task_id: TaskId,
    /// Valid proposals sorted by verification score descending.
    /// First element = highest score = preferred by ScoreOrdered merge.
    pub valid_proposals: Vec<ProposalEvent>,
    pub pruned_proposals: Vec<BranchPrunedEvent>,
}

impl SemilatticeResult {
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

        Self {
            task_id,
            valid_proposals: scored.into_iter().map(|(p, _)| p).collect(),
            pruned_proposals: pruned,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use h2ai_types::config::AdapterKind;
    use h2ai_types::identity::{ExplorerId, TaskId};
    use h2ai_types::physics::TauValue;

    fn prop(text: &str) -> ProposalEvent {
        ProposalEvent {
            task_id: TaskId::new(),
            explorer_id: ExplorerId::new(),
            tau: TauValue::new(0.5).unwrap(),
            generation: 0,
            raw_output: text.into(),
            token_cost: 1,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn score_ordered_compile_sorts_by_score_descending() {
        let task_id = TaskId::new();
        let p_low = prop("low score proposal");
        let p_high = prop("high score proposal");
        let p_medium = prop("medium score proposal");

        let mut set = ProposalSet::new();
        set.insert_scored(p_low.clone(), 0.3);
        set.insert_scored(p_high.clone(), 0.9);
        set.insert_scored(p_medium.clone(), 0.6);

        let result = SemilatticeResult::compile(task_id, set, vec![]);
        assert_eq!(result.valid_proposals.len(), 3);
        assert_eq!(
            result.valid_proposals[0].raw_output, p_high.raw_output,
            "highest-scored proposal must be first"
        );
        assert_eq!(
            result.valid_proposals[2].raw_output, p_low.raw_output,
            "lowest-scored proposal must be last"
        );
    }

    #[test]
    fn insert_without_score_defaults_to_zero() {
        let task_id = TaskId::new();
        let mut set = ProposalSet::new();
        set.insert(prop("unscored proposal"));
        let result = SemilatticeResult::compile(task_id, set, vec![]);
        assert_eq!(result.valid_proposals.len(), 1);
    }

    fn prop_with_id(text: &str, id: ExplorerId) -> ProposalEvent {
        ProposalEvent {
            task_id: TaskId::new(),
            explorer_id: id,
            tau: TauValue::new(0.5).unwrap(),
            generation: 0,
            raw_output: text.into(),
            token_cost: 1,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn insert_scored_keeps_higher_score_for_same_explorer() {
        let task_id = TaskId::new();
        let explorer_id = ExplorerId::new();
        let low = prop_with_id("low score output", explorer_id.clone());
        let high = prop_with_id("high score output", explorer_id.clone());

        let mut set = ProposalSet::new();
        set.insert_scored(low, 0.3);
        set.insert_scored(high, 0.9);

        let result = SemilatticeResult::compile(task_id, set, vec![]);
        assert_eq!(result.valid_proposals.len(), 1, "one explorer → one slot");
        assert_eq!(
            result.valid_proposals[0].raw_output, "high score output",
            "LUB must keep higher-scored proposal"
        );
    }

    #[test]
    fn join_is_idempotent() {
        let task_id = TaskId::new();
        let explorer_id = ExplorerId::new();
        let p = prop_with_id("proposal text", explorer_id.clone());

        let mut s1 = ProposalSet::new();
        s1.insert_scored(p.clone(), 0.7);
        let mut s2 = ProposalSet::new();
        s2.insert_scored(p, 0.7);

        let joined = ProposalSet::join(s1, s2);
        let result = SemilatticeResult::compile(task_id, joined, vec![]);
        assert_eq!(
            result.valid_proposals.len(),
            1,
            "join(S, S) = S (idempotent)"
        );
    }
}
