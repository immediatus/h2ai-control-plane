use h2ai_types::events::{BranchPrunedEvent, ProposalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use std::collections::{HashMap, HashSet};

pub struct ProposalSet(HashMap<ExplorerId, ProposalEvent>);

impl ProposalSet {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn insert(&mut self, proposal: ProposalEvent) {
        self.0
            .entry(proposal.explorer_id.clone())
            .or_insert(proposal);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Default for ProposalSet {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SemilatticeResult {
    pub task_id: TaskId,
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

        let valid_proposals = proposals
            .0
            .into_values()
            .filter(|p| !pruned_ids.contains(&p.explorer_id))
            .collect();

        Self {
            task_id,
            valid_proposals,
            pruned_proposals: pruned,
        }
    }
}
