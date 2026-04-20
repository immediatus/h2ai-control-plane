use h2ai_types::events::ProposalEvent;

pub struct BftConsensus;

impl BftConsensus {
    pub fn resolve(proposals: &[ProposalEvent]) -> Option<&ProposalEvent> {
        proposals.iter().min_by_key(|p| p.token_cost)
    }
}
