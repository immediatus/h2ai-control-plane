use h2ai_context::jaccard::{jaccard, tokenize};
use h2ai_types::events::ProposalEvent;

/// Returns `true` if every pairwise Jaccard similarity between proposal outputs
/// meets or exceeds `threshold`. Fewer than 2 proposals always returns `false`.
/// A threshold of `1.0` or greater effectively disables the gate (no finite
/// Jaccard value can exceed 1.0, so this is always `false`).
pub fn is_uniform(proposals: &[ProposalEvent], threshold: f64) -> bool {
    if proposals.len() < 2 {
        return false;
    }
    // threshold >= 1.0 means "disabled" — Jaccard is in [0, 1] so it can never
    // strictly exceed 1.0; treating this as "never uniform" preserves backward
    // compatibility for configs that have not opted in to the gate.
    if threshold >= 1.0 {
        return false;
    }
    let token_sets: Vec<_> = proposals.iter().map(|p| tokenize(&p.raw_output)).collect();
    for i in 0..token_sets.len() {
        for j in (i + 1)..token_sets.len() {
            if jaccard(&token_sets[i], &token_sets[j]) < threshold {
                return false;
            }
        }
    }
    true
}
