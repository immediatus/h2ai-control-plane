use super::{Gap, GapKind, GapSeverity, GapSource};
use std::collections::HashSet;

/// Pure function. Extracts quality gaps from the pruned-proposal reasons emitted by
/// `SelectionResolvedEvent.pruned_proposals: Vec<(ExplorerId, String)>`.
///
/// Each pruned reason string becomes a `Gap` with `kind = MissingProvision` and
/// `source = SelectionPruning`. Duplicate reasons (same text across multiple explorers)
/// are deduplicated — only one gap per unique description.
///
/// Gap IDs are deterministic: "g{1-based-index}" in order of first appearance.
pub fn extract_gaps_from_pruned(pruned: &[(String, String)]) -> Vec<Gap> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut gaps = Vec::new();

    for (_explorer_id, reason) in pruned {
        let description = reason.trim().to_string();
        if description.is_empty() || !seen.insert(description.clone()) {
            continue;
        }
        let id = format!("g{}", gaps.len() + 1);
        gaps.push(Gap {
            id,
            kind: GapKind::MissingProvision,
            severity: GapSeverity::High,
            description,
            affected_provisions: vec![],
            depends_on: None,
            source: GapSource::SelectionPruning,
        });
    }

    gaps
}
