use super::{Gap, GapKind, GapSeverity, GapSource};

pub(crate) const UNCERTAINTY_KEYWORDS: &[&str] =
    &["unsettled", "best-effort basis", "rapidly evolving"];

/// Seed `UncertainDomain` gaps from the task context string.
///
/// When the task author explicitly flags a domain as unsettled (e.g. "UK and US AI law
/// is unsettled"), one `UncertainDomain` gap is created per occurrence of a known
/// uncertainty marker.  These gaps have kind `UncertainDomain` so `MicroExplorerResolver`
/// cannot close them — they always remain as `RequiresReview` provisions in the
/// ProvenanceMap, reflecting genuine epistemic uncertainty in the domain.
///
/// This is a pure function: no I/O, no LLM calls, fully deterministic.
pub fn seed_uncertainty_gaps(task_context: &str) -> Vec<Gap> {
    if task_context.is_empty() {
        return vec![];
    }
    let lower = task_context.to_lowercase();
    let mut gaps = Vec::new();
    for (i, kw) in UNCERTAINTY_KEYWORDS
        .iter()
        .enumerate()
        .filter(|(_, kw)| lower.contains(*kw))
    {
        gaps.push(Gap {
            id: format!("g-uncertain-ctx-{i}"),
            kind: GapKind::UncertainDomain,
            severity: GapSeverity::Medium,
            description: format!("Task context flags domain as uncertain (keyword: \"{kw}\")"),
            affected_provisions: vec![format!("context_uncertainty_{i}")],
            depends_on: None,
            source: GapSource::TaskContextSeeding,
        });
    }
    gaps
}
