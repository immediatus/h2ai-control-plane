use h2ai_constraints::types::ConstraintDoc;
use h2ai_types::manifest::ExplorerSlotConfig;
use std::collections::HashSet;

/// Collect all unique domain tags from a constraint corpus, sorted.
#[must_use]
pub fn corpus_domain_tags(corpus: &[ConstraintDoc]) -> Vec<String> {
    let mut domains: HashSet<String> = HashSet::new();
    for doc in corpus {
        for domain in &doc.domains {
            domains.insert(domain.clone());
        }
    }
    let mut result: Vec<String> = domains.into_iter().collect();
    result.sort();
    result
}

/// Fraction of `corpus_domains` covered by the combined `constraint_domains` of all slots.
///
/// Returns `1.0` when `corpus_domains` is empty (no domains to cover).
/// Returns `0.0` when all slots have empty `constraint_domains`.
///
/// Matching is case-insensitive: `"Auth"` covers corpus tag `"auth"`. This is the
/// second line of defence against vocabulary mismatch; the primary fix is injecting
/// the corpus vocabulary into the STEP3 decomposition prompt so the LLM emits
/// verbatim strings in the first place.
#[must_use]
pub fn compute_coverage_score(slots: &[ExplorerSlotConfig], corpus_domains: &[String]) -> f64 {
    if corpus_domains.is_empty() {
        return 1.0;
    }
    let corpus: HashSet<String> = corpus_domains.iter().map(|s| s.to_lowercase()).collect();
    let covered: HashSet<String> = slots
        .iter()
        .flat_map(|s| s.constraint_domains.iter().map(|d| d.to_lowercase()))
        .collect();
    covered.intersection(&corpus).count() as f64 / corpus.len() as f64
}
