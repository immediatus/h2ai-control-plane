use h2ai_constraints::types::ConstraintDoc;
use h2ai_types::manifest::ExplorerSlotConfig;
use std::collections::HashSet;

/// Collect all unique domain tags from a constraint corpus, sorted.
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

#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_constraints::types::ConstraintDoc;
    use h2ai_types::manifest::ExplorerSlotConfig;

    fn doc_with_domains(id: &str, domains: &[&str]) -> ConstraintDoc {
        let mut doc = ConstraintDoc::new_llm_judge(id, "rule");
        doc.domains = domains.iter().map(|s| s.to_string()).collect();
        doc
    }

    fn slot_with_domains(domains: &[&str]) -> ExplorerSlotConfig {
        ExplorerSlotConfig {
            constraint_domains: domains.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn empty_corpus_returns_full_coverage() {
        let score = compute_coverage_score(&[slot_with_domains(&[])], &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn no_slot_domains_returns_zero() {
        let corpus = vec!["security".to_string(), "performance".to_string()];
        let score = compute_coverage_score(&[slot_with_domains(&[])], &corpus);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn full_coverage() {
        let corpus = vec!["security".to_string(), "performance".to_string()];
        let slots = vec![
            slot_with_domains(&["security"]),
            slot_with_domains(&["performance"]),
        ];
        assert_eq!(compute_coverage_score(&slots, &corpus), 1.0);
    }

    #[test]
    fn partial_coverage_half() {
        let corpus = vec![
            "security".to_string(),
            "performance".to_string(),
            "correctness".to_string(),
            "data".to_string(),
        ];
        let slots = vec![slot_with_domains(&["security", "performance"])];
        let score = compute_coverage_score(&slots, &corpus);
        assert!((score - 0.5).abs() < 1e-9, "expected 0.5, got {score}");
    }

    #[test]
    fn corpus_domain_tags_deduplicates() {
        let corpus = vec![
            doc_with_domains("A", &["security", "auth"]),
            doc_with_domains("B", &["security", "performance"]),
            doc_with_domains("C", &[]),
        ];
        let tags = corpus_domain_tags(&corpus);
        assert_eq!(tags.len(), 3, "expected 3 unique tags, got {tags:?}");
        assert!(tags.contains(&"security".to_string()));
        assert!(tags.contains(&"auth".to_string()));
        assert!(tags.contains(&"performance".to_string()));
    }

    #[test]
    fn coverage_ignores_unknown_slot_domains() {
        let corpus = vec!["security".to_string()];
        let slots = vec![slot_with_domains(&["security", "unknown"])];
        assert_eq!(compute_coverage_score(&slots, &corpus), 1.0);
    }

    #[test]
    fn coverage_is_case_insensitive() {
        // Corpus uses lowercase; LLM might emit "Auth" or "PERFORMANCE".
        let corpus = vec!["auth".to_string(), "performance".to_string()];
        let slots = vec![
            slot_with_domains(&["Auth"]),
            slot_with_domains(&["PERFORMANCE"]),
        ];
        assert_eq!(
            compute_coverage_score(&slots, &corpus),
            1.0,
            "uppercase slot domain must match lowercase corpus tag"
        );
    }

    #[test]
    fn coverage_case_insensitive_partial() {
        let corpus = vec!["auth".to_string(), "latency".to_string()];
        let slots = vec![slot_with_domains(&["AUTH"])];
        let score = compute_coverage_score(&slots, &corpus);
        assert!(
            (score - 0.5).abs() < 1e-9,
            "only one of two corpus tags covered; expected 0.5, got {score}"
        );
    }
}
