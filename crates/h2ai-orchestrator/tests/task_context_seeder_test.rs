use h2ai_orchestrator::gap_checkers::task_context_seeder::seed_uncertainty_gaps;
use h2ai_orchestrator::gap_checkers::{GapKind, GapSource};

#[test]
fn empty_context_produces_no_gaps() {
    assert!(seed_uncertainty_gaps("").is_empty());
}

#[test]
fn context_without_keywords_produces_no_gaps() {
    let ctx = "All provisions are well-established and mandatory.";
    assert!(seed_uncertainty_gaps(ctx).is_empty());
}

#[test]
fn unsettled_keyword_produces_uncertain_domain_gap() {
    let ctx = "UK and US state AI law is unsettled and evolving.";
    let gaps = seed_uncertainty_gaps(ctx);
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].kind, GapKind::UncertainDomain);
    assert_eq!(gaps[0].source, GapSource::TaskContextSeeding);
    assert_eq!(gaps[0].id, "g-uncertain-ctx-0");
}

#[test]
fn matching_is_case_insensitive() {
    let ctx = "The law is UNSETTLED in this jurisdiction.";
    let gaps = seed_uncertainty_gaps(ctx);
    assert_eq!(gaps.len(), 1, "case-insensitive match on UNSETTLED");
}

#[test]
fn multiple_keywords_produce_one_gap_each() {
    let ctx = "Law is unsettled. Assessment is on a best-effort basis.";
    let gaps = seed_uncertainty_gaps(ctx);
    assert_eq!(gaps.len(), 2);
    let ids: Vec<&str> = gaps.iter().map(|g| g.id.as_str()).collect();
    assert!(ids.contains(&"g-uncertain-ctx-0"));
    assert!(ids.contains(&"g-uncertain-ctx-1"));
}

#[test]
fn repeated_keyword_counts_once_per_keyword_not_per_occurrence() {
    // "unsettled" appears twice but is one entry in UNCERTAINTY_KEYWORDS
    let ctx = "Law is unsettled. Regulations are also unsettled.";
    let gaps = seed_uncertainty_gaps(ctx);
    assert_eq!(gaps.len(), 1, "one gap per keyword, not per occurrence");
}

#[test]
fn each_gap_has_uncertain_domain_kind_and_task_context_seeding_source() {
    let ctx = "The domain is rapidly evolving with unsettled guidance.";
    let gaps = seed_uncertainty_gaps(ctx);
    assert!(!gaps.is_empty());
    for g in &gaps {
        assert_eq!(g.kind, GapKind::UncertainDomain);
        assert_eq!(g.source, GapSource::TaskContextSeeding);
    }
}
