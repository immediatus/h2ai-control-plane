#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use h2ai_constraints::types::ConstraintDoc;
use h2ai_orchestrator::domain_coverage::{compute_coverage_score, corpus_domain_tags};
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
