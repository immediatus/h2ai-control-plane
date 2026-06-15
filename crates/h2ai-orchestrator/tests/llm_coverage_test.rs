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
use h2ai_constraints::types::{ConstraintDoc, ConstraintSeverity};
use h2ai_orchestrator::phases::llm_coverage;
use std::collections::HashSet;

fn make_hard(id: &str, domains: &[&str]) -> ConstraintDoc {
    let mut c = ConstraintDoc::new_llm_judge(id, "rubric");
    c.domains = domains.iter().map(|s| s.to_string()).collect();
    c
}

fn make_soft(id: &str, domains: &[&str]) -> ConstraintDoc {
    let mut c = ConstraintDoc::new_soft_llm_judge(id, "rubric");
    c.domains = domains.iter().map(|s| s.to_string()).collect();
    c
}

fn make_advisory(id: &str, domains: &[&str]) -> ConstraintDoc {
    let mut c = ConstraintDoc::new_llm_judge(id, "rubric");
    c.severity = ConstraintSeverity::Advisory;
    c.domains = domains.iter().map(|s| s.to_string()).collect();
    c
}

#[test]
fn no_survivors_returns_empty() {
    let corpus = vec![make_hard("C1", &["billing"])];
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 0,
        bypassed_ids: &HashSet::new(),
    });
    assert!(output.covered_domains.is_empty());
}

#[test]
fn empty_corpus_returns_empty() {
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &[],
        survivor_count: 5,
        bypassed_ids: &HashSet::new(),
    });
    assert!(output.covered_domains.is_empty());
}

#[test]
fn single_hard_constraint_covered() {
    let corpus = vec![make_hard("C1", &["billing"])];
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert_eq!(output.covered_domains, vec!["billing"]);
}

#[test]
fn hard_constraint_multiple_domains() {
    let corpus = vec![make_hard("C1", &["billing", "compliance"])];
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert_eq!(output.covered_domains, vec!["billing", "compliance"]);
}

#[test]
fn soft_constraint_excluded() {
    let corpus = vec![make_soft("C1", &["audit"])];
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert!(output.covered_domains.is_empty());
}

#[test]
fn advisory_constraint_excluded() {
    let corpus = vec![make_advisory("C1", &["perf"])];
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert!(output.covered_domains.is_empty());
}

#[test]
fn mixed_severities_only_hard_included() {
    let corpus = vec![
        make_hard("C1", &["billing"]),
        make_soft("C2", &["audit"]),
        make_advisory("C3", &["perf"]),
    ];
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert_eq!(output.covered_domains, vec!["billing"]);
}

#[test]
fn domains_deduplicated_across_two_hard() {
    let corpus = vec![
        make_hard("C1", &["billing"]),
        make_hard("C2", &["billing", "compliance"]),
    ];
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert_eq!(output.covered_domains, vec!["billing", "compliance"]);
}

#[test]
fn domains_sorted_alphabetically() {
    let corpus = vec![make_hard("C1", &["z", "a", "m"])];
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert_eq!(output.covered_domains, vec!["a", "m", "z"]);
}

#[test]
fn hard_constraint_empty_domains_no_crash() {
    let corpus = vec![make_hard("C1", &[])];
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    assert!(output.covered_domains.is_empty());
}

#[test]
fn large_survivor_count_same_result_as_one() {
    let corpus = vec![make_hard("C1", &["billing"])];
    let out1 = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &HashSet::new(),
    });
    let out100 = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 100,
        bypassed_ids: &HashSet::new(),
    });
    assert_eq!(out1.covered_domains, out100.covered_domains);
}

#[test]
fn bypassed_hard_domain_not_covered() {
    // C1 is not bypassed → billing covered; C2 is bypassed → audit NOT covered.
    let corpus = vec![make_hard("C1", &["billing"]), make_hard("C2", &["audit"])];
    let bypassed: HashSet<String> = ["C2".to_string()].into_iter().collect();
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &bypassed,
    });
    assert_eq!(output.covered_domains, vec!["billing"]);
}

#[test]
fn all_hard_bypassed_returns_empty() {
    // The only Hard constraint is bypassed → no domains covered despite survivor.
    let corpus = vec![make_hard("C1", &["billing"])];
    let bypassed: HashSet<String> = ["C1".to_string()].into_iter().collect();
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &bypassed,
    });
    assert!(output.covered_domains.is_empty());
}

#[test]
fn bypassed_soft_constraint_irrelevant() {
    // C2 is Soft and in bypassed_ids — it was always excluded; billing from Hard C1 still covered.
    let corpus = vec![make_hard("C1", &["billing"]), make_soft("C2", &["audit"])];
    let bypassed: HashSet<String> = ["C2".to_string()].into_iter().collect();
    let output = llm_coverage::run(llm_coverage::Input {
        corpus: &corpus,
        survivor_count: 1,
        bypassed_ids: &bypassed,
    });
    assert_eq!(output.covered_domains, vec!["billing"]);
}
