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
use h2ai_orchestrator::correlated_hallucination::compute_cv;

#[test]
fn single_proposal_returns_none() {
    assert!(compute_cv(&["some text"]).is_none());
}

#[test]
fn empty_returns_none() {
    assert!(compute_cv(&[]).is_none());
}

#[test]
fn identical_proposals_return_zero_cv() {
    let s = compute_cv(&[
        "the quick brown fox",
        "the quick brown fox",
        "the quick brown fox",
    ])
    .unwrap();
    assert_eq!(s.cv, 0.0);
    assert_eq!(s.mean_jaccard_distance, 0.0);
}

#[test]
fn diverse_proposals_return_high_mean_distance() {
    let proposals = &[
        "quantum entanglement photon polarization measurement",
        "sql database transaction isolation deadlock prevention",
        "rust borrow checker lifetime ownership memory safety",
        "neural network backpropagation gradient descent loss",
    ];
    let s = compute_cv(proposals).unwrap();
    assert!(
        s.mean_jaccard_distance > 0.5,
        "diverse proposals should have large distances, got {}",
        s.mean_jaccard_distance
    );
}

#[test]
fn similar_proposals_return_low_mean_distance() {
    let proposals = &[
        "stateless JWT authentication token validation",
        "stateless JWT authentication bearer token",
        "stateless token based JWT authentication scheme",
    ];
    let s = compute_cv(proposals).unwrap();
    assert!(
        s.mean_jaccard_distance < 0.6,
        "similar proposals should have small distances: got {}",
        s.mean_jaccard_distance
    );
}

#[test]
fn two_diverse_proposals_returns_none() {
    // N=2 with non-zero distance: single-point distribution → CV meaningless → None
    assert!(compute_cv(&["foo bar baz", "foo bar qux"]).is_none());
}

#[test]
fn two_identical_proposals_returns_zero_cv() {
    // N=2 identical: mean=0 → definite correlation signal
    let s = compute_cv(&["foo bar baz", "foo bar baz"]).unwrap();
    assert_eq!(s.cv, 0.0);
    assert_eq!(s.mean_jaccard_distance, 0.0);
}
