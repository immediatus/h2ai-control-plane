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
use h2ai_state::weiszfeld::weiszfeld_select;

#[test]
fn weiszfeld_selects_honest_with_one_byzantine() {
    // 4 honest embeddings near [1,0,...,0], 1 Byzantine at [-1,0,...,0]
    let dim = 8;
    let honest: Vec<Vec<f32>> = (0..4)
        .map(|i| {
            let mut v = vec![0.0f32; dim];
            v[0] = 0.05f32.mul_add(-(i as f32), 1.0); // slightly varied but near [1,0,...]
            v[1] = 0.05 * i as f32;
            v
        })
        .collect();
    let byzantine = vec![-1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let mut embeddings = honest;
    embeddings.push(byzantine);

    let selected = weiszfeld_select(&embeddings, 20);
    assert!(
        selected < 4,
        "Should select an honest proposal (index 0-3), got {selected}"
    );
}

#[test]
fn weiszfeld_single_returns_zero() {
    let embs = vec![vec![1.0f32, 0.0, 0.0]];
    assert_eq!(weiszfeld_select(&embs, 20), 0);
}

#[test]
fn weiszfeld_empty_returns_zero() {
    let embs: Vec<Vec<f32>> = vec![];
    assert_eq!(weiszfeld_select(&embs, 20), 0);
}

#[test]
fn weiszfeld_zero_dim_embeddings_returns_zero() {
    // dim == 0 guard: all inner vectors are empty → return 0 immediately
    let embs: Vec<Vec<f32>> = vec![vec![], vec![]];
    assert_eq!(weiszfeld_select(&embs, 20), 0);
}
