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
use h2ai_orchestrator::specification_grounding::{
    check_specification_grounding, extract_arch_nouns,
};

// ── extract_arch_nouns ─────────────────────────────────────────────────────────

#[test]
fn camelcase_compound_extracted() {
    let nouns = extract_arch_nouns("Use ZooKeeper for coordination and ElasticSearch for indexing");
    assert!(nouns.contains("ZooKeeper"), "got: {nouns:?}");
    assert!(nouns.contains("ElasticSearch"), "got: {nouns:?}");
}

#[test]
fn branded_suffix_extracted() {
    let nouns =
        extract_arch_nouns("Store state in CockroachDB and use RedisLock for mutual exclusion");
    assert!(nouns.contains("CockroachDB"), "got: {nouns:?}");
    assert!(nouns.contains("RedisLock"), "got: {nouns:?}");
}

#[test]
fn lexicon_terms_extracted() {
    let nouns = extract_arch_nouns("Use etcd for service discovery alongside Consul and Vault");
    assert!(nouns.contains("etcd"), "got: {nouns:?}");
    assert!(nouns.contains("Consul"), "got: {nouns:?}");
    assert!(nouns.contains("Vault"), "got: {nouns:?}");
}

#[test]
fn plain_english_not_extracted() {
    let nouns = extract_arch_nouns("Use a cache and a queue for better performance");
    assert!(!nouns.contains("cache"), "got: {nouns:?}");
    assert!(!nouns.contains("queue"), "got: {nouns:?}");
}

// ── check_specification_grounding ─────────────────────────────────────────────

#[test]
fn fewer_than_two_proposals_returns_none() {
    let result = check_specification_grounding("use Redis", &["use Redis and ZooKeeper"]);
    assert!(result.is_none());
}

#[test]
fn redis_in_spec_is_not_ungrounded() {
    let spec = "Use Redis for caching and Kafka for messaging";
    let p1 = "Use Redis for caching and Kafka for messaging with retry logic";
    let p2 = "Use Redis and Kafka with exponential backoff";
    let result = check_specification_grounding(spec, &[p1, p2]).unwrap();
    assert_eq!(result.cfi, 0.0, "Redis is grounded; CFI must be 0");
    assert!(result.shared_ungrounded.is_empty());
}

#[test]
fn cockroachdb_not_in_spec_gives_cfi_one() {
    let spec = "Use Redis for caching and Kafka for event log";
    let p1 = "Use Redis and Kafka. CockroachDB advisory locks prevent double-spend.";
    let p2 = "Use Redis and Kafka. CockroachDB provides distributed locking.";
    let result = check_specification_grounding(spec, &[p1, p2]).unwrap();
    assert!(
        (result.cfi - 1.0).abs() < 1e-9,
        "expected CFI=1.0, got {}",
        result.cfi
    );
    assert!(
        result.shared_ungrounded.iter().any(|e| e == "CockroachDB"),
        "CockroachDB must be in shared_ungrounded; got {:?}",
        result.shared_ungrounded
    );
}

#[test]
fn partial_overlap_gives_cfi_between_zero_and_one() {
    // Ungrounded(p1) = {CockroachDB, YugabyteDB}, Ungrounded(p2) = {CockroachDB}
    // CFI = 1 / max(2,1) = 0.5
    let spec = "Use Redis for state";
    let p1 = "Use Redis. CockroachDB and YugabyteDB provide ACID guarantees.";
    let p2 = "Use Redis. CockroachDB provides distributed transactions.";
    let result = check_specification_grounding(spec, &[p1, p2]).unwrap();
    assert!(
        result.cfi > 0.0 && result.cfi < 1.0,
        "expected 0 < CFI < 1, got {}",
        result.cfi
    );
    assert!(
        result.shared_ungrounded.iter().any(|e| e == "CockroachDB"),
        "CockroachDB must be in shared_ungrounded"
    );
}

#[test]
fn no_shared_ungrounded_gives_cfi_zero() {
    let spec = "Use Redis and Kafka";
    let p1 = "Use Redis and Kafka with ZooKeeper for coordination";
    let p2 = "Use Redis and Kafka with Consul for service discovery";
    let result = check_specification_grounding(spec, &[p1, p2]).unwrap();
    assert_eq!(result.cfi, 0.0, "no shared ungrounded; CFI must be 0");
}

#[test]
fn proposal_count_matches_input_length() {
    let spec = "Use Kafka";
    let p1 = "Kafka and CockroachDB";
    let p2 = "Kafka and CockroachDB";
    let p3 = "Kafka and CockroachDB";
    let result = check_specification_grounding(spec, &[p1, p2, p3]).unwrap();
    assert_eq!(result.proposal_count, 3);
}
