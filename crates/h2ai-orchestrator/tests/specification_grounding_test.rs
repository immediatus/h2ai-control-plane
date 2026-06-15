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

// ── Constraint corpus grounding (SRANI anti-alignment fix) ────────────────────

#[test]
fn constraint_mandated_entity_not_flagged_when_spec_includes_constraint_text() {
    // Reproduces the SRANI anti-alignment: task description says nothing about Redis or
    // ClickHouse, but the constraint docs mandate them. When the call site in srani.rs
    // concatenates constraint text into the effective spec, these entities must not appear
    // in shared_ungrounded and CFI must remain 0.
    let task_description = "Build a billing system with idempotency guarantees";
    let constraint_text = "MUST use Redis atomic Lua EVAL for the idempotency key check. \
                           MUST use ClickHouse for the immutable audit log.";
    let effective_spec = format!("{task_description}\n{constraint_text}");

    let p1 = "Use Redis atomic Lua EVAL and ClickHouse for the audit log to satisfy constraints";
    let p2 = "Use Redis EVAL for idempotency and ClickHouse for audit trail as required";

    let result = check_specification_grounding(&effective_spec, &[p1, p2]).unwrap();
    assert_eq!(
        result.cfi, 0.0,
        "constraint-mandated entities must not be ungrounded when constraint text is in spec; \
         got CFI={}, ungrounded={:?}",
        result.cfi, result.shared_ungrounded
    );
    assert!(
        result.shared_ungrounded.is_empty(),
        "shared_ungrounded must be empty; got {:?}",
        result.shared_ungrounded
    );
}

#[test]
fn constraint_entities_flagged_without_constraint_text_in_spec() {
    // Confirms the pre-fix behaviour: without constraint text, constraint-mandated entities
    // ARE treated as ungrounded and CFI > 0. This test documents the defect that the
    // srani.rs effective_spec fix resolves.
    let task_description = "Build a billing system with idempotency guarantees";
    let p1 = "Use Redis atomic Lua EVAL and ClickHouse for the audit log to satisfy constraints";
    let p2 = "Use Redis EVAL for idempotency and ClickHouse for audit trail as required";

    let result = check_specification_grounding(task_description, &[p1, p2]).unwrap();
    assert!(
        result.cfi > 0.0,
        "without constraint text in spec, shared constraint-mandated entities must be flagged; \
         got CFI={}",
        result.cfi
    );
}

// ── manifest.context grounding (SRANI spec-boundary fix) ─────────────────────

#[test]
fn manifest_context_entities_not_flagged_when_context_in_spec() {
    // Reproduces the SRANI spec-boundary bug: manifest.description did not mention Redis/Kafka,
    // but manifest.context listed them as required infrastructure. When srani.rs now includes
    // manifest.context in the effective_spec, these entities must not be ungrounded.
    let task_description = "Design a billing quota enforcement mechanism";
    let manifest_context = "Infrastructure: Redis 7.2 (quota counters), Kafka (billing events), \
                            ClickHouse (audit log), CockroachDB (operational state)";
    let effective_spec = format!("{task_description}\n{manifest_context}");

    let p1 = "Use Redis Lua EVAL for atomicity, Kafka for billing events, ClickHouse for audit, CockroachDB for state";
    let p2 = "Use Redis atomic operations, Kafka pipeline, ClickHouse and CockroachDB for storage";

    let result = check_specification_grounding(&effective_spec, &[p1, p2]).unwrap();
    assert_eq!(
        result.cfi, 0.0,
        "manifest.context entities must not be ungrounded when context is in spec; \
         CFI={}, shared_ungrounded={:?}",
        result.cfi, result.shared_ungrounded
    );
    assert!(
        result.shared_ungrounded.is_empty(),
        "shared_ungrounded must be empty when all entities are in context; got {:?}",
        result.shared_ungrounded
    );
}

#[test]
fn manifest_context_entities_flagged_without_context_in_spec() {
    // Documents the pre-fix SRANI false-positive: manifest.context was excluded from
    // effective_spec, causing infrastructure entities to be treated as fabrications.
    let task_description = "Design a billing quota enforcement mechanism";
    let p1 = "Use Redis Lua EVAL for atomicity, Kafka for billing events, ClickHouse for audit, CockroachDB for state";
    let p2 = "Use Redis atomic operations, Kafka pipeline, ClickHouse and CockroachDB for storage";

    let result = check_specification_grounding(task_description, &[p1, p2]).unwrap();
    assert!(
        result.cfi > 0.0,
        "without manifest.context in spec, shared infrastructure entities must be ungrounded; \
         CFI={}",
        result.cfi
    );
    let has_infra_entity = result
        .shared_ungrounded
        .iter()
        .any(|e| matches!(e.as_str(), "Kafka" | "CockroachDB" | "Redis" | "ClickHouse"));
    assert!(
        has_infra_entity,
        "expected Redis/Kafka/ClickHouse/CockroachDB in shared_ungrounded; got {:?}",
        result.shared_ungrounded
    );
}
