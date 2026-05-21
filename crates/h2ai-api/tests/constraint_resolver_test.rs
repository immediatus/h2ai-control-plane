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
use h2ai_constraints::source::FsConstraintStore;
use h2ai_constraints::types::{ConstraintMeta, ConstraintSeverity, PredicateKind};
use h2ai_constraints::wiki::WikiCache;
use std::fs;
use tempfile::TempDir;

#[test]
fn wiki_cache_resolve_returns_matching_ids() {
    let mut cache = WikiCache::default();
    cache
        .context_map
        .insert("eu_data".into(), vec!["GDPR-001".into()]);
    cache.metas.insert(
        "GDPR-001".into(),
        ConstraintMeta {
            id: "GDPR-001".into(),
            summary: "Minimize personal data.".into(),
            severity: ConstraintSeverity::Hard { threshold: 0.8 },
            predicate_kind: PredicateKind::LlmJudge,
            domains: vec!["eu_data".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            payload_version: "v1".into(),
            inline_predicate: None,
            source: None,
            last_updated_ms: None,
        },
    );

    let resolved = cache.resolve(&["eu_data".to_string()], &[]);
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].id, "GDPR-001");
    assert_eq!(resolved[0].predicate_kind, PredicateKind::LlmJudge);
}

#[tokio::test]
async fn fs_constraint_resolver_loads_and_resolves_by_id() {
    use h2ai_constraints::resolver::ConstraintResolver;
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    fs::write(
        dir.path().join("ADR-001.yaml"),
        "id: ADR-001\ntitle: Cite Source\nseverity: hard\ncriteria:\n  pass: Cites a source reference\n  fail: No source cited\n",
    )
    .unwrap();

    let (index, store) = FsConstraintStore::load(dir.path()).unwrap();
    let resolver = ConstraintResolver::new(Arc::new(index), Arc::new(store));
    let docs = resolver.resolve(&["ADR-001".to_string()], &[], "").await;

    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].id, "ADR-001");
}
