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
use h2ai_api::tenant_registry::TenantRegistry;
use h2ai_config::H2AIConfig;
use h2ai_types::identity::TenantId;
use std::sync::Arc;

#[test]
fn two_tenant_ids_get_separate_state() {
    let cfg = H2AIConfig::default();
    let registry = TenantRegistry::new();
    let s1 = registry.get_or_create(&TenantId::from("acme"), &cfg);
    let s2 = registry.get_or_create(&TenantId::from("beta"), &cfg);
    assert!(!Arc::ptr_eq(&s1, &s2));
}

#[test]
fn same_tenant_id_returns_same_arc() {
    let cfg = H2AIConfig::default();
    let registry = TenantRegistry::new();
    let t = TenantId::from("acme");
    let s1 = registry.get_or_create(&t, &cfg);
    let s2 = registry.get_or_create(&t, &cfg);
    assert!(Arc::ptr_eq(&s1, &s2));
}

#[test]
fn concurrent_first_access_yields_same_arc() {
    use std::thread;
    let cfg = H2AIConfig::default();
    let registry = Arc::new(TenantRegistry::new());
    let t = TenantId::from("race-tenant");
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let r = Arc::clone(&registry);
            let tid = t.clone();
            let c = cfg.clone();
            thread::spawn(move || r.get_or_create(&tid, &c))
        })
        .collect();
    let arcs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // All 8 threads must get back the same Arc (same allocation).
    for arc in &arcs {
        assert!(Arc::ptr_eq(&arcs[0], arc));
    }
}

#[test]
fn default_tenant_is_distinct_from_named_tenant() {
    let cfg = H2AIConfig::default();
    let registry = TenantRegistry::new();
    let s_default = registry.get_or_create(&TenantId::default_tenant(), &cfg);
    let s_named = registry.get_or_create(&TenantId::from("acme"), &cfg);
    assert!(!Arc::ptr_eq(&s_default, &s_named));
}
