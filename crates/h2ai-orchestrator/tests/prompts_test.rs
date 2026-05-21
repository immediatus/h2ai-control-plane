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
use h2ai_orchestrator::prompts::resolve_prompt;
use h2ai_state::InMemoryStateBackend;

// ── resolve_prompt: no-NATS fallback path ────────────────────────────────────

/// When no NATS client is provided, resolve_prompt must return the default text as-is.
#[tokio::test]
async fn resolve_prompt_returns_default_when_no_nats() {
    let result = resolve_prompt(
        "adapter-x",
        "thinking_prompt",
        "fallback text",
        None::<&InMemoryStateBackend>,
    )
    .await;
    assert_eq!(result, "fallback text");
}

/// resolve_prompt treats the key "{adapter}/{prompt_key}" as a cache key.
/// Two different adapters with the same prompt_key must be independent.
#[tokio::test]
async fn resolve_prompt_different_adapters_are_independent() {
    let a = resolve_prompt(
        "adapter-a",
        "prompt1",
        "text-a",
        None::<&InMemoryStateBackend>,
    )
    .await;
    let b = resolve_prompt(
        "adapter-b",
        "prompt1",
        "text-b",
        None::<&InMemoryStateBackend>,
    )
    .await;
    assert_eq!(a, "text-a");
    assert_eq!(b, "text-b");
}

/// The default text may be empty — resolve_prompt must return it unchanged.
#[tokio::test]
async fn resolve_prompt_empty_default_is_preserved() {
    let result = resolve_prompt("adapter", "empty_key", "", None::<&InMemoryStateBackend>).await;
    assert_eq!(result, "");
}

/// A default text that contains template-like placeholders is returned verbatim.
#[tokio::test]
async fn resolve_prompt_preserves_placeholder_text() {
    let default = "Hello {name}, your task is {task_id}";
    let result = resolve_prompt(
        "adapter",
        "greeting",
        default,
        None::<&InMemoryStateBackend>,
    )
    .await;
    assert_eq!(result, default);
}

/// resolve_prompt with a long default is returned in full (no truncation).
#[tokio::test]
async fn resolve_prompt_long_default_returned_in_full() {
    let long = "x".repeat(4096);
    let result = resolve_prompt("adapter", "long_key", &long, None::<&InMemoryStateBackend>).await;
    assert_eq!(result.len(), 4096);
}

/// Calling resolve_prompt twice with the same key and default gives the same result.
#[tokio::test]
async fn resolve_prompt_idempotent_without_nats() {
    let key = "idempotent_key";
    let default = "stable default";
    let r1 = resolve_prompt("adapter", key, default, None::<&InMemoryStateBackend>).await;
    let r2 = resolve_prompt("adapter", key, default, None::<&InMemoryStateBackend>).await;
    assert_eq!(r1, r2);
}
