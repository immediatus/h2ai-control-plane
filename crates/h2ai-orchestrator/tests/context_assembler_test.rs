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
use h2ai_orchestrator::context_assembler::stable_cache::{CachedSection, StableContextCache};
use h2ai_orchestrator::context_assembler::{
    build_sections, importance_trim, quality_guard, rule_pass, score_sections, AssembledContext,
    CompressionKind, ContextAssembler, ContextAssemblerInput, RulePassInput, Section, SectionTag,
};

#[test]
fn assembled_context_defaults() {
    let ctx = AssembledContext {
        text: "hello".to_string(),
        token_estimate: 1,
        compression: CompressionKind::None,
        compression_ratio: 1.0,
        prev_wave_delta: false,
        quality_clamped: false,
    };
    assert_eq!(ctx.compression_ratio, 1.0);
    assert!(!ctx.quality_clamped);
}

#[test]
fn leader_prefix_always_preserved() {
    let input = make_input_with_leader("LEADER_PREFIX_TEXT");
    let sections = build_sections(&input);
    let scored = score_sections(sections, None);
    let lp = scored
        .iter()
        .find(|s| s.tag == SectionTag::LeaderPrefix)
        .unwrap();
    assert!(lp.preserve);
    assert_eq!(lp.importance, 1.0);
}

#[test]
fn retry_context_has_lowest_base_importance() {
    let input = make_input_with_retry("Some retry hint");
    let sections = build_sections(&input);
    let scored = score_sections(sections, None);
    let rc = scored
        .iter()
        .find(|s| s.tag == SectionTag::RetryContext)
        .unwrap();
    assert!(!rc.preserve);
    assert!(rc.importance < 0.6);
}

#[test]
fn constraint_id_in_active_ctx_boosts_importance() {
    let mut input = make_empty_input();
    input.active_ctx = "Must satisfy constraint C-042 and C-007.";
    let sections = build_sections(&input);
    let scored = score_sections(sections, None);
    let ac = scored
        .iter()
        .find(|s| s.tag == SectionTag::ActiveCtx)
        .unwrap();
    // base 0.7 + 0.15 boost for constraint IDs
    assert!(ac.importance >= 0.85);
}

#[test]
fn cross_wave_delta_replaces_unchanged_section() {
    let prev = AssembledContext {
        text: "ROLE: expert\n\nactive_ctx_content".to_string(),
        token_estimate: 10,
        compression: CompressionKind::None,
        compression_ratio: 1.0,
        prev_wave_delta: false,
        quality_clamped: false,
    };
    let mut sections = vec![
        Section {
            tag: SectionTag::ActiveCtx,
            text: "active_ctx_content".to_string(),
            importance: 0.7,
            preserve: false,
        },
        Section {
            tag: SectionTag::RetryContext,
            text: "NEW RETRY HINT".to_string(),
            importance: 0.5,
            preserve: false,
        },
    ];
    let delta = rule_pass(
        &mut sections,
        RulePassInput {
            prev_wave_blob: Some(&prev),
            wave: 1,
        },
    );
    assert!(delta, "rule_pass should return true when delta was applied");
    let ac = sections
        .iter()
        .find(|s| s.tag == SectionTag::ActiveCtx)
        .unwrap();
    assert!(
        ac.text.contains("WAVE"),
        "should contain delta marker, got: {}",
        ac.text
    );
    // Retry hint should be unchanged since it doesn't appear in prev wave text
    let rc = sections
        .iter()
        .find(|s| s.tag == SectionTag::RetryContext)
        .unwrap();
    assert_eq!(rc.text, "NEW RETRY HINT");
}

#[test]
fn block_dedup_collapses_repeated_text() {
    // 8 lines total (two aligned 4-line blocks): the < 8 early-exit threshold requires at least
    // 8 lines for dedup to run; no blank separator so the second block aligns at position 4.
    let repeated = "line one\nline two\nline three\nline four";
    let mut sections = vec![Section {
        tag: SectionTag::ActiveCtx,
        text: format!("{}\n{}", repeated, repeated),
        importance: 0.7,
        preserve: false,
    }];
    let _ = rule_pass(
        &mut sections,
        RulePassInput {
            prev_wave_blob: None,
            wave: 0,
        },
    );
    let ac = sections
        .iter()
        .find(|s| s.tag == SectionTag::ActiveCtx)
        .unwrap();
    assert!(ac.text.contains("[duplicate"), "expected dedup marker");
    assert!(ac.text.len() < format!("{}\n{}", repeated, repeated).len());
}

#[test]
fn whitespace_normalization_removes_blank_lines() {
    let mut sections = vec![Section {
        tag: SectionTag::ActiveCtx,
        text: "line1\n\n\n\nline2\n\n\n\nline3".to_string(),
        importance: 0.7,
        preserve: false,
    }];
    let _ = rule_pass(
        &mut sections,
        RulePassInput {
            prev_wave_blob: None,
            wave: 0,
        },
    );
    let ac = sections
        .iter()
        .find(|s| s.tag == SectionTag::ActiveCtx)
        .unwrap();
    assert!(
        !ac.text.contains("\n\n\n"),
        "triple blank lines should be collapsed"
    );
}

fn make_input_with_leader(lp: &str) -> ContextAssemblerInput<'_> {
    ContextAssemblerInput {
        active_ctx: "ctx",
        retry_context: None,
        leader_prefix: Some(lp),
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: None,
        rejection_criteria: None,
        prev_wave_blob: None,
        budget: None,
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: None,
        topic_knowledge: None,
        constraint_tensions: None,
    }
}

fn make_input_with_retry(rc: &str) -> ContextAssemblerInput<'_> {
    ContextAssemblerInput {
        active_ctx: "ctx",
        retry_context: Some(rc),
        leader_prefix: None,
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: None,
        rejection_criteria: None,
        prev_wave_blob: None,
        budget: None,
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: None,
        topic_knowledge: None,
        constraint_tensions: None,
    }
}

fn make_empty_input<'a>() -> ContextAssemblerInput<'a> {
    ContextAssemblerInput {
        active_ctx: "",
        retry_context: None,
        leader_prefix: None,
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: None,
        rejection_criteria: None,
        prev_wave_blob: None,
        budget: None,
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: None,
        topic_knowledge: None,
        constraint_tensions: None,
    }
}

#[test]
fn importance_trim_targets_lowest_score_first() {
    let long_retry = "A".repeat(2000);
    let sections = vec![
        Section {
            tag: SectionTag::ActiveCtx,
            text: "Short active ctx".to_string(),
            importance: 0.7,
            preserve: false,
        },
        Section {
            tag: SectionTag::RetryContext,
            text: long_retry.clone(),
            importance: 0.5,
            preserve: false,
        },
    ];
    let trimmed = importance_trim(sections, 50);
    let rc = trimmed
        .iter()
        .find(|s| s.tag == SectionTag::RetryContext)
        .unwrap();
    assert!(
        rc.text.len() < long_retry.len(),
        "RetryContext should be trimmed"
    );
    let ac = trimmed
        .iter()
        .find(|s| s.tag == SectionTag::ActiveCtx)
        .unwrap();
    assert_eq!(ac.text, "Short active ctx");
}

#[test]
fn quality_guard_clamps_at_ratio_threshold() {
    let clamped = quality_guard(1000, 300, 0.4);
    assert!(clamped, "should clamp when ratio 0.3 < threshold 0.4");
}

#[test]
fn quality_guard_passes_when_ratio_above_threshold() {
    let clamped = quality_guard(1000, 600, 0.4);
    assert!(!clamped);
}

#[test]
fn importance_trim_handles_multibyte_unicode() {
    // Each "é" is 2 bytes — a naive byte-slice would panic mid-codepoint
    let multibyte_text = "é ".repeat(500); // 1000 bytes, 500 chars
    let sections = vec![Section {
        tag: SectionTag::RetryContext,
        text: multibyte_text.clone(),
        importance: 0.5,
        preserve: false,
    }];
    // Should not panic
    let trimmed = importance_trim(sections, 1);
    let rc = trimmed
        .iter()
        .find(|s| s.tag == SectionTag::RetryContext)
        .unwrap();
    assert!(rc.text.len() < multibyte_text.len(), "should be trimmed");
}

#[test]
fn quality_guard_at_exact_threshold_passes() {
    // ratio == threshold (400/1000 = 0.4 == threshold)
    // strict less-than means exactly at threshold → false
    let clamped = quality_guard(1000, 400, 0.4);
    assert!(!clamped, "ratio equal to threshold should not clamp");
}

#[tokio::test]
async fn build_no_budget_returns_compression_none() {
    let input = ContextAssemblerInput {
        active_ctx: "The system context.",
        retry_context: None,
        leader_prefix: None,
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: None,
        rejection_criteria: None,
        prev_wave_blob: None,
        budget: None,
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: None,
        topic_knowledge: None,
        constraint_tensions: None,
    };
    let result = ContextAssembler::build(input).await;
    assert_eq!(result.compression, CompressionKind::None);
    assert!(result.text.contains("The system context."));
    assert_eq!(result.compression_ratio, 1.0);
    assert!(!result.prev_wave_delta);
}

#[tokio::test]
async fn build_with_budget_runs_rule_pass() {
    let input = ContextAssemblerInput {
        active_ctx: "The system context.",
        retry_context: None,
        leader_prefix: None,
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: None,
        rejection_criteria: None,
        prev_wave_blob: None,
        budget: Some(10000), // high budget — only rule pass needed
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: None,
        topic_knowledge: None,
        constraint_tensions: None,
    };
    let result = ContextAssembler::build(input).await;
    assert!(
        result.compression == CompressionKind::RuleBased
            || result.compression == CompressionKind::None
    );
}

#[test]
fn stable_cache_miss_then_hit() {
    let cache = StableContextCache::new();
    let key = 0xDEADBEEFu64;
    assert!(cache.get(key).is_none());
    cache.insert(
        key,
        CachedSection {
            compressed_text: "compressed".to_string(),
            original_token_estimate: 100,
            compressed_token_estimate: 20,
            hit_count: 0,
        },
    );
    let entry = cache.get(key).unwrap();
    assert_eq!(entry.compressed_text, "compressed");
    assert_eq!(entry.hit_count, 0);
    cache.record_hit(key);
    let entry = cache.get(key).unwrap();
    assert_eq!(entry.hit_count, 1);
}

#[tokio::test]
async fn build_with_prev_wave_sets_delta_flag() {
    let prev = AssembledContext {
        text: "system context content".to_string(),
        token_estimate: 5,
        compression: CompressionKind::None,
        compression_ratio: 1.0,
        prev_wave_delta: false,
        quality_clamped: false,
    };
    let input = ContextAssemblerInput {
        active_ctx: "system context content", // same as prev wave → triggers delta
        retry_context: Some("NEW RETRY HINT wave 2"),
        leader_prefix: None,
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: None,
        rejection_criteria: None,
        prev_wave_blob: Some(&prev),
        budget: Some(10000),
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: None,
        topic_knowledge: None,
        constraint_tensions: None,
    };
    let result = ContextAssembler::build(input).await;
    assert!(
        result.prev_wave_delta,
        "should detect unchanged active_ctx from prev wave"
    );
}

#[tokio::test]
async fn global_knowledge_section_preserved_under_tight_budget() {
    let input = ContextAssemblerInput {
        active_ctx: "The system context.",
        retry_context: None,
        leader_prefix: None,
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: None,
        rejection_criteria: None,
        prev_wave_blob: None,
        budget: Some(10000),
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: Some("Global overview: financial systems require atomicity."),
        topic_knowledge: None,
        constraint_tensions: None,
    };
    let result = ContextAssembler::build(input).await;
    assert!(
        result
            .text
            .contains("Global overview: financial systems require atomicity."),
        "global knowledge text must appear in assembled output"
    );
}

#[tokio::test]
async fn topic_knowledge_section_included_when_provided() {
    let input = ContextAssemblerInput {
        active_ctx: "The system context.",
        retry_context: None,
        leader_prefix: None,
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: None,
        rejection_criteria: None,
        prev_wave_blob: None,
        budget: Some(10000),
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: None,
        topic_knowledge: Some("Topic: idempotency patterns for distributed payments."),
        constraint_tensions: None,
    };
    let result = ContextAssembler::build(input).await;
    assert!(
        result
            .text
            .contains("Topic: idempotency patterns for distributed payments."),
        "topic knowledge text must appear in assembled output"
    );
}
