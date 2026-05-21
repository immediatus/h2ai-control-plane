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
use h2ai_orchestrator::context_assembler::{
    assemble_raw, build_sections, ContextAssemblerInput, SectionTag,
};

fn base_input<'a>(active: &'a str) -> ContextAssemblerInput<'a> {
    ContextAssemblerInput {
        active_ctx: active,
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
fn tension_section_appears_in_build_sections() {
    let input = ContextAssemblerInput {
        constraint_tensions: Some("tension: GDPR vs performance"),
        ..base_input("active")
    };
    let sections = build_sections(&input);
    let tags: Vec<_> = sections.iter().map(|s| &s.tag).collect();
    assert!(tags.contains(&&SectionTag::ConstraintTension));
    let tension_sec = sections
        .iter()
        .find(|s| s.tag == SectionTag::ConstraintTension)
        .unwrap();
    assert!(tension_sec.text.contains("GDPR"));
    assert_eq!(tension_sec.importance, 0.85);
    assert!(!tension_sec.preserve);
}

#[test]
fn tension_appears_in_assemble_raw() {
    let input = ContextAssemblerInput {
        constraint_tensions: Some("- GDPR conflicts with caching"),
        ..base_input("task description")
    };
    let raw = assemble_raw(&input);
    assert!(raw.contains("[CONSTRAINT TENSIONS]"));
    assert!(raw.contains("GDPR"));
}

#[test]
fn no_tension_when_none() {
    let input = base_input("task");
    let sections = build_sections(&input);
    assert!(!sections
        .iter()
        .any(|s| s.tag == SectionTag::ConstraintTension));
    let raw = assemble_raw(&input);
    assert!(!raw.contains("[CONSTRAINT TENSIONS]"));
}

#[test]
fn global_knowledge_in_assemble_raw() {
    let input = ContextAssemblerInput {
        global_knowledge: Some("GDPR requires explicit consent"),
        ..base_input("task")
    };
    let raw = assemble_raw(&input);
    assert!(raw.contains("[KNOWLEDGE]"));
    assert!(raw.contains("GDPR requires explicit consent"));
}
