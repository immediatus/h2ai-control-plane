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
