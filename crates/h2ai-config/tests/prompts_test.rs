use h2ai_config::prompts::{
    PromptTemplate, COMPILER_CONSTRAINT_ORDERING, I1_GAP_EXTRACTOR_SYSTEM, I1_GAP_EXTRACTOR_TASK,
    I1_SEMANTIC_REPAIR_SLOT, I1_SYNTHESIS_VALIDATOR_TASK, THINKING_ARCHETYPE_SELECT_ITER1,
    THINKING_SYNTHESIS_MD_SYSTEM, THINKING_SYNTHESIS_TASK,
};

#[test]
fn semantic_ordering_template_renders_all_fields() {
    let rendered = COMPILER_CONSTRAINT_ORDERING.render(&[
        ("id", "C-005"),
        ("first", "account debit"),
        ("then", "Kafka publish"),
    ]);
    assert!(rendered.contains("C-005"));
    assert!(rendered.contains("account debit"));
    assert!(rendered.contains("Kafka publish"));
}

#[test]
fn archetype_select_iter1_renders_required_fields() {
    let rendered = THINKING_ARCHETYPE_SELECT_ITER1.render(&[
        ("description", "design a caching layer"),
        ("constraints", "CONSTRAINT-001"),
        ("research_context", "Redis is the standard choice"),
        ("n", "3"),
    ]);
    assert!(rendered.contains("design a caching layer"));
    assert!(rendered.contains("CONSTRAINT-001"));
    assert!(rendered.contains("Redis is the standard choice"));
    assert!(rendered.contains('3'));
}

#[test]
fn synthesis_task_renders_perspectives_and_prior() {
    let rendered = THINKING_SYNTHESIS_TASK.render(&[
        ("perspectives", "arch A: use Redis"),
        ("prior_understanding", ""),
    ]);
    assert!(rendered.contains("arch A: use Redis"));
}

#[test]
fn prompt_template_display_returns_raw_text() {
    const T: PromptTemplate = PromptTemplate("hello {world}");
    assert_eq!(format!("{T}"), "hello {world}");
}

#[test]
fn prompt_template_as_str_returns_raw_template() {
    const T: PromptTemplate = PromptTemplate("raw {template}");
    assert_eq!(T.as_str(), "raw {template}");
}

#[test]
fn synthesis_md_system_does_not_instruct_json_output() {
    // THINKING_SYNTHESIS_MD_SYSTEM is used with tournament_merge + THINKING_SYNTHESIS_MD_PAIRWISE,
    // which expects markdown sections. Verifying it does NOT say "JSON" prevents the conflict
    // where the system says JSON but the template expects markdown (coverage_score=0.5 fallback bug).
    assert!(
        !THINKING_SYNTHESIS_MD_SYSTEM.to_lowercase().contains("json"),
        "THINKING_SYNTHESIS_MD_SYSTEM must not instruct JSON output — use with markdown pairwise template"
    );
    assert!(
        THINKING_SYNTHESIS_MD_SYSTEM.contains("markdown"),
        "THINKING_SYNTHESIS_MD_SYSTEM should instruct markdown format"
    );
}

#[test]
fn gap_extractor_prompts_defined() {
    assert!(!I1_GAP_EXTRACTOR_SYSTEM.is_empty());
    assert!(I1_GAP_EXTRACTOR_TASK.contains("{check_text}"));
    assert!(I1_GAP_EXTRACTOR_TASK.contains("{verifier_reasons}"));
}

#[test]
fn synthesis_validator_prompt_defined() {
    assert!(!I1_SYNTHESIS_VALIDATOR_TASK.is_empty());
    assert!(I1_SYNTHESIS_VALIDATOR_TASK.contains("{check_text}"));
    assert!(I1_SYNTHESIS_VALIDATOR_TASK.contains("{incorrect_pattern}"));
    assert!(I1_SYNTHESIS_VALIDATOR_TASK.contains("{correct_pattern}"));
    assert!(I1_SYNTHESIS_VALIDATOR_TASK.contains("{mechanistic_reason}"));
}

#[test]
fn semantic_repair_slot_template_defined() {
    assert!(I1_SEMANTIC_REPAIR_SLOT.contains("{incorrect_pattern}"));
    assert!(I1_SEMANTIC_REPAIR_SLOT.contains("{correct_pattern}"));
    assert!(I1_SEMANTIC_REPAIR_SLOT.contains("{mechanistic_reason}"));
}

#[test]
fn i1_semantic_repair_slot_uses_prior_approach_not_wrong_belief() {
    assert!(
        !I1_SEMANTIC_REPAIR_SLOT.contains("WRONG BELIEF"),
        "I1_SEMANTIC_REPAIR_SLOT must use PRIOR APPROACH, not WRONG BELIEF"
    );
    assert!(I1_SEMANTIC_REPAIR_SLOT.contains("PRIOR APPROACH"));
}

#[test]
fn i1_synthesis_validator_task_uses_prior_approach_not_wrong_belief() {
    assert!(
        !I1_SYNTHESIS_VALIDATOR_TASK.contains("WRONG BELIEF"),
        "I1_SYNTHESIS_VALIDATOR_TASK must use PRIOR APPROACH, not WRONG BELIEF"
    );
    assert!(I1_SYNTHESIS_VALIDATOR_TASK.contains("PRIOR APPROACH"));
}
