use h2ai_config::prompts::{
    PromptTemplate, COMPILER_CONSTRAINT_ORDERING, THINKING_ARCHETYPE_SELECT_ITER1,
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
