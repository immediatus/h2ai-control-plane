use h2ai_config::prompts::{
    COMPILER_CONSTRAINT_ORDERING, THINKING_ARCHETYPE_SELECT_ITER1, THINKING_SYNTHESIS_TASK,
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
