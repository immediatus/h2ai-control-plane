use h2ai_types::prompts::{
    auditor_system_prompt_default, ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT, AUDITOR_PROMPT_TEMPLATE,
    AUDITOR_SYSTEM_PROMPT, BINARY_CLASSIFIER_SYSTEM_PROMPT, COT_RUBRIC, EVALUATOR_SYSTEM_PROMPT,
    SYNTHESIS_CRITIQUE_PROMPT, SYNTHESIS_WRITE_PROMPT, TAO_OBSERVATION_FAIL_PATTERN,
    TAO_OBSERVATION_FAIL_SCHEMA, TAO_OBSERVATION_PASS, TAO_RETRY_INSTRUCTION,
};

#[test]
fn cot_rubric_is_nonempty_and_contains_score_key() {
    assert!(!COT_RUBRIC.is_empty());
    assert!(COT_RUBRIC.contains("score"));
}

#[test]
fn evaluator_system_prompt_contains_json_format_instruction() {
    assert!(EVALUATOR_SYSTEM_PROMPT.contains("score"));
    assert!(EVALUATOR_SYSTEM_PROMPT.contains("reason"));
}

#[test]
fn binary_classifier_prompt_is_yes_or_no() {
    assert!(
        BINARY_CLASSIFIER_SYSTEM_PROMPT.contains("YES")
            || BINARY_CLASSIFIER_SYSTEM_PROMPT.contains("YES or NO")
    );
}

#[test]
fn adversarial_evaluator_prompt_is_nonempty() {
    assert!(!ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT.is_empty());
    assert!(ADVERSARIAL_EVALUATOR_SYSTEM_PROMPT.contains("score"));
}

#[test]
fn auditor_system_prompt_ends_with_json_template() {
    assert!(AUDITOR_SYSTEM_PROMPT.contains("approved"));
    assert!(AUDITOR_SYSTEM_PROMPT.contains("violated"));
}

#[test]
fn auditor_system_prompt_default_matches_constant() {
    assert_eq!(auditor_system_prompt_default(), AUDITOR_SYSTEM_PROMPT);
}

#[test]
fn auditor_prompt_template_has_placeholders() {
    assert!(AUDITOR_PROMPT_TEMPLATE.contains("{constraints}"));
    assert!(AUDITOR_PROMPT_TEMPLATE.contains("{proposal}"));
}

#[test]
fn tao_observation_pass_is_nonempty() {
    assert!(!TAO_OBSERVATION_PASS.is_empty());
}

#[test]
fn tao_observation_fail_pattern_has_turn_placeholder() {
    assert!(TAO_OBSERVATION_FAIL_PATTERN.contains("{turn}"));
}

#[test]
fn tao_observation_fail_schema_has_turn_and_error_placeholders() {
    assert!(TAO_OBSERVATION_FAIL_SCHEMA.contains("{turn}"));
    assert!(TAO_OBSERVATION_FAIL_SCHEMA.contains("{error}"));
}

#[test]
fn tao_retry_instruction_has_turn_placeholder() {
    assert!(TAO_RETRY_INSTRUCTION.contains("{turn}"));
}

#[test]
fn synthesis_critique_prompt_has_required_placeholders() {
    assert!(SYNTHESIS_CRITIQUE_PROMPT.contains("{task_description}"));
    assert!(SYNTHESIS_CRITIQUE_PROMPT.contains("{constraint_list}"));
    assert!(SYNTHESIS_CRITIQUE_PROMPT.contains("{proposals_block}"));
    assert!(SYNTHESIS_CRITIQUE_PROMPT.contains("{critique_schema}"));
}

#[test]
fn synthesis_write_prompt_has_required_placeholders() {
    assert!(SYNTHESIS_WRITE_PROMPT.contains("{task_description}"));
    assert!(SYNTHESIS_WRITE_PROMPT.contains("{constraint_list}"));
    assert!(SYNTHESIS_WRITE_PROMPT.contains("{proposals_block}"));
    assert!(SYNTHESIS_WRITE_PROMPT.contains("{critique_document}"));
}
