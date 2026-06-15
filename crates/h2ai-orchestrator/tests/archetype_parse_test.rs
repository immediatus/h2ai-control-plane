use h2ai_orchestrator::thinking_loop::parse_archetypes_from_markdown;
use h2ai_types::manifest::CotStyle;
use h2ai_types::thinking::ModelTier;

fn two_archetype_doc() -> String {
    r#"
## Archetype 1: security-engineer

**Lens:** Security systems from a threat-modelling perspective

**Persona:** You are a security engineer who prioritizes defense-in-depth. You scan for auth boundaries and escalation paths before anything else.

**Scope:** Authentication, authorization, and secrets management

**Confidence:** 0.90

**Tau:** 0.20

**Model tier:** capable

**CoT style:** step_by_step

## Archetype 2: systems-architect

**Lens:** Distributed systems from a resilience and latency angle

**Persona:** You are a systems architect who thinks in failure modes. You always ask "what happens when this call fails?" before proposing any design.

**Scope:** Service topology, data flow, and SLA guarantees

**Confidence:** 0.75

**Tau:** 0.40

**Model tier:** standard

**CoT style:** backward_chaining
"#
    .to_string()
}

#[test]
fn parse_two_archetypes_returns_vec_of_len_2() {
    let result = parse_archetypes_from_markdown(&two_archetype_doc()).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn first_archetype_name_is_kebab_case() {
    let result = parse_archetypes_from_markdown(&two_archetype_doc()).unwrap();
    assert_eq!(result[0].name, "security-engineer");
}

#[test]
fn persona_field_extracted_correctly() {
    let result = parse_archetypes_from_markdown(&two_archetype_doc()).unwrap();
    assert!(
        result[0].persona.contains("defense-in-depth"),
        "persona must contain the filled-in text"
    );
}

#[test]
fn confidence_parsed_as_float() {
    let result = parse_archetypes_from_markdown(&two_archetype_doc()).unwrap();
    assert!((result[0].confidence - 0.90).abs() < 0.01);
    assert!((result[1].confidence - 0.75).abs() < 0.01);
}

#[test]
fn tau_parsed_as_float() {
    let result = parse_archetypes_from_markdown(&two_archetype_doc()).unwrap();
    assert!((result[0].tau - 0.20).abs() < 0.01);
}

#[test]
fn model_tier_parsed_correctly() {
    let result = parse_archetypes_from_markdown(&two_archetype_doc()).unwrap();
    assert_eq!(result[0].model_tier, ModelTier::Capable);
    assert_eq!(result[1].model_tier, ModelTier::Standard);
}

#[test]
fn cot_style_parsed_correctly() {
    let result = parse_archetypes_from_markdown(&two_archetype_doc()).unwrap();
    assert_eq!(result[0].cot_style, CotStyle::StepByStep);
    assert_eq!(result[1].cot_style, CotStyle::BackwardChaining);
}

#[test]
fn unknown_cot_style_silently_becomes_none() {
    let doc = r#"
## Archetype 1: test-archetype

**Lens:** test

**Persona:** You are a tester.

**Scope:** testing

**Confidence:** 0.5

**Tau:** 0.5

**Model tier:** fast

**CoT style:** totally_unknown_style
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert_eq!(result[0].cot_style, CotStyle::None);
}

#[test]
fn missing_persona_field_returns_error() {
    let doc = r#"
## Archetype 1: no-persona

**Lens:** test

**Scope:** testing

**Confidence:** 0.5

**Tau:** 0.5

**Model tier:** fast

**CoT style:** none
"#;
    let result = parse_archetypes_from_markdown(doc);
    assert!(result.is_err(), "missing Persona must return Err");
}

#[test]
fn no_archetype_headers_returns_error() {
    let result = parse_archetypes_from_markdown("no headers here at all");
    assert!(result.is_err());
}
