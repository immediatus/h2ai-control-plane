//! Tests for pure helper functions in `thinking_loop` that are not covered by
//! existing test files (thinking_loop_test.rs, thinking_loop_max_tokens_test.rs,
//! archetype_parse_test.rs, synthesis_parse_test.rs).
//!
//! Focuses on:
//!   - `format_constraint_context` (non-empty and empty corpus)
//!   - `parse_synthesis_from_markdown` edge cases (no **Score:** line)
//!   - `parse_archetypes_from_markdown` — CoT styles not yet exercised
//!   - `scheduled_tau` — edge cases (zero max_iterations not applicable due to u32)
//!   - `adaptive_n` / `adaptive_n_guarded` — iter-0 edge cases

#![allow(
    clippy::float_cmp,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::doc_markdown
)]

use h2ai_orchestrator::thinking_loop::{
    format_constraint_context, format_retry_hint_priors, parse_archetypes_from_markdown,
    parse_synthesis_from_markdown, scheduled_tau,
};

// ── format_constraint_context ─────────────────────────────────────────────────

#[test]
fn format_constraint_context_empty_corpus_returns_empty_string() {
    let result = format_constraint_context(&[]);
    assert_eq!(result, "", "empty corpus must produce empty string");
}

#[test]
fn format_constraint_context_single_constraint_contains_id() {
    use h2ai_constraints::types::ConstraintDoc;

    let doc = ConstraintDoc::new_llm_judge("CONSTRAINT-001", "must not exceed budget");
    let result = format_constraint_context(&[doc]);
    assert!(
        result.contains("CONSTRAINT-001"),
        "output must contain constraint id; got: {result}"
    );
}

#[test]
fn format_constraint_context_includes_binary_checks() {
    use h2ai_constraints::types::{
        CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity,
    };

    let mut doc = ConstraintDoc {
        id: "CONSTRAINT-042".to_string(),
        source_file: "test.yaml".to_string(),
        description: "budget enforcement".to_string(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![
            "response_time < 200ms".to_string(),
            "error_rate < 1%".to_string(),
        ],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    doc.description = "latency constraint".to_string();

    let result = format_constraint_context(&[doc]);
    assert!(
        result.contains("response_time < 200ms"),
        "must include first binary check"
    );
    assert!(
        result.contains("error_rate < 1%"),
        "must include second binary check"
    );
    // Checks are formatted as [1] and [2]
    assert!(result.contains("[1]"), "must use [1] indexing");
    assert!(result.contains("[2]"), "must use [2] indexing");
}

#[test]
fn format_constraint_context_description_included() {
    use h2ai_constraints::types::ConstraintDoc;

    // new_llm_judge uses empty description; use a custom doc with a description
    use h2ai_constraints::types::{CompositeOp, ConstraintPredicate, ConstraintSeverity};
    let doc = ConstraintDoc {
        id: "CONSTRAINT-007".to_string(),
        source_file: "test.yaml".to_string(),
        description: "enforce TLS everywhere".to_string(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let result = format_constraint_context(&[doc]);
    assert!(
        result.contains("enforce TLS everywhere"),
        "description must appear in output after ' — '; got: {result}"
    );
    assert!(result.contains(" — "), "must use em-dash separator");
}

#[test]
fn format_constraint_context_multiple_constraints() {
    use h2ai_constraints::types::ConstraintDoc;

    let doc_a = ConstraintDoc::new_llm_judge("CONSTRAINT-A", "rule a");
    let doc_b = ConstraintDoc::new_llm_judge("CONSTRAINT-B", "rule b");
    let result = format_constraint_context(&[doc_a, doc_b]);
    assert!(result.contains("CONSTRAINT-A"));
    assert!(result.contains("CONSTRAINT-B"));
}

// ── parse_synthesis_from_markdown — missing Coverage Score ────────────────────

#[test]
fn parse_synthesis_from_markdown_no_score_line_defaults_to_half() {
    let doc = r#"## Shared Understanding
The system should use eventual consistency.

## Unresolved Tensions
- Whether to use CRDTs or OT

## Coverage Assessment
No explicit score provided in this output.
"#;
    let report = parse_synthesis_from_markdown(doc);
    assert!(
        (report.coverage_score - 0.5).abs() < 1e-9,
        "missing **Score:** line must default to 0.5, got {}",
        report.coverage_score
    );
    assert!(!report.shared_understanding.is_empty());
}

#[test]
fn parse_synthesis_from_markdown_empty_tensions_section() {
    // Tensions section present but contains no bullet items
    let doc = r#"## Shared Understanding
All experts agree on the solution.

## Unresolved Tensions

## Coverage Assessment
**Score:** 0.95
"#;
    let report = parse_synthesis_from_markdown(doc);
    assert!(
        report.tensions.is_empty(),
        "tensions section with no bullets must produce empty vec"
    );
    assert!(
        (report.coverage_score - 0.95).abs() < 1e-9,
        "score must be 0.95, got {}",
        report.coverage_score
    );
}

#[test]
fn parse_synthesis_from_markdown_tension_whitespace_lines_excluded() {
    // Ensure lines that start with "- " but are empty after trimming are dropped
    let doc = r#"## Shared Understanding
Good understanding exists.

## Unresolved Tensions
- Real tension here
-
- Another real tension

## Coverage Assessment
**Score:** 0.80
"#;
    let report = parse_synthesis_from_markdown(doc);
    // Only 2 real tensions; the "- " followed by whitespace is filtered
    assert_eq!(
        report.tensions.len(),
        2,
        "whitespace-only tension bullets must be dropped; got: {:?}",
        report.tensions
    );
}

// ── parse_archetypes_from_markdown — additional CoT styles ────────────────────

#[test]
fn parse_archetypes_from_markdown_first_principles_cot_style() {
    use h2ai_types::manifest::CotStyle;

    let doc = r#"
## Archetype 1: first-principles-thinker

**Persona:** You are a first-principles thinker.

**Scope:** Ground-up reasoning

**Confidence:** 0.8

**Tau:** 0.4

**Model tier:** standard

**CoT style:** first_principles
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert_eq!(result[0].cot_style, CotStyle::FirstPrinciples);
}

#[test]
fn parse_archetypes_from_markdown_devils_advocate_cot_style() {
    use h2ai_types::manifest::CotStyle;

    let doc = r#"
## Archetype 1: contrarian

**Persona:** You are an adversarial reviewer.

**Scope:** Challenge assumptions

**Confidence:** 0.7

**Tau:** 0.5

**Model tier:** standard

**CoT style:** devil_s_advocate
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert_eq!(result[0].cot_style, CotStyle::DevilsAdvocate);
}

#[test]
fn parse_archetypes_from_markdown_devils_advocate_alternate_spelling() {
    use h2ai_types::manifest::CotStyle;

    let doc = r#"
## Archetype 1: contrarian

**Persona:** You are an adversarial reviewer.

**Scope:** Challenge assumptions

**Confidence:** 0.7

**Tau:** 0.5

**Model tier:** standard

**CoT style:** devils_advocate
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert_eq!(result[0].cot_style, CotStyle::DevilsAdvocate);
}

#[test]
fn parse_archetypes_from_markdown_fast_model_tier() {
    use h2ai_types::thinking::ModelTier;

    let doc = r#"
## Archetype 1: fast-thinker

**Persona:** You are a fast reasoner.

**Scope:** Quick iteration

**Confidence:** 0.6

**Tau:** 0.3

**Model tier:** fast

**CoT style:** none
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert_eq!(result[0].model_tier, ModelTier::Fast);
}

#[test]
fn parse_archetypes_from_markdown_capable_model_tier() {
    use h2ai_types::thinking::ModelTier;

    let doc = r#"
## Archetype 1: deep-thinker

**Persona:** You are a deep reasoner.

**Scope:** High-complexity problems

**Confidence:** 0.9

**Tau:** 0.2

**Model tier:** capable

**CoT style:** step_by_step
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert_eq!(result[0].model_tier, ModelTier::Capable);
}

#[test]
fn parse_archetypes_from_markdown_unknown_model_tier_defaults_to_standard() {
    use h2ai_types::thinking::ModelTier;

    let doc = r#"
## Archetype 1: unknown-tier

**Persona:** You are a thinker.

**Scope:** General

**Confidence:** 0.7

**Tau:** 0.4

**Model tier:** turbo_ultra_max

**CoT style:** none
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert_eq!(result[0].model_tier, ModelTier::Standard);
}

#[test]
fn parse_archetypes_from_markdown_name_without_colon_prefix() {
    // First non-empty line has no colon → entire line becomes name
    let doc = r#"
## Archetype no-prefix-name

**Persona:** You are a thinker with no numeric prefix.

**Scope:** General

**Confidence:** 0.7

**Tau:** 0.4

**Model tier:** standard

**CoT style:** none
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert_eq!(
        result[0].name, "no-prefix-name",
        "name without colon prefix must use full line as name"
    );
}

#[test]
fn parse_archetypes_from_markdown_missing_scope_defaults_to_empty() {
    // Scope is optional in parse_archetype_block (uses unwrap_or_default)
    let doc = r#"
## Archetype 1: no-scope

**Persona:** You are a thinker without a scope field.

**Confidence:** 0.7

**Tau:** 0.4

**Model tier:** standard

**CoT style:** none
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert_eq!(
        result[0].scope, "",
        "missing scope must default to empty string"
    );
}

#[test]
fn parse_archetypes_from_markdown_missing_confidence_defaults_to_0_7() {
    let doc = r#"
## Archetype 1: no-confidence

**Persona:** You are a thinker without explicit confidence.

**Scope:** General

**Tau:** 0.4

**Model tier:** standard

**CoT style:** none
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert!(
        (result[0].confidence - 0.7).abs() < 1e-9,
        "missing confidence must default to 0.7, got {}",
        result[0].confidence
    );
}

#[test]
fn parse_archetypes_from_markdown_missing_tau_defaults_to_0_5() {
    let doc = r#"
## Archetype 1: no-tau

**Persona:** You are a thinker without explicit tau.

**Scope:** General

**Confidence:** 0.8

**Model tier:** standard

**CoT style:** none
"#;
    let result = parse_archetypes_from_markdown(doc).unwrap();
    assert!(
        (result[0].tau - 0.5).abs() < 1e-9,
        "missing tau must default to 0.5, got {}",
        result[0].tau
    );
}

// ── scheduled_tau — additional edge cases ────────────────────────────────────

#[test]
fn scheduled_tau_single_iteration_returns_tau_max() {
    // max_iterations = 1: progress = 0 at iter 0 → tau_max regardless of tau_min
    let t = scheduled_tau(0, 1, 0.9, 0.1);
    assert!(
        (t - 0.9).abs() < 1e-9,
        "single iteration must return tau_max; got {t}"
    );
}

#[test]
fn scheduled_tau_clamped_at_boundaries() {
    // tau_max = tau_min → constant schedule
    let t0 = scheduled_tau(0, 5, 0.5, 0.5);
    let t4 = scheduled_tau(4, 5, 0.5, 0.5);
    assert!(
        (t0 - 0.5).abs() < 1e-9,
        "equal tau_max/tau_min must produce constant 0.5 at iter 0"
    );
    assert!(
        (t4 - 0.5).abs() < 1e-9,
        "equal tau_max/tau_min must produce constant 0.5 at last iter"
    );
}

#[test]
fn scheduled_tau_midpoint_is_average() {
    // iter 1 out of [0, 2] → progress = 0.5 → midpoint
    let t = scheduled_tau(1, 3, 1.0, 0.0);
    assert!(
        (t - 0.5).abs() < 1e-9,
        "midpoint iteration must yield average of tau_max/tau_min; got {t}"
    );
}

// ── format_retry_hint_priors ─────────────────────────────────────────────────

#[test]
fn format_retry_hint_priors_empty_returns_empty_string() {
    let result = format_retry_hint_priors(&[]);
    assert_eq!(result, "", "empty patterns must produce empty string");
}

#[test]
fn format_retry_hint_priors_contains_hint_text() {
    use h2ai_types::memory::RetryHintPattern;

    let patterns = vec![RetryHintPattern {
        trigger_tags: vec!["http".to_string(), "timeout".to_string()],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "use idempotent retry with backoff".to_string(),
        success_count: 5,
        attempt_count: 7,
    }];
    let result = format_retry_hint_priors(&patterns);
    assert!(
        result.contains("use idempotent retry with backoff"),
        "output must contain hint_text; got: {result}"
    );
    assert!(
        result.contains("RETRY HISTORY"),
        "output must contain section header; got: {result}"
    );
}

#[test]
fn format_retry_hint_priors_caps_at_five_patterns() {
    use h2ai_types::memory::RetryHintPattern;

    let patterns: Vec<RetryHintPattern> = (0..8)
        .map(|i| RetryHintPattern {
            trigger_tags: vec!["tag".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: format!("hint-{i}"),
            success_count: i,
            attempt_count: i + 2,
        })
        .collect();
    let result = format_retry_hint_priors(&patterns);
    // Only first 5 hints should appear — last 3 must not be in the output
    let hint_5_present = result.contains("hint-5");
    let hint_4_present = result.contains("hint-4");
    assert!(hint_4_present, "hint-4 (5th) must be present");
    assert!(
        !hint_5_present,
        "hint-5 (6th) must be excluded; only top-5 allowed"
    );
}
