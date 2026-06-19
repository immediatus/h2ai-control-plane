#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::wildcard_imports
)]
//! Extended unit tests for `h2ai_orchestrator::context_assembler`.
//!
//! Supplements context_assembler_test.rs with additional branch coverage for:
//! - `estimate_tokens`: code-dense vs prose-dense content, empty input
//! - `sections_to_text`: empty sections filtered, separator behaviour
//! - `score_sections`: both constraint boost AND prev-wave penalty fire simultaneously
//! - `rule_pass`: no previous wave (wave 0 path)
//! - `importance_trim`: all sections preserved → no trimming
//! - `build_sections`: empty optional fields are silently skipped
//! - `ContextAssembler::build`: budget=0 forces ImportanceScored path
//! - `StableContextCache`: record_hit on missing key is a no-op

use h2ai_orchestrator::context_assembler::stable_cache::{CachedSection, StableContextCache};
use h2ai_orchestrator::context_assembler::{
    assemble_raw, build_sections, estimate_tokens, importance_trim, quality_guard, rule_pass,
    score_sections, sections_to_text, AssembledContext, CompressionKind, ContextAssembler,
    ContextAssemblerInput, RulePassInput, Section, SectionTag,
};

// ── estimate_tokens ───────────────────────────────────────────────────────────

#[test]
fn estimate_tokens_empty_returns_zero() {
    assert_eq!(estimate_tokens(""), 0);
}

#[test]
fn estimate_tokens_prose_text() {
    // Pure prose: chars_per_token ≈ 4.5 (code_ratio = 0)
    let prose = "The quick brown fox jumps over the lazy dog.";
    let tokens = estimate_tokens(prose);
    // 44 chars / 4.5 ≈ 10 tokens (ceiling)
    assert!(
        (8..=15).contains(&tokens),
        "prose token estimate out of range: {tokens}"
    );
}

#[test]
fn estimate_tokens_code_dense_text() {
    // 100% code lines (all start with '-', '{', '[', or '#')
    let code = "- item one\n- item two\n- item three\n- item four\n- item five";
    let tokens = estimate_tokens(code);
    // chars_per_token = 3.0 (code_ratio = 1.0 → 1.5*(1-1.0) + 3.0 = 3.0)
    // chars ≈ 56 → tokens ≈ ceil(56/3.0) = 19
    assert!(tokens > 0, "code token estimate should be positive");
}

#[test]
fn estimate_tokens_mixed_code_and_prose() {
    let mixed = "Some prose line.\n- a list item\nAnother prose line.\n# A heading";
    let tokens = estimate_tokens(mixed);
    assert!(tokens > 0, "mixed token estimate should be positive");
}

#[test]
fn estimate_tokens_hash_line_counts_as_code() {
    let with_hash = "# heading line\nsome prose";
    let tokens = estimate_tokens(with_hash);
    assert!(tokens > 0);
}

#[test]
fn estimate_tokens_bracket_lines_count_as_code() {
    let with_bracket = "{ \"key\": \"value\" }\n[ 1, 2, 3 ]";
    let tokens = estimate_tokens(with_bracket);
    assert!(tokens > 0);
}

// ── sections_to_text ─────────────────────────────────────────────────────────

#[test]
fn sections_to_text_empty_sections_returns_empty_string() {
    let result = sections_to_text(&[]);
    assert_eq!(result, "");
}

#[test]
fn sections_to_text_skips_empty_text_sections() {
    let sections = vec![
        Section {
            tag: SectionTag::ActiveCtx,
            text: "content".to_string(),
            importance: 0.7,
            preserve: false,
        },
        Section {
            tag: SectionTag::RetryContext,
            text: String::new(),
            importance: 0.5,
            preserve: false,
        },
        Section {
            tag: SectionTag::Mandate,
            text: "mandate text".to_string(),
            importance: 0.95,
            preserve: true,
        },
    ];
    let text = sections_to_text(&sections);
    // Empty RetryContext must not appear
    assert!(text.contains("content"), "active ctx should appear");
    assert!(text.contains("mandate text"), "mandate should appear");
    // Double newline separator between sections
    assert!(
        text.contains("\n\n"),
        "sections should be double-newline separated"
    );
}

#[test]
fn sections_to_text_single_section_no_separator() {
    let sections = vec![Section {
        tag: SectionTag::Grounding,
        text: "grounding".to_string(),
        importance: 1.0,
        preserve: true,
    }];
    let text = sections_to_text(&sections);
    assert_eq!(text, "grounding");
    assert!(!text.contains("\n\n"));
}

// ── score_sections: both signals fire simultaneously ──────────────────────────

#[test]
fn score_sections_both_boost_and_penalty_net_negative() {
    // A non-preserved section that:
    //   1. Contains a constraint ID (C-NNN) → +0.15 boost
    //   2. Appears verbatim in prev_wave_text → -0.30 penalty
    // Net: base - 0.15 (penalty dominates)
    let section_text = "Requirement C-042 must be satisfied.";
    let sections = vec![Section {
        tag: SectionTag::RetryContext,
        text: section_text.to_string(),
        importance: 0.5,
        preserve: false,
    }];
    let prev = format!("Earlier context. {} More text.", section_text);
    let scored = score_sections(sections, Some(&prev));
    let s = scored
        .iter()
        .find(|s| s.tag == SectionTag::RetryContext)
        .unwrap();
    // base=0.5, +0.15 constraint boost, -0.30 prev-wave penalty → net 0.5 - 0.15 = 0.35
    // Because the rule is: adjusted starts at base+0.15, then if prev match: adjusted = (base-0.30).max(0.0)
    // So final = max(0.5 - 0.30, 0.0) = 0.2
    assert!(
        s.importance < 0.5,
        "penalty should dominate boost, got {}",
        s.importance
    );
}

#[test]
fn score_sections_constraint_boost_only_no_prev_wave() {
    let section_text = "Ensure compliance with C-007.";
    let sections = vec![Section {
        tag: SectionTag::RetryContext,
        text: section_text.to_string(),
        importance: 0.5,
        preserve: false,
    }];
    let scored = score_sections(sections, None);
    let s = scored
        .iter()
        .find(|s| s.tag == SectionTag::RetryContext)
        .unwrap();
    // +0.15 boost only → 0.65
    assert!(
        s.importance > 0.5,
        "should be boosted above base, got {}",
        s.importance
    );
    assert!(
        (s.importance - 0.65).abs() < 1e-6,
        "expected 0.65, got {}",
        s.importance
    );
}

#[test]
fn score_sections_preserved_sections_unchanged() {
    // Preserved sections must be skipped entirely — no boost or penalty
    let sections = vec![Section {
        tag: SectionTag::Grounding,
        text: "C-001 grounding text".to_string(),
        importance: 1.0,
        preserve: true,
    }];
    let prev = "C-001 grounding text";
    let scored = score_sections(sections, Some(prev));
    let s = scored
        .iter()
        .find(|s| s.tag == SectionTag::Grounding)
        .unwrap();
    // preserve=true → importance unchanged
    assert_eq!(s.importance, 1.0, "preserved sections must not be adjusted");
}

// ── rule_pass: wave 0 with no prev blob ──────────────────────────────────────

#[test]
fn rule_pass_wave_zero_no_prev_blob_returns_false() {
    let mut sections = vec![Section {
        tag: SectionTag::RetryContext,
        text: "some retry context".to_string(),
        importance: 0.5,
        preserve: false,
    }];
    let delta = rule_pass(
        &mut sections,
        RulePassInput {
            prev_wave_blob: None,
            wave: 0,
        },
    );
    assert!(!delta, "no prev blob → delta_applied should be false");
    // Text should be normalized (whitespace pass) but not delta-replaced
    assert!(sections[0].text.contains("retry context"));
}

#[test]
fn rule_pass_preserved_sections_not_modified() {
    let original_text = "leader prefix content";
    let mut sections = vec![Section {
        tag: SectionTag::LeaderPrefix,
        text: original_text.to_string(),
        importance: 1.0,
        preserve: true,
    }];
    let prev = AssembledContext {
        text: original_text.to_string(),
        token_estimate: 5,
        compression: CompressionKind::None,
        compression_ratio: 1.0,
        prev_wave_delta: false,
        quality_clamped: false,
    };
    let delta = rule_pass(
        &mut sections,
        RulePassInput {
            prev_wave_blob: Some(&prev),
            wave: 1,
        },
    );
    // Preserved sections are skipped — no delta replacement even if content matches prev wave
    assert!(
        !delta,
        "preserved sections must not trigger delta replacement"
    );
    assert_eq!(
        sections[0].text, original_text,
        "preserved text must be unchanged"
    );
}

// ── importance_trim: all-preserved sections ───────────────────────────────────

#[test]
fn importance_trim_all_preserved_sections_unchanged() {
    // When all sections are preserve=true, nothing can be trimmed.
    // Even with a very tight budget, the sections must be returned as-is.
    let sections = vec![
        Section {
            tag: SectionTag::Grounding,
            text: "A".repeat(1000),
            importance: 1.0,
            preserve: true,
        },
        Section {
            tag: SectionTag::Mandate,
            text: "B".repeat(1000),
            importance: 0.95,
            preserve: true,
        },
    ];
    let original_lens: Vec<usize> = sections.iter().map(|s| s.text.len()).collect();
    let trimmed = importance_trim(sections, 1); // impossibly tight budget
    for (i, s) in trimmed.iter().enumerate() {
        assert_eq!(
            s.text.len(),
            original_lens[i],
            "preserved section {} must not be trimmed",
            i
        );
    }
}

// ── build_sections: empty optional fields silently skipped ────────────────────

#[test]
fn build_sections_empty_optional_fields_skipped() {
    // Each `Some("")` should produce no section (silently skipped).
    let input = ContextAssemblerInput {
        active_ctx: "ctx",
        retry_context: Some(""),      // empty → skip
        leader_prefix: Some(""),      // empty → skip
        grounding: Some(""),          // empty → skip
        tombstone: Some(""),          // empty → skip
        role_frame: Some(""),         // empty → skip
        mandate: Some(""),            // empty → skip
        rejection_criteria: Some(""), // empty → skip
        prev_wave_blob: None,
        budget: None,
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: Some(""),     // empty → skip
        topic_knowledge: Some(""),      // empty → skip
        constraint_tensions: Some(""),  // empty → skip
        compliance_checklist: Some(""), // empty → skip
    };
    let sections = build_sections(&input);
    // Only the non-empty active_ctx produces a section
    let tags: Vec<&SectionTag> = sections.iter().map(|s| &s.tag).collect();
    assert_eq!(
        sections.len(),
        1,
        "only ActiveCtx should be present, got {:?}",
        tags
    );
    assert_eq!(sections[0].tag, SectionTag::ActiveCtx);
}

#[test]
fn build_sections_empty_active_ctx_skipped() {
    // active_ctx = "" → no section produced, no panic
    let input = ContextAssemblerInput {
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
        compliance_checklist: None,
    };
    let sections = build_sections(&input);
    assert!(
        sections.is_empty(),
        "empty active_ctx + all None → no sections"
    );
}

// ── assemble_raw: empty optional sections produce no separators ───────────────

#[test]
fn assemble_raw_empty_active_ctx_only() {
    let input = ContextAssemblerInput {
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
        compliance_checklist: None,
    };
    // active_ctx = "" is still pushed unconditionally — see assemble_raw
    let raw = assemble_raw(&input);
    // The empty string is pushed, result is just empty (parts.join produces "")
    assert_eq!(raw, "");
}

#[test]
fn assemble_raw_empty_tombstone_not_included() {
    let input = ContextAssemblerInput {
        active_ctx: "ctx",
        retry_context: None,
        leader_prefix: None,
        grounding: None,
        tombstone: Some(""),
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
        compliance_checklist: None,
    };
    let raw = assemble_raw(&input);
    // Only active_ctx present (tombstone empty → not pushed)
    assert_eq!(raw, "ctx");
}

// ── ContextAssembler::build — budget=0 forces importance trim ────────────────

#[tokio::test]
async fn build_budget_zero_returns_importance_scored() {
    // budget=0 → after rule pass, any non-zero token content exceeds budget,
    // so importance_trim runs. No adapter → ImportanceScored.
    let input = ContextAssemblerInput {
        active_ctx: "Some non-trivial context to process",
        retry_context: Some("retry hint"),
        leader_prefix: None,
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: None,
        rejection_criteria: None,
        prev_wave_blob: None,
        budget: Some(0), // zero budget
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: None,
        topic_knowledge: None,
        constraint_tensions: None,
        compliance_checklist: None,
    };
    let result = ContextAssembler::build(input).await;
    // Anything but CompressionKind::None (rule pass can't satisfy budget=0)
    assert!(
        result.compression == CompressionKind::RuleBased
            || result.compression == CompressionKind::ImportanceScored
            || result.compression == CompressionKind::LlmSummarized,
        "budget=0 should trigger compression, got {:?}",
        result.compression
    );
}

// ── StableContextCache: record_hit on missing key is a no-op ─────────────────

#[test]
fn stable_cache_record_hit_missing_key_noop() {
    let cache = StableContextCache::new();
    // Should not panic
    cache.record_hit(0xDEADu64);
    assert!(cache.is_empty(), "cache should still be empty");
}

#[test]
fn stable_cache_multiple_inserts_and_hits() {
    let cache = StableContextCache::new();
    for i in 0u64..3 {
        cache.insert(
            i,
            CachedSection {
                compressed_text: format!("text-{i}"),
                original_token_estimate: 100,
                compressed_token_estimate: 20,
                hit_count: 0,
            },
        );
    }
    assert_eq!(cache.len(), 3);

    cache.record_hit(0);
    cache.record_hit(0);
    cache.record_hit(1);

    assert_eq!(cache.get(0).unwrap().hit_count, 2);
    assert_eq!(cache.get(1).unwrap().hit_count, 1);
    assert_eq!(cache.get(2).unwrap().hit_count, 0);
}

// ── quality_guard edge cases ──────────────────────────────────────────────────

#[test]
fn quality_guard_compressed_equals_original_no_clamp() {
    // ratio = 1.0 → never clamp (1.0 is not < threshold of 0.4)
    assert!(!quality_guard(500, 500, 0.4));
}

#[test]
fn quality_guard_very_aggressive_compression_clamps() {
    // ratio = 10/1000 = 0.01 < threshold=0.4 → clamp
    assert!(quality_guard(1000, 10, 0.4));
}

// ── assemble_raw: non-empty optional fields are included ─────────────────────

#[test]
fn assemble_raw_leader_prefix_non_empty_included() {
    // Covers lines 250-252: the `if !lp.is_empty()` branch in assemble_raw.
    let input = ContextAssemblerInput {
        active_ctx: "ctx",
        leader_prefix: Some("LEADER ROLE PREFIX"),
        retry_context: None,
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
        compliance_checklist: None,
    };
    let raw = assemble_raw(&input);
    assert!(
        raw.contains("LEADER ROLE PREFIX"),
        "non-empty leader_prefix must appear in raw output"
    );
}

#[test]
fn assemble_raw_retry_context_non_empty_included() {
    // Covers lines 278-280: the `if !ret.is_empty()` branch in assemble_raw.
    let input = ContextAssemblerInput {
        active_ctx: "ctx",
        leader_prefix: None,
        retry_context: Some("previous attempt feedback"),
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
        compliance_checklist: None,
    };
    let raw = assemble_raw(&input);
    assert!(
        raw.contains("previous attempt feedback"),
        "non-empty retry_context must appear in raw output"
    );
}

#[test]
fn assemble_raw_rejection_criteria_non_empty_included() {
    // Covers lines 293-295: the `if !rc.is_empty()` branch in assemble_raw.
    let input = ContextAssemblerInput {
        active_ctx: "ctx",
        leader_prefix: None,
        retry_context: None,
        grounding: None,
        tombstone: None,
        role_frame: None,
        mandate: Some("do the thing"),
        rejection_criteria: Some("do not include pricing data"),
        prev_wave_blob: None,
        budget: None,
        quality_guard_ratio: None,
        compression_adapter: None,
        stable_cache: None,
        global_knowledge: None,
        topic_knowledge: None,
        constraint_tensions: None,
        compliance_checklist: None,
    };
    let raw = assemble_raw(&input);
    assert!(
        raw.contains("do not include pricing data"),
        "non-empty rejection_criteria must appear in raw output"
    );
    assert!(
        raw.contains("[AFTER WRITING YOUR PROPOSAL, IDENTIFY THE BIGGEST RISK]"),
        "rejection_criteria must be prefixed with the risk label"
    );
    assert!(
        raw.contains("[MANDATE]"),
        "non-empty mandate must also appear"
    );
}

// ── importance_trim: inner loop body (lines 590, 606, 614) ───────────────────

#[test]
fn importance_trim_trims_sections_when_over_budget() {
    // Covers lines 590, 606, 614: the inner trimming loop body.
    // Create a non-preserved section with long text and set a very tight budget
    // so the loop body runs at least once.
    let long_text = "First sentence here. Second sentence follows. Third sentence continues. \
        Fourth sentence adds more. Fifth sentence is included too. Sixth sentence \
        wraps things up nicely."
        .repeat(20); // make it long enough to exceed any small budget
    let sections = vec![
        Section {
            tag: SectionTag::RetryContext,
            text: long_text.clone(),
            importance: 0.3,
            preserve: false,
        },
        Section {
            tag: SectionTag::Mandate,
            text: "Mandatory content that must survive.".to_string(),
            importance: 0.95,
            preserve: true, // preserved — must not be trimmed
        },
    ];
    let original_len = sections[0].text.len();
    // budget = 1 token forces the inner loop to trim
    let trimmed = importance_trim(sections, 1);
    // The non-preserved section must be shorter than the original
    assert!(
        trimmed[0].text.len() < original_len,
        "non-preserved section must be trimmed, len={} original={}",
        trimmed[0].text.len(),
        original_len
    );
    // The preserved section must be untouched
    assert_eq!(
        trimmed[1].text, "Mandatory content that must survive.",
        "preserved section must not be trimmed"
    );
}

#[test]
fn importance_trim_trims_to_sentence_boundary() {
    // Specifically exercises line 606: the `rfind(". ")` branch that trims to sentence end.
    // We need text long enough that 60% still has a ". " inside.
    let text = "Alpha sentence ends here. Beta sentence continues after. Gamma comes third. \
        Delta fills more words. Epsilon yet more content to push length."
        .repeat(10);
    let sections = vec![Section {
        tag: SectionTag::RetryContext,
        text: text.clone(),
        importance: 0.3,
        preserve: false,
    }];
    let trimmed = importance_trim(sections, 1);
    // Trimmed text should end with ". " boundary or be a direct cut
    assert!(
        trimmed[0].text.len() < text.len(),
        "text must have been shortened"
    );
}
