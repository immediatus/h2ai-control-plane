use h2ai_context::compaction::{compact, CompactionConfig};

fn cfg(max_tokens: usize, keywords: Vec<&str>) -> CompactionConfig {
    CompactionConfig {
        max_tokens,
        preserve_keywords: keywords.into_iter().map(String::from).collect(),
    }
}

#[test]
fn compaction_preserves_content_under_budget() {
    let ctx = "short context";
    let cfg = CompactionConfig {
        max_tokens: 4096,
        preserve_keywords: vec![],
    };
    let out = compact(ctx, &cfg);
    assert_eq!(out, ctx);
}

#[test]
fn compaction_truncates_over_budget_and_preserves_keywords() {
    // Budget = 10 tokens ≈ 40 chars. Context is much longer.
    let long_ctx = format!("{} CONSTRAINT_KEYWORD {}", "A".repeat(200), "B".repeat(200));
    let cfg = CompactionConfig {
        max_tokens: 10,
        preserve_keywords: vec!["CONSTRAINT_KEYWORD".into()],
    };
    let out = compact(&long_ctx, &cfg);
    let token_estimate = out.len() / 4;
    assert!(
        token_estimate <= 15,
        "too long: {} tokens (len={})",
        token_estimate,
        out.len()
    );
    assert!(
        out.contains("CONSTRAINT_KEYWORD"),
        "keyword lost in compaction"
    );
}

#[test]
fn compaction_injects_missing_keyword_at_end() {
    let ctx = "some context without the keyword";
    let cfg = CompactionConfig {
        max_tokens: 4096,
        preserve_keywords: vec!["important_constraint".into()],
    };
    let out = compact(ctx, &cfg);
    assert!(
        out.contains("important_constraint"),
        "keyword should be injected when missing from context"
    );
}

#[test]
fn compaction_does_not_duplicate_existing_keyword() {
    let ctx = "context with EXISTING_KW already present";
    let cfg = CompactionConfig {
        max_tokens: 4096,
        preserve_keywords: vec!["EXISTING_KW".into()],
    };
    let out = compact(ctx, &cfg);
    assert_eq!(
        out.matches("EXISTING_KW").count(),
        1,
        "keyword should appear exactly once"
    );
}

#[test]
fn compaction_preserves_head_and_tail_dropping_middle() {
    let long_ctx = format!("HEAD_SENTINEL {} TAIL_SENTINEL", "X".repeat(2000));
    let config = CompactionConfig {
        max_tokens: 30,
        preserve_keywords: vec![],
    };
    let out = compact(&long_ctx, &config);
    assert!(
        out.contains("HEAD_SENTINEL"),
        "head sentinel must survive truncation"
    );
    assert!(
        out.contains("TAIL_SENTINEL"),
        "tail sentinel must survive truncation"
    );
    assert!(
        out.contains("[...compacted...]"),
        "truncation marker must appear"
    );
}

#[test]
fn compaction_max_tokens_zero_returns_empty_or_keywords_only() {
    // max_tokens=0 → body budget saturates to 0 → only keywords remain (or empty).
    let out_no_kw = compact("some long context here", &cfg(0, vec![]));
    assert!(
        out_no_kw.is_empty(),
        "zero budget with no keywords must return empty"
    );

    let out_kw = compact("some long context here", &cfg(0, vec!["ADR-001"]));
    assert!(
        out_kw.contains("ADR-001"),
        "zero budget must still emit the keyword suffix"
    );
}

#[test]
fn compaction_duplicate_keywords_in_list_appear_once() {
    // Duplicate keywords in preserve_keywords should not double-inject.
    let ctx = "context without either kw";
    let out = compact(ctx, &cfg(4096, vec!["KW_A", "KW_A", "KW_B"]));
    assert_eq!(
        out.matches("KW_A").count(),
        1,
        "duplicate keyword in preserve list must appear only once"
    );
    assert!(out.contains("KW_B"));
}

#[test]
fn compaction_empty_context_empty_keywords_returns_empty() {
    let out = compact("", &cfg(4096, vec![]));
    assert_eq!(out, "");
}

#[test]
fn compaction_empty_context_with_keyword_injects_keyword() {
    let out = compact("", &cfg(4096, vec!["INJECT_ME"]));
    assert!(out.contains("INJECT_ME"));
}

// ── Short context over budget: else branch (no middle gap) ───────────────────

#[test]
fn compaction_short_context_over_budget_no_marker() {
    // Use emoji (4 bytes per char) so byte-based token estimate exceeds max_tokens (7)
    // while char count (8) equals char_budget/2 (4 each) → tail_start == head_end → else branch.
    // emoji "😀" = 4 bytes; 8 emoji = 32 bytes → estimate_tokens = (32+3)/4 = 8 > max_tokens=7.
    // body_budget = 7 - 0 (no keywords) - 5 (marker) = 2; char_budget = 8; half = 4.
    // chars.len()=8; head_end = min(4,8)=4; tail_start = 8-4=4; 4 > 4 is false → else branch.
    let emoji_ctx = "😀😀😀😀😀😀😀😀"; // 8 emoji
    let out = compact(emoji_ctx, &cfg(7, vec![]));
    // No [...compacted...] marker — else branch just takes chars[..head_end]
    assert!(!out.contains("[...compacted...]"));
}

#[test]
fn compaction_truncated_body_missing_keyword_appends_it() {
    // Context is long enough to be truncated AND the keyword only appears in the middle
    // (which gets dropped), so post_suffix is non-empty — exercises line 82.
    let middle_kw = "MIDDLE_KEYWORD";
    let long_ctx = format!("{} {middle_kw} {}", "A".repeat(400), "B".repeat(400));
    let config = CompactionConfig {
        max_tokens: 50,
        preserve_keywords: vec![middle_kw.to_string()],
    };
    let out = compact(&long_ctx, &config);
    assert!(
        out.contains(middle_kw),
        "keyword dropped from middle must be re-injected as post-suffix"
    );
    assert!(
        out.contains("[...compacted...]"),
        "truncation marker must appear when middle is dropped"
    );
}
