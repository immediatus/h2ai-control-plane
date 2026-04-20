use h2ai_context::compaction::{compact, CompactionConfig};

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
    assert!(token_estimate <= 15, "too long: {} tokens (len={})", token_estimate, out.len());
    assert!(out.contains("CONSTRAINT_KEYWORD"), "keyword lost in compaction");
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
    let cfg = CompactionConfig {
        max_tokens: 30,
        preserve_keywords: vec![],
    };
    let out = compact(&long_ctx, &cfg);
    assert!(out.contains("HEAD_SENTINEL"), "head sentinel must survive truncation");
    assert!(out.contains("TAIL_SENTINEL"), "tail sentinel must survive truncation");
    assert!(out.contains("[...compacted...]"), "truncation marker must appear");
}
