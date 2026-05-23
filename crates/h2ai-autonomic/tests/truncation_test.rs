use h2ai_autonomic::repair::{partial_max_chars, truncate_proposal};

#[test]
fn test_partial_max_chars_formula() {
    // model_max_tokens=32768, max_k=3, overhead=5.0 → 32768*4/(3+5) = 16384
    assert_eq!(partial_max_chars(32768, 3, 5.0), 16384);
    // model_max_tokens=4096, max_k=2, overhead=5.0 → 4096*4/(2+5) ≈ 2340
    assert_eq!(partial_max_chars(4096, 2, 5.0), 2340);
    // Minimum floor: very small model_max_tokens must not produce 0
    assert!(partial_max_chars(64, 3, 5.0) >= 32);
}

#[test]
fn test_truncate_short_proposal_unchanged() {
    let text = "short text";
    assert_eq!(truncate_proposal(text, 1500), text);
}

#[test]
fn test_truncate_long_proposal_snaps_to_newline() {
    // Put a newline at char 1400 (past the halfway mark of 750).
    let mut text = "a".repeat(1400);
    text.push('\n');
    text.push_str(&"b".repeat(1000));
    let result = truncate_proposal(&text, 1500);
    // Should cut at 1400 (the newline), not 1500.
    assert!(result.starts_with(&"a".repeat(1400)));
    assert!(result.contains("truncated at"));
    assert!(
        !result.contains("bbbb"),
        "should not include content past newline"
    );
}

#[test]
fn test_truncate_long_proposal_newline_too_early() {
    // Newline at char 100 — below halfway (750), must be ignored. Cut at 1500.
    let mut text = "a".repeat(100);
    text.push('\n');
    text.push_str(&"b".repeat(2000));
    let result = truncate_proposal(&text, 1500);
    // Should cut at 1500 chars, not 100.
    assert!(result.len() > 1100, "should keep most of the b block");
    assert!(result.contains("truncated at"));
}

#[test]
fn test_truncate_long_proposal_no_newline() {
    let text = "x".repeat(3000);
    let result = truncate_proposal(&text, 1500);
    assert!(result.starts_with(&"x".repeat(1500)));
    assert!(result.contains("truncated at 3000 chars"));
}

#[test]
fn test_truncate_suffix_contains_original_length() {
    let text = "z".repeat(3000);
    let result = truncate_proposal(&text, 1500);
    assert!(result.contains("3000"));
}

#[test]
fn test_truncate_multibyte_char_boundary() {
    // '⇒' is a 3-byte UTF-8 sequence (U+21D2).
    // 1499 ASCII bytes then '⇒' — chars().take(1500) safely takes 1499 a's + 1 '⇒'.
    let prefix = "a".repeat(1499);
    let suffix = "⇒".repeat(1000);
    let text = format!("{}{}", prefix, suffix);
    // Must not panic:
    let result = truncate_proposal(&text, 1500);
    assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    assert!(result.contains("truncated at"));
}

#[test]
fn test_truncate_respects_custom_max_chars() {
    let text = "y".repeat(500);
    // With max_chars=200, should truncate
    let result = truncate_proposal(&text, 200);
    assert!(result.contains("truncated at 500 chars"));
    // With max_chars=600, should not truncate
    assert_eq!(truncate_proposal(&text, 600), text);
}
