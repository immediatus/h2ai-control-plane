use h2ai_adapters::a2a::{extract_proposal, next_backoff_interval, BackoffState, OutputFormat};

#[test]
fn extract_proposal_returns_raw_when_no_pattern_matches() {
    let text = "This is a plain response with no fences or preamble.";
    let result = extract_proposal(text, OutputFormat::Text).unwrap();
    assert_eq!(result, text.trim());
}

#[test]
fn extract_proposal_strips_markdown_fences_text() {
    let text = "```\nhello world\n```";
    let result = extract_proposal(text, OutputFormat::Text).unwrap();
    assert_eq!(result, "hello world");
}

#[test]
fn extract_proposal_strips_json_from_fences() {
    let text = "```json\n{\"key\": \"value\"}\n```";
    let result = extract_proposal(text, OutputFormat::Json).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["key"], "value");
}

#[test]
fn extract_proposal_strips_preamble() {
    let text = "Here is the solution:\nActual answer text.";
    let result = extract_proposal(text, OutputFormat::Text).unwrap();
    assert_eq!(result, "Actual answer text.");
}

#[test]
fn extract_proposal_json_uses_last_block_not_first() {
    let text = "```markdown\nSome plan description\n```\n\n```json\n{\"final\": true}\n```";
    let result = extract_proposal(text, OutputFormat::Json).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["final"], true);
}

#[test]
fn extract_proposal_direct_json_no_fences() {
    let text = "   {\"answer\": 42}   ";
    let result = extract_proposal(text, OutputFormat::Json).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["answer"], 42);
}

#[test]
fn backoff_starts_at_initial_interval() {
    let mut state = BackoffState::new(2000, 30_000);
    let dur = next_backoff_interval(&mut state);
    assert!(
        dur.as_millis() >= 1600 && dur.as_millis() <= 2400,
        "first backoff {} not in [1600, 2400]",
        dur.as_millis()
    );
}

#[test]
fn backoff_grows_and_caps_at_max() {
    let mut state = BackoffState::new(2000, 5_000);
    for _ in 0..10 {
        next_backoff_interval(&mut state);
    }
    let dur = next_backoff_interval(&mut state);
    assert!(
        dur.as_millis() <= 6000,
        "saturated backoff {} exceeds 6000ms",
        dur.as_millis()
    );
}

#[test]
fn auth_bearer_header_value() {
    use h2ai_adapters::a2a::{build_auth_header, AuthScheme};
    let hdr = build_auth_header(&AuthScheme::Bearer, "mytoken123").unwrap();
    assert_eq!(
        hdr,
        Some(("Authorization".to_string(), "Bearer mytoken123".to_string()))
    );
}

#[test]
fn auth_api_key_header_value() {
    use h2ai_adapters::a2a::{build_auth_header, AuthScheme};
    let hdr = build_auth_header(&AuthScheme::ApiKey, "mytoken123").unwrap();
    assert_eq!(
        hdr,
        Some(("X-API-Key".to_string(), "mytoken123".to_string()))
    );
}

#[test]
fn auth_none_returns_no_header() {
    use h2ai_adapters::a2a::{build_auth_header, AuthScheme};
    let result = build_auth_header(&AuthScheme::None, "").unwrap();
    assert!(result.is_none());
}

#[test]
fn a2a_adapter_implements_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<h2ai_adapters::a2a::A2aExplorerAdapter>();
}

/// `extract_proposal` with empty text → Err("empty artifact text")  (line 121-122)
#[test]
fn extract_proposal_returns_error_on_empty_text() {
    let result = extract_proposal("   ", OutputFormat::Text);
    assert!(result.is_err(), "empty text should return Err");
    assert_eq!(result.unwrap_err(), "empty artifact text");
}

/// `extract_proposal` with empty text in Json mode → Err
#[test]
fn extract_proposal_returns_error_on_empty_json_text() {
    let result = extract_proposal("", OutputFormat::Json);
    assert!(result.is_err(), "empty JSON text should return Err");
}

/// `extract_proposal`: JSON fences that contain non-JSON → falls through to preamble check (line 103-107)
#[test]
fn extract_proposal_json_with_non_json_fences_falls_through_to_raw() {
    let text = "```\nnot valid json at all\n```";
    // Format=Json, fences contain non-JSON → falls through to preamble, then raw
    let result = extract_proposal(text, OutputFormat::Json);
    // Should succeed and return the non-JSON content via preamble/raw path
    assert!(
        result.is_ok(),
        "non-JSON fenced content should not error: {result:?}"
    );
}

/// `extract_proposal` with preamble "output:" (lines 113-118)
#[test]
fn extract_proposal_strips_output_preamble() {
    let text = "output: some title\nThe actual content.";
    let result = extract_proposal(text, OutputFormat::Text).unwrap();
    assert_eq!(result, "The actual content.");
}

/// `extract_proposal` with preamble "result:" (lines 113-118)
#[test]
fn extract_proposal_strips_result_preamble() {
    let text = "result: something\nThe real result.";
    let result = extract_proposal(text, OutputFormat::Text).unwrap();
    assert_eq!(result, "The real result.");
}

/// `AuthScheme::parse` with unknown string → None
#[test]
fn auth_scheme_parse_unknown_returns_none() {
    use h2ai_adapters::a2a::AuthScheme;
    assert_eq!(AuthScheme::parse("unknown"), AuthScheme::None);
    assert_eq!(AuthScheme::parse(""), AuthScheme::None);
}

/// `extract_proposal`: preamble strips whole content leaving empty stripped — raw fallback (line 124).
/// The preamble regex matches the entire text (a preamble line + only whitespace after),
/// leaving `stripped` empty but `trimmed` non-empty → falls through to `Ok(trimmed.to_owned())`.
#[test]
fn extract_proposal_raw_fallback_when_preamble_consumes_all_content() {
    // Text has no fences, no direct JSON, and after preamble removal it becomes empty
    // but trimmed is non-empty. We trigger this via a non-preamble plain string where
    // stripped == trimmed (preamble regex doesn't match) and stripped is non-empty.
    // Actually line 124 is reached when stripped is EMPTY and trimmed is NOT empty.
    // That happens when preamble regex matches but the replacement yields only whitespace.
    // e.g. "Here is the solution:\n   " → stripped = "".trim() = "" but trimmed = "Here is..."
    let text = "Here is the solution:\n   ";
    let result = extract_proposal(text, OutputFormat::Text).unwrap();
    // trimmed = "Here is the solution:" (non-empty), stripped = "" → line 124
    assert!(
        !result.is_empty(),
        "should return raw trimmed text: {result}"
    );
}
