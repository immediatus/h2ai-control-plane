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
    let hdr = build_auth_header(AuthScheme::Bearer, "mytoken123").unwrap();
    assert_eq!(
        hdr,
        Some(("Authorization".to_string(), "Bearer mytoken123".to_string()))
    );
}

#[test]
fn auth_api_key_header_value() {
    use h2ai_adapters::a2a::{build_auth_header, AuthScheme};
    let hdr = build_auth_header(AuthScheme::ApiKey, "mytoken123").unwrap();
    assert_eq!(
        hdr,
        Some(("X-API-Key".to_string(), "mytoken123".to_string()))
    );
}

#[test]
fn auth_none_returns_no_header() {
    use h2ai_adapters::a2a::{build_auth_header, AuthScheme};
    let result = build_auth_header(AuthScheme::None, "").unwrap();
    assert!(result.is_none());
}

#[test]
fn a2a_adapter_implements_debug() {
    fn _assert_debug<T: std::fmt::Debug>() {}
    _assert_debug::<h2ai_adapters::a2a::A2aExplorerAdapter>();
}
