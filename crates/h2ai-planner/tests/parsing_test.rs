use h2ai_planner::parsing::extract_json;

#[test]
fn extract_json_clean_object() {
    let s = r#"{"approved": true, "reason": "ok"}"#;
    assert_eq!(extract_json(s), s);
}

#[test]
fn extract_json_strips_fence_and_trailing_text() {
    let s = "```json\n{\"approved\": true}\n```\nSome trailing note.";
    assert_eq!(extract_json(s), r#"{"approved": true}"#);
}

#[test]
fn extract_json_strips_preamble_prose() {
    let s = "Here is the JSON:\n{\"approved\": false, \"reason\": \"bad\"}";
    assert_eq!(extract_json(s), r#"{"approved": false, "reason": "bad"}"#);
}

#[test]
fn extract_json_handles_nested_braces_in_strings() {
    // A '}' inside a JSON string must not terminate the object early.
    let s = r#"{"key": "value with } inside"}"#;
    assert_eq!(extract_json(s), s);
}

#[test]
fn extract_json_no_object_returns_input() {
    let s = "no json here";
    assert_eq!(extract_json(s), s);
}

#[test]
fn extract_json_invalid_json_after_brace_returns_tail() {
    // `{` found but the text is not valid JSON → serde_json returns Some(Err(_)) → `_ => tail`
    let s = "prefix { invalid json";
    assert_eq!(extract_json(s), "{ invalid json");
}
