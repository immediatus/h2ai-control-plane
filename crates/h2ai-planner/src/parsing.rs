/// Strip any opening `` ```json `` or `` ``` `` fence line and trailing `` ``` ``,
/// then extract the first `{…}` JSON object using serde_json's own parser to locate
/// the matching `}`. Returns the exact object slice so callers can pass it directly
/// to `serde_json::from_str` without trailing-content errors.
pub fn extract_json(text: &str) -> &str {
    let s = text.trim();
    let s = s.strip_prefix("```").map_or(s, |after_fence| {
        let after_tag = after_fence.trim_start_matches(|c: char| c.is_alphanumeric());
        after_tag
            .trim_start_matches(['\n', '\r'])
            .trim_end_matches("```")
            .trim()
    });
    let Some(start) = s.find('{') else { return s };
    let tail = &s[start..];
    let mut stream = serde_json::Deserializer::from_str(tail).into_iter::<serde_json::Value>();
    match stream.next() {
        Some(Ok(_)) => &tail[..stream.byte_offset()],
        _ => tail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
