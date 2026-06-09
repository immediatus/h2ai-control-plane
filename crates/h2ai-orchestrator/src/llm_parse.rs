//! Shared LLM output parsing helpers used by thinking_loop and awareness_probe.
//! Moved from thinking_loop.rs to avoid duplication (GAP-F6 spec §llm_parse).

/// Remove ```json ... ``` or ``` ... ``` fences from LLM output.
pub(crate) fn strip_json_fences(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with("```") {
        let after_open = s.find('\n').map_or(s, |i| &s[i + 1..]);
        if let Some(close) = after_open.rfind("```") {
            return after_open[..close].trim();
        }
    }
    s
}

/// Find the first `[…]` JSON array in `s`, delegating boundary detection to serde_json.
/// Returns the slice covering the array, `None` if no valid array is found.
pub(crate) fn extract_first_json_array(s: &str) -> Option<&str> {
    let start = s.find('[')?;
    let tail = &s[start..];
    let mut stream = serde_json::Deserializer::from_str(tail).into_iter::<serde_json::Value>();
    stream.next()?.ok()?;
    Some(&tail[..stream.byte_offset()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_first_json_array_finds_array_with_preamble() {
        let s = "Ignore this.\n[1, [2, 3], \"]\"]";
        let extracted = extract_first_json_array(s);
        assert_eq!(extracted, Some(r#"[1, [2, 3], "]"]"#));
    }

    #[test]
    fn extract_first_json_array_returns_none_on_no_array() {
        assert_eq!(extract_first_json_array("no brackets here"), None);
    }

    #[test]
    fn strip_json_fences_removes_fences() {
        assert_eq!(strip_json_fences("```json\n[1,2]\n```"), "[1,2]");
    }

    #[test]
    fn strip_json_fences_passthrough_plain() {
        assert_eq!(strip_json_fences("[1,2]"), "[1,2]");
    }

    #[test]
    fn strip_json_fences_no_closing_fence_returns_original() {
        // No closing ``` → fallback returns the trimmed input unchanged.
        let input = "```json\n[1,2]";
        assert_eq!(strip_json_fences(input), input);
    }
}
