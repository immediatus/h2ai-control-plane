//! Shared LLM output parsing helpers used by thinking_loop and awareness_probe.
//! Moved from thinking_loop.rs to avoid duplication (GAP-F6 spec §llm_parse).

/// Remove ```json ... ``` or ``` ... ``` fences from LLM output.
pub fn strip_json_fences(s: &str) -> &str {
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
pub fn extract_first_json_array(s: &str) -> Option<&str> {
    let start = s.find('[')?;
    let tail = &s[start..];
    let mut stream = serde_json::Deserializer::from_str(tail).into_iter::<serde_json::Value>();
    stream.next()?.ok()?;
    Some(&tail[..stream.byte_offset()])
}
