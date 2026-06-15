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
