/// Strip any opening `` ```json `` or `` ``` `` fence line and trailing `` ``` ``,
/// then advance to the first `{` so serde_json can consume the rest.
///
/// serde_json stops at the end of the top-level object and ignores trailing
/// content, so we never need to locate the closing `}` ourselves.
pub(crate) fn extract_json(text: &str) -> &str {
    let s = text.trim();
    let s = if let Some(after_fence) = s.strip_prefix("```") {
        let after_tag = after_fence.trim_start_matches(|c: char| c.is_alphanumeric());
        after_tag
            .trim_start_matches(['\n', '\r'])
            .trim_end_matches("```")
            .trim()
    } else {
        s
    };
    if let Some(start) = s.find('{') {
        &s[start..]
    } else {
        s
    }
}
