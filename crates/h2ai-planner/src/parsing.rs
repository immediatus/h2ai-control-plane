/// Strip any opening `` ```json `` or `` ``` `` fence line and trailing `` ``` ``,
/// then advance to the first `{` so `serde_json` can consume the rest.
///
/// `serde_json` stops at the end of the top-level object and ignores trailing
/// content, so we never need to locate the closing `}` ourselves.
pub fn extract_json(text: &str) -> &str {
    let s = text.trim();
    let s = s.strip_prefix("```").map_or(s, |after_fence| {
        let after_tag = after_fence.trim_start_matches(|c: char| c.is_alphanumeric());
        after_tag
            .trim_start_matches(['\n', '\r'])
            .trim_end_matches("```")
            .trim()
    });
    s.find('{').map_or(s, |start| &s[start..])
}
