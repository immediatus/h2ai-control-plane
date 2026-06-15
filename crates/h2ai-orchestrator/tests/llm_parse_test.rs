use h2ai_orchestrator::llm_parse::{extract_first_json_array, strip_json_fences};

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
    let input = "```json\n[1,2]";
    assert_eq!(strip_json_fences(input), input);
}
