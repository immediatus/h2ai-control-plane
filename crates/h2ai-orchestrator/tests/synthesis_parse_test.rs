use h2ai_orchestrator::thinking_loop::parse_synthesis_from_markdown;

fn full_synthesis_doc() -> &'static str {
    r#"
## Shared Understanding
Both perspectives agree that the authentication layer needs explicit session invalidation on logout. The service must not rely on token expiry alone. Rate limiting should be applied at the gateway, not the microservice.

## Unresolved Tensions
- Whether to use opaque tokens (simpler revocation) or JWTs (stateless validation)
- Whether rate limits should be per-user or per-IP

## Coverage Assessment
**Score:** 0.73
The combined view covers auth and rate limiting well but leaves storage encryption unaddressed.
"#
}

#[test]
fn shared_understanding_extracted() {
    let report = parse_synthesis_from_markdown(full_synthesis_doc());
    assert!(
        report.shared_understanding.contains("session invalidation"),
        "shared_understanding must contain filled text"
    );
}

#[test]
fn tensions_extracted_as_vec() {
    let report = parse_synthesis_from_markdown(full_synthesis_doc());
    assert_eq!(report.tensions.len(), 2, "must extract exactly 2 tensions");
    assert!(report.tensions[0].contains("opaque tokens"));
    assert!(report.tensions[1].contains("rate limits"));
}

#[test]
fn coverage_score_parsed() {
    let report = parse_synthesis_from_markdown(full_synthesis_doc());
    assert!(
        (report.coverage_score - 0.73).abs() < 0.01,
        "coverage_score must be 0.73, got {}",
        report.coverage_score
    );
}

#[test]
fn missing_tensions_section_gives_empty_vec() {
    let doc = r#"
## Shared Understanding
Only one perspective, nothing to conflict.

## Coverage Assessment
**Score:** 0.50
Fully covered.
"#;
    let report = parse_synthesis_from_markdown(doc);
    assert!(report.tensions.is_empty());
    assert!((report.coverage_score - 0.50).abs() < 0.01);
}

#[test]
fn plain_text_fallback_when_no_headers_found() {
    // Model may ignore the template entirely. Fallback: treat whole text as shared_understanding.
    let report = parse_synthesis_from_markdown("This is some free-form synthesis output.");
    assert!(report.shared_understanding.contains("free-form synthesis"));
    assert_eq!(report.coverage_score, 0.5, "fallback coverage_score must be 0.5");
}
