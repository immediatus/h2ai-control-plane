use h2ai_orchestrator::gap_checkers::grounding::{
    parse_grounding_response, FindingKind, GroundingJudge, LlmGroundingJudge,
};
use h2ai_test_utils::{failing_adapter, mock_adapter};
use std::sync::Arc;

#[test]
fn parse_valid_entity_finding() {
    let raw = r#"{"findings":[{"text":"Kafka","kind":"entity","reason":"not in spec","confidence":0.9}]}"#;
    let findings = parse_grounding_response(raw);
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].text, "Kafka");
    assert_eq!(findings[0].kind, FindingKind::Entity);
    assert_eq!(findings[0].reason, "not in spec");
    assert!((findings[0].confidence - 0.9).abs() < 1e-9);
}

#[test]
fn parse_valid_claim_finding() {
    let raw = r#"{"findings":[{"text":"O(1) complexity","kind":"claim","reason":"spec says O(n)","confidence":0.8}]}"#;
    let findings = parse_grounding_response(raw);
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].kind, FindingKind::Claim);
}

#[test]
fn parse_empty_findings_array() {
    let raw = r#"{"findings":[]}"#;
    let findings = parse_grounding_response(raw);
    assert!(findings.is_empty());
}

#[test]
fn parse_malformed_json_returns_empty() {
    let findings = parse_grounding_response("not json at all");
    assert!(findings.is_empty());
}

#[test]
fn parse_missing_findings_key_returns_empty() {
    let raw = r#"{"results":[]}"#;
    let findings = parse_grounding_response(raw);
    assert!(findings.is_empty());
}

#[test]
fn parse_confidence_below_half_excluded() {
    let raw = r#"{"findings":[{"text":"X","kind":"entity","reason":"low","confidence":0.49}]}"#;
    let findings = parse_grounding_response(raw);
    assert!(findings.is_empty(), "confidence < 0.5 must be excluded");
}

#[test]
fn parse_confidence_exactly_half_included() {
    let raw =
        r#"{"findings":[{"text":"X","kind":"entity","reason":"borderline","confidence":0.5}]}"#;
    let findings = parse_grounding_response(raw);
    assert_eq!(findings.len(), 1, "confidence == 0.5 must be included");
}

#[tokio::test]
async fn llm_judge_adapter_error_returns_empty() {
    let judge = LlmGroundingJudge::new(Arc::new(failing_adapter()), 1024, 0.2);
    let findings = judge.judge("output", "spec").await;
    assert!(findings.is_empty());
}

#[tokio::test]
async fn llm_judge_valid_response_returns_findings() {
    let json = r#"{"findings":[{"text":"Kafka","kind":"entity","reason":"not in spec","confidence":0.9}]}"#;
    let judge = LlmGroundingJudge::new(Arc::new(mock_adapter(json)), 1024, 0.2);
    let findings = judge.judge("output text", "spec text").await;
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].text, "Kafka");
}

#[tokio::test]
async fn llm_judge_multiple_findings_order_preserved() {
    let json = r#"{"findings":[
        {"text":"Kafka","kind":"entity","reason":"a","confidence":0.9},
        {"text":"O(1)","kind":"claim","reason":"b","confidence":0.7}
    ]}"#;
    let judge = LlmGroundingJudge::new(Arc::new(mock_adapter(json)), 1024, 0.2);
    let findings = judge.judge("output", "spec").await;
    assert_eq!(findings.len(), 2);
    assert_eq!(findings[0].text, "Kafka");
    assert_eq!(findings[1].text, "O(1)");
}
