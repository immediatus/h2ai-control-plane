use h2ai_orchestrator::gap_checkers::coherence::CoherenceChecker;
use h2ai_orchestrator::gap_checkers::{GapCheckContext, GapChecker, GapKind, GapSource};
use h2ai_test_utils::mock_adapter;
use std::sync::Arc;

#[tokio::test]
async fn coherence_checker_returns_empty_on_no_conflicts_json() {
    let adapter = Arc::new(mock_adapter("[]"));
    let checker = CoherenceChecker::new(adapter, "medium".into());
    let ctx = GapCheckContext {
        verified_provision_list: vec!["Section 1".into()],
        constraint_text: "1. Limitation of liability cap...".into(),
    };
    let gaps = checker.check("Some document text.", &ctx).await;
    assert!(gaps.is_empty());
}

#[tokio::test]
async fn coherence_checker_returns_gap_for_high_severity_conflict() {
    let json = r#"[{"provision_a": "Section 1", "provision_b": "Section 3", "risk": "Contradictory scope", "severity": "high"}]"#;
    let adapter = Arc::new(mock_adapter(json));
    let checker = CoherenceChecker::new(adapter, "medium".into());
    let ctx = GapCheckContext {
        verified_provision_list: vec![],
        constraint_text: "".into(),
    };
    let gaps = checker.check("text", &ctx).await;
    assert_eq!(gaps.len(), 1);
    assert!(matches!(gaps[0].kind, GapKind::InterProvisionConflict));
    assert!(matches!(gaps[0].source, GapSource::CoherenceCheck));
}

#[tokio::test]
async fn coherence_checker_filters_below_min_severity() {
    let json = r#"[{"provision_a": "S1", "provision_b": "S2", "risk": "Minor wording", "severity": "low"}]"#;
    let adapter = Arc::new(mock_adapter(json));
    let checker = CoherenceChecker::new(adapter, "medium".into());
    let ctx = GapCheckContext {
        verified_provision_list: vec![],
        constraint_text: "".into(),
    };
    let gaps = checker.check("text", &ctx).await;
    assert!(gaps.is_empty(), "low severity filtered when min is medium");
}

#[tokio::test]
async fn coherence_checker_handles_malformed_llm_json_gracefully() {
    let adapter = Arc::new(mock_adapter("not json at all"));
    let checker = CoherenceChecker::new(adapter, "low".into());
    let ctx = GapCheckContext {
        verified_provision_list: vec![],
        constraint_text: "".into(),
    };
    let gaps = checker.check("text", &ctx).await;
    assert!(
        gaps.is_empty(),
        "malformed JSON yields no gaps, not a panic"
    );
}
