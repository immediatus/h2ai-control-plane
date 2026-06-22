use h2ai_orchestrator::gap_checkers::grounding::{
    CompositeGroundingJudge, FindingKind, GroundingFinding, GroundingJudge, HeuristicGroundingJudge,
};
use std::sync::Arc;

mockall::mock! {
    pub Judge {}
    #[async_trait::async_trait]
    impl GroundingJudge for Judge {
        async fn judge(&self, output: &str, spec: &str) -> Vec<GroundingFinding>;
    }
}

#[tokio::test]
async fn heuristic_entity_absent_from_spec_produces_finding() {
    let judge = HeuristicGroundingJudge;
    let findings = judge
        .judge("The system uses Kafka for streaming.", "Use Redis.")
        .await;
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].kind, FindingKind::Entity);
    assert!((findings[0].confidence - 0.8).abs() < 1e-9);
    assert!(findings[0].text.contains("Kafka") || findings[0].text == "Kafka");
}

#[tokio::test]
async fn heuristic_entity_present_in_spec_produces_no_finding() {
    let judge = HeuristicGroundingJudge;
    let findings = judge
        .judge("The system uses Redis.", "Use Redis as the caching layer.")
        .await;
    assert!(findings.is_empty());
}

#[tokio::test]
async fn heuristic_empty_output_produces_no_findings() {
    let judge = HeuristicGroundingJudge;
    let findings = judge.judge("", "Use Redis.").await;
    assert!(findings.is_empty());
}

#[tokio::test]
async fn heuristic_empty_spec_flags_all_arch_entities() {
    let judge = HeuristicGroundingJudge;
    let findings = judge.judge("The system uses Kafka and Redis.", "").await;
    assert!(!findings.is_empty());
}

#[tokio::test]
async fn composite_two_judges_distinct_findings_both_present() {
    let mut mock1 = MockJudge::new();
    mock1.expect_judge().returning(|_, _| {
        vec![GroundingFinding {
            text: "Kafka".into(),
            kind: FindingKind::Entity,
            reason: "not in spec".into(),
            confidence: 0.8,
        }]
    });
    let mut mock2 = MockJudge::new();
    mock2.expect_judge().returning(|_, _| {
        vec![GroundingFinding {
            text: "Elasticsearch".into(),
            kind: FindingKind::Entity,
            reason: "not in spec".into(),
            confidence: 0.9,
        }]
    });
    let composite = CompositeGroundingJudge::new(vec![Arc::new(mock1), Arc::new(mock2)]);
    let findings = composite.judge("output", "spec").await;
    assert_eq!(findings.len(), 2);
}

#[tokio::test]
async fn composite_deduplication_same_text_appears_once() {
    let mut mock1 = MockJudge::new();
    mock1.expect_judge().returning(|_, _| {
        vec![GroundingFinding {
            text: "Kafka".into(),
            kind: FindingKind::Entity,
            reason: "first judge".into(),
            confidence: 0.8,
        }]
    });
    let mut mock2 = MockJudge::new();
    mock2.expect_judge().returning(|_, _| {
        vec![GroundingFinding {
            text: "Kafka".into(),
            kind: FindingKind::Entity,
            reason: "second judge".into(),
            confidence: 0.9,
        }]
    });
    let composite = CompositeGroundingJudge::new(vec![Arc::new(mock1), Arc::new(mock2)]);
    let findings = composite.judge("output", "spec").await;
    assert_eq!(findings.len(), 1, "duplicate text must be deduplicated");
    assert_eq!(
        findings[0].reason, "first judge",
        "first judge wins on dedup"
    );
}

#[tokio::test]
async fn composite_zero_judges_returns_empty() {
    let composite = CompositeGroundingJudge::new(vec![]);
    let findings = composite.judge("output", "spec").await;
    assert!(findings.is_empty());
}

#[tokio::test]
async fn composite_single_judge_returns_same_as_direct() {
    let mut mock = MockJudge::new();
    mock.expect_judge().returning(|_, _| {
        vec![GroundingFinding {
            text: "Kafka".into(),
            kind: FindingKind::Entity,
            reason: "not in spec".into(),
            confidence: 0.8,
        }]
    });
    let composite = CompositeGroundingJudge::new(vec![Arc::new(mock)]);
    let findings = composite.judge("output", "spec").await;
    assert_eq!(findings.len(), 1);
}
