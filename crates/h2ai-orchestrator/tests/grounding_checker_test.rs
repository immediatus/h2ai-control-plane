use h2ai_orchestrator::gap_checkers::grounding::{
    FindingKind, GroundingChecker, GroundingFinding, GroundingJudge,
};
use h2ai_orchestrator::gap_checkers::{
    GapCheckContext, GapChecker, GapKind, GapSeverity, GapSource,
};
use std::sync::Arc;

mockall::mock! {
    pub GroundingJudge {}
    #[async_trait::async_trait]
    impl GroundingJudge for GroundingJudge {
        async fn judge(&self, output: &str, spec: &str) -> Vec<GroundingFinding>;
    }
}

fn ctx() -> GapCheckContext {
    GapCheckContext {
        verified_provision_list: vec![],
        constraint_text: String::new(),
    }
}

fn finding(text: &str, kind: FindingKind, confidence: f64) -> GroundingFinding {
    GroundingFinding {
        text: text.into(),
        kind,
        reason: "test reason".into(),
        confidence,
    }
}

#[tokio::test]
async fn finding_above_min_confidence_emits_gap() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("Kafka", FindingKind::Entity, 0.8)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.7);
    let gaps = checker.check("output", &ctx()).await;
    assert_eq!(gaps.len(), 1);
}

#[tokio::test]
async fn finding_below_min_confidence_not_emitted() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("Kafka", FindingKind::Entity, 0.6)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.7);
    let gaps = checker.check("output", &ctx()).await;
    assert!(gaps.is_empty());
}

#[tokio::test]
async fn finding_exactly_at_min_confidence_is_included() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("Kafka", FindingKind::Entity, 0.7)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.7);
    let gaps = checker.check("output", &ctx()).await;
    assert_eq!(gaps.len(), 1);
}

#[tokio::test]
async fn high_confidence_maps_to_high_severity() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("X", FindingKind::Entity, 0.95)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.5);
    let gaps = checker.check("output", &ctx()).await;
    assert_eq!(gaps[0].severity, GapSeverity::High);
}

#[tokio::test]
async fn medium_confidence_maps_to_medium_severity() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("X", FindingKind::Entity, 0.75)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.5);
    let gaps = checker.check("output", &ctx()).await;
    assert_eq!(gaps[0].severity, GapSeverity::Medium);
}

#[tokio::test]
async fn low_confidence_maps_to_low_severity() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("X", FindingKind::Entity, 0.55)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.5);
    let gaps = checker.check("output", &ctx()).await;
    assert_eq!(gaps[0].severity, GapSeverity::Low);
}

#[tokio::test]
async fn gap_kind_is_ungrounded_content() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("X", FindingKind::Entity, 0.8)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.5);
    let gaps = checker.check("output", &ctx()).await;
    assert_eq!(gaps[0].kind, GapKind::UngroundedContent);
}

#[tokio::test]
async fn gap_source_is_grounding_check() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("X", FindingKind::Entity, 0.8)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.5);
    let gaps = checker.check("output", &ctx()).await;
    assert_eq!(gaps[0].source, GapSource::GroundingCheck);
}

#[tokio::test]
async fn gap_id_format_lowercased_underscored() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("Kafka Streams", FindingKind::Entity, 0.8)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.5);
    let gaps = checker.check("output", &ctx()).await;
    assert_eq!(gaps[0].id, "grounding:kafka_streams");
}

#[tokio::test]
async fn description_entity_prefix() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("Kafka", FindingKind::Entity, 0.8)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.5);
    let gaps = checker.check("output", &ctx()).await;
    assert!(gaps[0].description.starts_with("[entity]"));
}

#[tokio::test]
async fn description_claim_prefix() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge()
        .returning(|_, _| vec![finding("O(1) lookup", FindingKind::Claim, 0.8)]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.5);
    let gaps = checker.check("output", &ctx()).await;
    assert!(gaps[0].description.starts_with("[claim]"));
}

#[tokio::test]
async fn empty_judge_result_returns_empty_gaps() {
    let mut mock = MockGroundingJudge::new();
    mock.expect_judge().returning(|_, _| vec![]);
    let checker = GroundingChecker::new(Arc::new(mock), "spec".into(), 0.5);
    let gaps = checker.check("output", &ctx()).await;
    assert!(gaps.is_empty());
}
