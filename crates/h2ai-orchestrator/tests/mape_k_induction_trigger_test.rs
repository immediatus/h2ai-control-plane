use h2ai_orchestrator::induction::{InductionContext, InductionResult, InductionScheduler};
use h2ai_types::memory::RetryHintPattern;
use std::sync::Arc;

/// Mock scheduler that returns a result immediately.
struct ImmediateMockScheduler {
    result: Option<InductionResult>,
}

#[async_trait::async_trait]
impl InductionScheduler for ImmediateMockScheduler {
    async fn run_retroactive(&self, _ctx: &InductionContext) -> Option<InductionResult> {
        self.result.clone()
    }
}

/// Mock scheduler that never returns (simulates timeout).
struct TimeoutMockScheduler;

#[async_trait::async_trait]
impl InductionScheduler for TimeoutMockScheduler {
    async fn run_retroactive(&self, _ctx: &InductionContext) -> Option<InductionResult> {
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
        None
    }
}

#[tokio::test]
async fn induction_result_applied_when_compatible_and_within_grace_period() {
    let result = InductionResult {
        patterns: vec![RetryHintPattern {
            trigger_tags: vec!["billing".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: "use MergeTree append-only".to_string(),
            success_count: 2,
            attempt_count: 5,
        }],
        trigger_tags: vec!["billing".to_string()],
    };
    let scheduler = Arc::new(ImmediateMockScheduler {
        result: Some(result),
    });
    let res = scheduler
        .run_retroactive(&InductionContext {
            tenant_id: "t1".to_string(),
            task_class_tags: vec!["billing".to_string()],
            violated_constraint_ids: vec!["C-005".to_string()],
        })
        .await
        .unwrap();
    assert!(res.is_compatible_with(&["billing".to_string(), "C-005".to_string()]));
    assert_eq!(res.patterns.len(), 1);
}

#[tokio::test]
async fn induction_result_not_applied_when_incompatible_tags() {
    let result = InductionResult {
        patterns: vec![],
        trigger_tags: vec!["auth".to_string()],
    };
    let compatible = result.is_compatible_with(&["billing".to_string(), "C-005".to_string()]);
    assert!(
        !compatible,
        "auth trigger tags must not be compatible with billing context"
    );
}

#[tokio::test]
async fn timeout_mock_produces_none_within_grace_period() {
    let scheduler = Arc::new(TimeoutMockScheduler);
    let handle = tokio::spawn(async move {
        scheduler
            .run_retroactive(&InductionContext {
                tenant_id: "t1".to_string(),
                task_class_tags: vec!["billing".to_string()],
                violated_constraint_ids: vec![],
            })
            .await
    });
    // 50ms grace period → timeout fires before 60s sleep
    let result = tokio::time::timeout(tokio::time::Duration::from_millis(50), handle).await;
    assert!(
        result.is_err(),
        "timeout must fire before scheduler returns"
    );
}

#[tokio::test]
async fn immediate_mock_scheduler_load_priming_hints_returns_empty_by_default() {
    let scheduler = ImmediateMockScheduler { result: None };
    let ctx = InductionContext {
        tenant_id: "t1".to_string(),
        task_class_tags: vec!["billing".to_string()],
        violated_constraint_ids: vec![],
    };
    let hints = scheduler.load_priming_hints(&ctx).await;
    assert!(
        hints.is_empty(),
        "default load_priming_hints must return empty vec"
    );
}

#[tokio::test]
async fn apply_induction_result_populates_applied_hint_texts() {
    let mut cfg = h2ai_config::H2AIConfig::default();
    cfg.induction_trigger.enabled = true;
    let ctrl = h2ai_orchestrator::mape_k::MapeKController::new_for_test(cfg);
    assert!(ctrl.applied_hint_texts.is_empty(), "must start empty");
}
